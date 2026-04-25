import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torch.nn as nn
import pandas as pd
import csv
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

NUM_EPOCHS    = 3
BATCH_SIZE    = 4
LEARNING_RATE = 5e-5
MAX_SEQ_LEN   = 1500
SAVE_EVERY    = 1

SYNTHETIC_PATH = os.path.join(SCRIPT_DIR, "../data/synthetic_data.csv")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(SCRIPT_DIR, "checkpoints/sft"))
LOG_PATH       = os.environ.get("LOG_PATH", os.path.join(SCRIPT_DIR, "sft_training_log.csv"))


def extract_system_prompt(final_obs):
    idx = final_obs.find("Negotiation so far:")
    return final_obs.strip() if idx == -1 else final_obs[:idx].strip()


def parse_transcript(buyer_transcript):
    turns = []
    role = None
    lines = []
    for line in buyer_transcript.split("\n"):
        if line.startswith("Seller: "):
            if role is not None:
                turns.append((role, "\n".join(lines)))
            role, lines = "Seller", [line[len("Seller: "):]]
        elif line.startswith("Buyer: "):
            if role is not None:
                turns.append((role, "\n".join(lines)))
            role, lines = "Buyer", [line[len("Buyer: "):]]
        elif role is not None:
            lines.append(line)
    if role is not None:
        turns.append((role, "\n".join(lines)))
    return turns


def build_sft_examples(df):
    examples = []
    for _, row in df.iterrows():
        if pd.isna(row["buyer_transcript"]):
            continue
        system_prompt = extract_system_prompt(row["final_obs"])
        history_lines = []
        for role, msg in parse_transcript(row["buyer_transcript"]):
            if role == "Seller":
                if history_lines:
                    obs = f"{system_prompt}\n\nNegotiation so far:\n" + "\n".join(history_lines) + "\n\nYour next message:"
                else:
                    obs = f"{system_prompt}\n\nYour next message:"
                examples.append({"obs": obs, "target": msg})
            label = "You" if role == "Seller" else "Buyer"
            history_lines.append(f"{label}: {msg}")
    return examples


def tokenize_example(tokenizer, obs, target):
    messages = [{"role": "user", "content": obs}]
    context_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    context_ids = tokenizer(context_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids = tokenizer(context_text + target + tokenizer.eos_token, return_tensors="pt", add_special_tokens=False).input_ids[0]

    if len(full_ids) > MAX_SEQ_LEN:
        full_ids = full_ids[:MAX_SEQ_LEN]

    loss_mask = torch.zeros(len(full_ids), dtype=torch.float32)
    loss_mask[len(context_ids):] = 1.0
    return full_ids, loss_mask


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading synthetic data...", flush=True)
    df = pd.read_csv(SYNTHETIC_PATH)
    df = df[(df["reward"] > 0) & (df["buyer_transcript"].notna())]
    print(f"Positive-reward rows with transcripts: {len(df)}", flush=True)

    examples = build_sft_examples(df)
    print(f"Total SFT training examples (seller turns): {len(examples)}", flush=True)

    print("Loading model...", flush=True)
    model_id = "Qwen/Qwen3.5-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    for param in model.parameters():
        param.requires_grad = False

    model = get_peft_model(model, lora_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Tokenizing examples...", flush=True)
    tokenized = [tokenize_example(tokenizer, ex["obs"], ex["target"]) for ex in examples]
    print(f"Tokenized {len(tokenized)} examples", flush=True)

    log_fields = ["epoch", "step", "loss", "avg_loss"]
    with open(LOG_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===", flush=True)

        indices = list(range(len(tokenized)))
        random.shuffle(indices)

        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for i, idx in enumerate(indices):
            input_ids, loss_mask = tokenized[idx]
            input_ids = input_ids.to(device)
            loss_mask = loss_mask.to(device)

            logits = model(input_ids.unsqueeze(0)).logits[0]

            shift_logits = logits[:-1]
            shift_labels = input_ids[1:]
            shift_mask = loss_mask[1:]

            per_token_loss = nn.functional.cross_entropy(shift_logits, shift_labels, reduction="none")
            masked_loss = (per_token_loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)

            (masked_loss / BATCH_SIZE).backward()

            epoch_loss += masked_loss.item()
            epoch_steps += 1
            global_step += 1

            if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                avg_loss = epoch_loss / epoch_steps
                if (i + 1) % (BATCH_SIZE * 10) == 0 or (i + 1) == len(indices):
                    print(f"  step {i+1}/{len(indices)}  loss: {masked_loss.item():.4f}  avg: {avg_loss:.4f}", flush=True)

                with open(LOG_PATH, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=log_fields).writerow({
                        "epoch": epoch, "step": global_step,
                        "loss": round(masked_loss.item(), 4),
                        "avg_loss": round(avg_loss, 4),
                    })

        print(f"  Epoch {epoch} complete — avg loss: {epoch_loss / epoch_steps:.4f}", flush=True)

        if epoch % SAVE_EVERY == 0:
            path = os.path.join(CHECKPOINT_DIR, f"lora_epoch_{epoch:02d}")
            model.save_pretrained(path)
            print(f"  Saved checkpoint: {path}", flush=True)

    final_path = os.path.join(CHECKPOINT_DIR, f"lora_epoch_{NUM_EPOCHS:02d}")
    if not os.path.exists(final_path):
        model.save_pretrained(final_path)
        print(f"  Saved final checkpoint: {final_path}", flush=True)

    print("\nSFT training complete.", flush=True)


if __name__ == "__main__":
    main()

    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        print("Training complete. Stopping pod...", flush=True)
        os.system(f"runpodctl stop pod {pod_id}")
