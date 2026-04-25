import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import torch
import torch.optim as optim
import pandas as pd
import csv
from grpo_agent import GrpoAgent
from grpo_env import NegotiationEnv
from shared.rollout_config import get_grpo_buyer
from shared.local_buyer import LocalBuyer, pick_buyer
from shared.config import SCENARIOS, PERSONAS, make_buyer_prompt
from shared.reward import price_reward
from safetensors.torch import load_file
from peft import set_peft_model_state_dict

NUM_UPDATES   = 50
EPISODES      = 8
G             = 8
N_EPOCHS      = 4
LEARNING_RATE = 5e-5
CLIP_EPS      = 0.2
ENTROPY_COEF  = 0.01
KL_COEF       = 0.05
SAVE_EVERY    = 5

CHECKPOINT_DIR   = os.environ.get("CHECKPOINT_DIR", os.path.join(SCRIPT_DIR, "checkpoints"))
LOG_PATH         = os.environ.get("LOG_PATH", os.path.join(SCRIPT_DIR, "training_log.csv"))
TRANSCRIPT_PATH  = os.environ.get("TRANSCRIPT_PATH", os.path.join(SCRIPT_DIR, "transcripts.csv"))

CRAIGSLIST_PATH = os.path.join(SCRIPT_DIR, "../data/craigslist_grpo_enriched.csv")
SYNTHETIC_PATH  = os.path.join(SCRIPT_DIR, "../data/synthetic_data.csv")

SCENARIOS_BY_ITEM = {s["item"]: s for s in SCENARIOS}


def load_checkpoint(agent):
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.txt")
    if not os.path.exists(latest_path):
        return 0
    with open(latest_path) as f:
        update = int(f.read().strip())
    lora_path = os.path.join(CHECKPOINT_DIR, f"lora_update_{update:04d}", "adapter_model.safetensors")
    set_peft_model_state_dict(agent.llm, load_file(lora_path, device=str(agent.device)))
    print(f"Resumed from update {update}", flush=True)
    return update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data with scenario-based reward")
    parser.add_argument("--fresh", action="store_true", help="Start a fresh run, ignoring existing checkpoints")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if args.fresh and os.path.exists(os.path.join(CHECKPOINT_DIR, "latest.txt")):
        os.remove(os.path.join(CHECKPOINT_DIR, "latest.txt"))
        print("Starting fresh run.", flush=True)

    print("Loading env and data...", flush=True)
    env = NegotiationEnv()

    if args.synthetic:
        env_df = pd.read_csv(SYNTHETIC_PATH)
        print(f"Mode: SYNTHETIC — {len(env_df)} rows, scenario-based reward", flush=True)
    else:
        env_df = pd.read_csv(CRAIGSLIST_PATH)
        print(f"Mode: CRAIGSLIST — {len(env_df)} rows, with buyer API calls", flush=True)

    print("Loading model...", flush=True)
    agent = GrpoAgent()
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    log_fields = ["update", "epoch", "policy_loss", "mean_kl", "mean_reward", "mean_entropy"]
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    tx_fields = ["update", "ep", "rollout", "reward", "item", "seller_msg"]
    if not os.path.exists(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=tx_fields).writeheader()

    local_buyer = LocalBuyer(agent)
    start_update = load_checkpoint(agent)

    for update in range(start_update + 1, NUM_UPDATES + 1):
        print(f"\n--- Update {update}/{NUM_UPDATES} ---", flush=True)

        groups = []
        update_rewards = []

        for ep in range(EPISODES):
            row = env_df.sample(1).iloc[0]

            if args.synthetic:
                obs = row["truncate_obs"]
                base_transcript = row["buyer_transcript"]
                scenario = SCENARIOS_BY_ITEM[row["scenario"]]
                buyer = pick_buyer(local_buyer, get_grpo_buyer)
                buyer_prompt = make_buyer_prompt(PERSONAS[int(row["persona_idx"])], scenario)
            else:
                obs = row["grpo_obs"]
                base_transcript = row["buyer_transcript"]
                buyer = pick_buyer(local_buyer, get_grpo_buyer)
                buyer_prompt = row["buyer_prompt"]
                listing_price = float(row["listing_price"])

            group = {"ctx": [], "gen": [], "old_lps": [], "ref_lps": [], "rewards": []}

            seller_msgs = []
            for _ in range(G):
                with torch.no_grad():
                    ctx, gen = agent.generate(obs)
                    old_lps, _ = agent.evaluate(ctx, gen)
                    ref_lps = agent.evaluate_ref(ctx, gen)
                group["ctx"].append(ctx)
                group["gen"].append(gen)
                group["old_lps"].append(old_lps.sum())
                group["ref_lps"].append(ref_lps)
                seller_msgs.append(agent.tokenizer.decode(gen[0], skip_special_tokens=True))

            transcripts = [base_transcript] * G
            buyer_replies, final_transcripts = env.buyer_replies_concurrent(
                buyer, buyer_prompt, transcripts, seller_msgs
            )

            if args.synthetic:
                rewards = [
                    price_reward(env.eval_model.extract_price(ft), scenario["seller_reserve"], scenario["buyer_max"])
                    for ft in final_transcripts
                ]
            else:
                rewards = env.judge_concurrent(final_transcripts, listing_price)

            group["rewards"] = rewards
            update_rewards.extend(rewards)
            groups.append(group)
            print(f"  ep {ep+1}: rewards {[round(r, 2) for r in rewards]}", flush=True)

            best_g = max(range(G), key=lambda g: rewards[g])
            worst_g = min(range(G), key=lambda g: rewards[g])
            item_name = row.get("scenario", row.get("item_title", ""))
            for label, g in [("best", best_g), ("worst", worst_g)]:
                with open(TRANSCRIPT_PATH, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=tx_fields).writerow({
                        "update": update, "ep": ep + 1, "rollout": f"{g+1}_{label}",
                        "reward": round(rewards[g], 3),
                        "item": item_name,
                        "seller_msg": seller_msgs[g],
                    })

            if ep == 0:
                print(f"\n  --- Sample (ep 1, best rollout {best_g+1}, reward={rewards[best_g]:.2f}) ---")
                print(f"  [Seller]: {seller_msgs[best_g][:200]}")
                print(f"  ---\n", flush=True)

        mean_reward = sum(update_rewards) / len(update_rewards)
        print(f"  mean reward: {mean_reward:.3f}", flush=True)

        for epoch in range(N_EPOCHS):
            optimizer.zero_grad()
            total_loss = 0.0
            total_kl = 0.0
            total_entropy = 0.0
            n_samples = EPISODES * G

            for group in groups:
                rewards = torch.tensor(group["rewards"], dtype=torch.float32)
                mean_r = rewards.mean()
                std_r = rewards.std() if G > 1 else torch.tensor(1.0)

                for g in range(G):
                    adv = (rewards[g] - mean_r) / (std_r + 1e-8)

                    new_lps, ent = agent.evaluate(group["ctx"][g], group["gen"][g])
                    ratio = torch.exp(new_lps.sum() - group["old_lps"][g])
                    ploss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv)
                    tr = torch.exp(new_lps - group["ref_lps"][g])
                    kl = (tr - torch.log(tr) - 1).sum()
                    ((ploss + KL_COEF * kl - ent.mean() * ENTROPY_COEF) / n_samples).backward()

                    total_loss += ploss.item()
                    total_kl += kl.item()
                    total_entropy += ent.mean().item()

            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            avg_loss = total_loss / n_samples
            avg_kl = total_kl / n_samples
            avg_ent = total_entropy / n_samples
            print(f"  epoch {epoch+1} loss: {avg_loss:.4f}  kl: {avg_kl:.4f}  ent: {avg_ent:.3f}", flush=True)

            with open(LOG_PATH, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=log_fields).writerow({
                    "update": update, "epoch": epoch + 1,
                    "policy_loss": round(avg_loss, 4),
                    "mean_kl": round(avg_kl, 4),
                    "mean_reward": round(mean_reward, 3),
                    "mean_entropy": round(avg_ent, 3),
                })

        if update == 1 or update % SAVE_EVERY == 0:
            path = os.path.join(CHECKPOINT_DIR, f"lora_update_{update:04d}")
            agent.save(path)
            with open(os.path.join(CHECKPOINT_DIR, "latest.txt"), "w") as f:
                f.write(str(update))
            print(f"  saved checkpoint: {path}", flush=True)

    path = os.path.join(CHECKPOINT_DIR, f"lora_update_{NUM_UPDATES:04d}")
    if not os.path.exists(path):
        agent.save(path)
        with open(os.path.join(CHECKPOINT_DIR, "latest.txt"), "w") as f:
            f.write(str(NUM_UPDATES))
        print(f"  saved final checkpoint: {path}", flush=True)


if __name__ == "__main__":
    main()

    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        print("Training complete. Stopping pod...", flush=True)
        os.system(f"runpodctl stop pod {pod_id}")
