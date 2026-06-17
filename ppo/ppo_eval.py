import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

import argparse
import csv
import torch
import pandas as pd
from ppo_agent import NegotiationAgent
from ppo_env import NegotiationEnv
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
from shared.rollout_config import get_ppo_buyer

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(SCRIPT_DIR, "checkpoints"))
# Held-out validation split: 153 uuids, disjoint from the 526 training uuids
# (enforced by data/check_splits.py). This documents the intended held-out
# design, but the file is not runnable as committed: the enrichment that builds
# the seller_prompt/buyer_prompt columns NegotiationEnv consumes was not persisted.
DATA_PATH = os.path.join(SCRIPT_DIR, "../data/craigslist_eval.csv")
OUT_DIR = os.environ.get("OUT_DIR", SCRIPT_DIR)


def load_trained_agent(update_num: int) -> NegotiationAgent:
    agent = NegotiationAgent()
    lora_path = f"{CHECKPOINT_DIR}/lora_update_{update_num:04d}.safetensors"
    lora_weights = load_file(lora_path, device=str(agent.device))
    set_peft_model_state_dict(agent.llm, lora_weights)
    print(f"Loaded LoRA weights from {lora_path}", flush=True)
    agent.eval()
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=int, required=True, help="Checkpoint update number to evaluate")
    parser.add_argument("--rows", type=int, default=50, help="Number of data rows to evaluate on")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    df = pd.read_csv(DATA_PATH)
    eval_rows = df.sample(n=min(args.rows, len(df)), random_state=42)
    print(f"Evaluating on {len(eval_rows)} rows", flush=True)

    print("Loading agent...", flush=True)
    agent = load_trained_agent(args.update)

    log_path = f"{OUT_DIR}/eval_log_update_{args.update:04d}.csv"
    log_fields = ["row_idx", "item_title", "listing_price", "agreed_price", "reward", "transcript"]

    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    rewards = []
    deals = 0

    for i, (idx, row) in enumerate(eval_rows.iterrows()):
        buyer = get_ppo_buyer()
        env = NegotiationEnv(buyer, df.iloc[[df.index.get_loc(idx)]])
        obs, _ = env.reset()
        seller_prompt = env.get_seller_prompt()

        for step in range(env.max_steps):
            _, generated_ids = agent.generate(obs, seller_prompt)
            agent_text = agent.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            obs, reward, terminated, truncated, info = env.step(agent_text)
            if terminated or truncated:
                break

        transcript = env._build_transcript()
        agreed = info.get("agreed_price")
        rewards.append(reward)
        if agreed is not None:
            deals += 1

        print(f"[{i+1}/{len(eval_rows)}] {row.get('item_title', 'Unknown')[:40]:40s}  "
              f"listing=${row['listing_price']:.0f}  agreed={agreed}  reward={reward:.3f}  buyer={buyer.name}",
              flush=True)

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "row_idx": idx,
                "item_title": row.get("item_title", "Unknown"),
                "listing_price": row["listing_price"],
                "agreed_price": agreed,
                "reward": round(reward, 4),
                "transcript": transcript,
            })

    mean_reward = sum(rewards) / len(rewards)
    deal_rate = deals / len(rewards)
    print(f"\n--- Results (update {args.update}) ---", flush=True)
    print(f"  Rows: {len(rewards)}  Deals: {deals}/{len(rewards)} ({deal_rate:.0%})  "
          f"Mean reward: {mean_reward:.3f}", flush=True)
    print(f"  Log saved to {log_path}", flush=True)


if __name__ == "__main__":
    main()
