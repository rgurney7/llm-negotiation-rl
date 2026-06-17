import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../ppo"))

import argparse
import csv
from collections import defaultdict
import torch
import pandas as pd
from grpo_agent import GrpoAgent
from ppo_env import NegotiationEnv
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
from shared.rollout_config import get_ppo_buyer


CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(SCRIPT_DIR, "checkpoints"))
# Held-out validation split: 153 uuids, disjoint from the 526 training uuids
# (enforced by data/check_splits.py). This documents the intended held-out
# design, but the file is not runnable as committed: the enrichment that builds
# the seller_prompt/buyer_prompt columns NegotiationEnv consumes was not persisted.
# The buyer is sampled from the production roster (get_ppo_buyer) so the eval
# opponent matches the training opponents, replacing the earlier off-roster
# Phi-4-mini buyer and the in-code SCENARIOS x PERSONAS grid.
DATA_PATH = os.path.join(SCRIPT_DIR, "../data/craigslist_eval.csv")
LOG_PATH = os.path.join(SCRIPT_DIR, "eval_log.csv")


def load_trained_agent(update_num):
    agent = GrpoAgent()
    lora_path = os.path.join(CHECKPOINT_DIR, f"lora_update_{update_num:04d}", "adapter_model.safetensors")
    set_peft_model_state_dict(agent.llm, load_file(lora_path, device=str(agent.device)))
    print(f"Loaded LoRA weights from {lora_path}", flush=True)
    agent.eval()
    return agent


def load_base_agent():
    agent = GrpoAgent()
    agent.eval()
    print("Loaded base model (no trained LoRA weights)", flush=True)
    return agent


def run_eval(agent, label, eval_rows, df, log_rows):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n", flush=True)

    for i, (idx, row) in enumerate(eval_rows.iterrows()):
        buyer = get_ppo_buyer()
        env = NegotiationEnv(buyer, df.iloc[[df.index.get_loc(idx)]])
        obs, _ = env.reset()
        seller_prompt = env.get_seller_prompt()

        for step in range(env.max_steps):
            full_prompt = seller_prompt + "\n\n" + obs
            _, generated_ids = agent.generate(full_prompt)
            agent_text = agent.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            obs, reward, terminated, truncated, info = env.step(agent_text)
            if terminated or truncated:
                break

        agreed = info.get("agreed_price")

        print(f"[{i+1}/{len(eval_rows)}] {row.get('item_title', 'Unknown')[:40]:40s}  "
              f"listing=${row['listing_price']:.0f}  agreed={agreed}  reward={reward:.3f}  buyer={buyer.name}",
              flush=True)

        log_rows.append({
            "agent": label,
            "row_idx": idx,
            "item_title": row.get("item_title", "Unknown"),
            "listing_price": row["listing_price"],
            "agreed_price": agreed,
            "reward": round(reward, 3),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=int, required=True, help="Checkpoint update number to evaluate (e.g. 20)")
    parser.add_argument("--rows", type=int, default=50, help="Number of data rows to evaluate on")
    parser.add_argument("--skip-base", action="store_true", help="Skip base model benchmark")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    df = pd.read_csv(DATA_PATH)
    eval_rows = df.sample(n=min(args.rows, len(df)), random_state=42)
    print(f"Evaluating on {len(eval_rows)} rows", flush=True)

    log_rows = []

    # Base model benchmark
    if not args.skip_base:
        base_agent = load_base_agent()
        run_eval(base_agent, "base", eval_rows, df, log_rows)
        del base_agent
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Trained agent
    trained_agent = load_trained_agent(args.update)
    run_eval(trained_agent, f"grpo_{args.update}", eval_rows, df, log_rows)

    # Write CSV
    fields = ["agent", "row_idx", "item_title", "listing_price", "agreed_price", "reward"]
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(log_rows)
    print(f"\nResults saved to {LOG_PATH}", flush=True)

    # Summary table
    by_agent = defaultdict(list)
    for row in log_rows:
        by_agent[row["agent"]].append(row)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for agent_name, rows in by_agent.items():
        rewards = [r["reward"] for r in rows]
        deals = sum(1 for r in rows if r["agreed_price"] is not None)
        mean_r = sum(rewards) / len(rewards)
        print(f"  {agent_name:20s}  deal_rate={deals}/{len(rows)}  mean_reward={mean_r:.3f}")
    print()


if __name__ == "__main__":
    main()
