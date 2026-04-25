import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

import argparse
import csv
import time
import torch
import torch.optim as optim
import pandas as pd
from ppo_env import NegotiationEnv
from ppo_agent import NegotiationAgent
from safetensors.torch import load_file, save_file
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from shared.rollout_config import get_ppo_buyer
from shared.local_buyer import LocalBuyer, pick_buyer
from shared.gae import compute_gae

NUM_UPDATES = 150
NUM_EPS = 8
NUM_EPOCHS = 3
NUM_STEPS = 8
LORA_LR = 1e-5
CRITIC_LR = 3e-4
ENTROPY_COEFF = 0.003
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
CHECKPOINT_EVERY = 5
VOLUME = "/workspace/ppo"
OUT_DIR = VOLUME if os.path.exists("/workspace") else "."
LOG_FILE = f"{OUT_DIR}/training_log.csv"
TRANSCRIPT_FILE = f"{OUT_DIR}/transcripts.csv"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/craigslist_grpo_enriched.csv")


def save_checkpoint(agent, update):
    ckpt_dir = f"{OUT_DIR}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    save_file(get_peft_model_state_dict(agent.llm), f"{ckpt_dir}/lora_update_{update:04d}.safetensors")
    torch.save(agent.critic.state_dict(), f"{ckpt_dir}/critic_update_{update:04d}.pt")
    with open(f"{ckpt_dir}/latest.txt", "w") as f:
        f.write(str(update))
    print(f"  Checkpoint saved: update_{update:04d}", flush=True)


def load_checkpoint(agent):
    ckpt_dir = f"{OUT_DIR}/checkpoints"
    if not os.path.exists(f"{ckpt_dir}/latest.txt"):
        return 0
    with open(f"{ckpt_dir}/latest.txt") as f:
        update = int(f.read().strip())
    set_peft_model_state_dict(agent.llm, load_file(f"{ckpt_dir}/lora_update_{update:04d}.safetensors"))
    agent.critic.load_state_dict(torch.load(f"{ckpt_dir}/critic_update_{update:04d}.pt"))
    print(f"Resumed from update {update}", flush=True)
    return update


def log_update(update, metrics):
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--init-lora", type=str, default=None,
                        help="Path to GRPO LoRA checkpoint to initialize from (e.g. /workspace/grpo_v4_checkpoints/lora_update_0040)")
    args = parser.parse_args()

    if args.fresh and os.path.exists(f"{OUT_DIR}/checkpoints/latest.txt"):
        os.remove(f"{OUT_DIR}/checkpoints/latest.txt")
        print("Starting fresh run.", flush=True)

    print("Loading data...", flush=True)
    craigslist_df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(craigslist_df)} training rows", flush=True)
    print("Loading model...", flush=True)
    agent = NegotiationAgent()
    print(f"Model loaded on {agent.device}", flush=True)

    # Initialize LoRA from GRPO pretrained checkpoint (before loading PPO checkpoint)
    if args.init_lora:
        lora_path = os.path.join(args.init_lora, "adapter_model.safetensors")
        lora_weights = load_file(lora_path, device=str(agent.device))
        set_peft_model_state_dict(agent.llm, lora_weights)
        print(f"Initialized LoRA from GRPO checkpoint: {lora_path}", flush=True)

    optimizer = optim.Adam([
        {"params": agent.llm.parameters(), "lr": LORA_LR},
        {"params": agent.critic.parameters(), "lr": CRITIC_LR},
    ])

    local_buyer = LocalBuyer(agent)
    start_update = load_checkpoint(agent)

    # Transcript logging — best and worst episode per update
    tx_fields = ["update", "ep", "label", "reward", "item", "listing_price",
                 "buyer_model", "transcript"]
    write_tx_header = not os.path.exists(TRANSCRIPT_FILE)
    if write_tx_header:
        with open(TRANSCRIPT_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=tx_fields).writeheader()

    for update in range(start_update, NUM_UPDATES):
        buyer = pick_buyer(local_buyer, get_ppo_buyer)
        print(f"  Buyer: {buyer.name}", flush=True)
        env = NegotiationEnv(buyer, craigslist_df)
        next_obs, _ = env.reset()
        t0 = time.time()
        print(f"\n--- Update {update + 1}/{NUM_UPDATES}: collecting rollout ---", flush=True)

        rewards = torch.zeros(NUM_EPS, NUM_STEPS).to(agent.device)
        values = torch.zeros(NUM_EPS, NUM_STEPS).to(agent.device)
        context_ids = {ep: [] for ep in range(NUM_EPS)}
        generated_ids = {ep: [] for ep in range(NUM_EPS)}
        old_log_probs = {ep: [] for ep in range(NUM_EPS)}
        num_steps_per_ep = {}
        all_advantages = {}
        all_returns = {}
        episode_rewards = []
        episode_infos = []

        for ep in range(NUM_EPS):
            seller_prompt = env.get_seller_prompt()

            for step in range(NUM_STEPS):
                with torch.no_grad():
                    ctx_ids, gen_ids = agent.generate(next_obs, seller_prompt)
                    old_lps, _, value = agent.evaluate(ctx_ids, gen_ids)

                context_ids[ep].append(ctx_ids)
                generated_ids[ep].append(gen_ids)
                old_log_probs[ep].append(old_lps)
                values[ep][step] = value

                agent_text = agent.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                next_obs, reward, terminated, truncated, info = env.step(agent_text)
                rewards[ep][step] = torch.tensor(reward).to(agent.device)

                if terminated or truncated:
                    num_steps_collected = step + 1
                    print(f"  ep {ep+1}: reward={reward:.3f}  agreed={info.get('agreed_price')}  item={info.get('scenario', '')[:40]}  steps={num_steps_collected}", flush=True)
                    info["transcript"] = env._build_transcript()
                    episode_rewards.append(reward)
                    episode_infos.append(info)
                    next_obs, _ = env.reset()
                    break
            else:
                num_steps_collected = NUM_STEPS

            num_steps_per_ep[ep] = num_steps_collected

            with torch.no_grad():
                ep_advantages, ep_returns = compute_gae(
                    rewards[ep][:num_steps_collected],
                    values[ep][:num_steps_collected],
                    GAMMA, GAE_LAMBDA,
                )

            all_advantages[ep] = ep_advantages
            all_returns[ep] = ep_returns

        # Normalize advantages across the full batch, not per-episode
        all_advs_flat = torch.cat([all_advantages[ep] for ep in range(NUM_EPS)])
        adv_mean = all_advs_flat.mean()
        adv_std = all_advs_flat.std()
        for ep in range(NUM_EPS):
            all_advantages[ep] = (all_advantages[ep] - adv_mean) / (adv_std + 1e-8)

        total_steps_per_epoch = sum(num_steps_per_ep.values())

        print("\n--- Running PPO update ---", flush=True)
        total_policy_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for epoch in range(NUM_EPOCHS):
            print(f"  Epoch {epoch + 1}/{NUM_EPOCHS}", flush=True)
            optimizer.zero_grad()
            for ep in range(NUM_EPS):
                for step in range(num_steps_per_ep[ep]):
                    new_lps, ent, new_value = agent.evaluate(context_ids[ep][step], generated_ids[ep][step])
                    old_lps = old_log_probs[ep][step]
                    ratio = torch.exp(new_lps - old_lps)
                    adv = all_advantages[ep][step]
                    ploss1 = ratio * adv
                    ploss2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                    ploss = -torch.min(ploss1, ploss2).mean()
                    closs = 0.5 * (all_returns[ep][step] - new_value) ** 2
                    eloss = ent.mean() * ENTROPY_COEFF
                    ((ploss + closs - eloss) / total_steps_per_epoch).backward()

                    if epoch == NUM_EPOCHS - 1:
                        total_policy_loss += ploss.item()
                        total_critic_loss += closs.item()
                        total_entropy += ent.mean().item()

            grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            if not grad_norm.isnan():
                optimizer.step()
            else:
                print("  WARNING: NaN gradients, skipping optimizer step", flush=True)
                optimizer.zero_grad()

        deals = [i for i in episode_infos if i.get("agreed_price") is not None]
        no_deals = len(episode_infos) - len(deals)
        mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        mean_price_ratio = sum(i["agreed_price"] / i["listing_price"] for i in deals) / len(deals) if deals else 0
        wall_time = time.time() - t0

        metrics = {
            "update": update + 1,
            "deals": len(deals),
            "no_deals": no_deals,
            "mean_reward": round(mean_reward, 4),
            "mean_price_ratio": round(mean_price_ratio, 4),
            "policy_loss": round(total_policy_loss / total_steps_per_epoch, 4),
            "critic_loss": round(total_critic_loss / total_steps_per_epoch, 4),
            "mean_entropy": round(total_entropy / total_steps_per_epoch, 4),
            "wall_time_s": round(wall_time, 1),
        }
        print(f"  Deals: {len(deals)}/{len(episode_infos)}  Mean reward: {mean_reward:.3f}  "
              f"Price ratio: {mean_price_ratio:.3f}  "
              f"Policy loss: {metrics['policy_loss']:.4f}  Critic loss: {metrics['critic_loss']:.4f}  "
              f"Entropy: {metrics['mean_entropy']:.4f}  Time: {wall_time:.1f}s", flush=True)
        log_update(update, metrics)

        # Save best and worst episode transcripts
        if episode_rewards:
            best_ep = max(range(len(episode_rewards)), key=lambda i: episode_rewards[i])
            worst_ep = min(range(len(episode_rewards)), key=lambda i: episode_rewards[i])
            for label, idx in [("best", best_ep), ("worst", worst_ep)]:
                info = episode_infos[idx]
                with open(TRANSCRIPT_FILE, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=tx_fields).writerow({
                        "update": update + 1, "ep": idx + 1, "label": label,
                        "reward": round(episode_rewards[idx], 3),
                        "item": info.get("scenario", ""),
                        "listing_price": info.get("listing_price", ""),
                        "buyer_model": getattr(buyer, "name", "unknown"),
                        "transcript": info.get("transcript", ""),
                    })

        if (update + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(agent, update + 1)

    print("\nSaving final checkpoint...", flush=True)
    save_checkpoint(agent, NUM_UPDATES)
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        print("Training complete. Stopping pod to save costs...", flush=True)
        os.system(f"runpodctl stop pod {pod_id}")


if __name__ == "__main__":
    main()
