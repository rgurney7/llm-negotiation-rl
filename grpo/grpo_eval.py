import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../ppo"))

import argparse
import csv
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from grpo_agent import GrpoAgent
from shared.config import SCENARIOS, PERSONAS, make_buyer_prompt, make_seller_prompt
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
from ppo_env import NegotiationEnv


CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
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


def run_eval(agent, buyer_model, buyer_tokenizer, label, scenarios, personas, log_rows):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n", flush=True)

    for scenario in scenarios:
        for persona in personas:
            persona_name = persona["name"]
            env = NegotiationEnv(buyer_model, buyer_tokenizer, agent.device)
            env.scenario = scenario
            env.persona = make_buyer_prompt(persona, scenario)
            obs = env._get_agent_obs()
            seller_prompt = make_seller_prompt(scenario)

            while env.current_step < env.max_steps:
                full_prompt = seller_prompt + "\n\n" + obs
                _, generated_ids = agent.generate(full_prompt)
                agent_text = agent.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                obs, reward, terminated, truncated, info = env.step(agent_text)

            transcript = env._build_transcript()
            agreed = info.get("agreed_price")
            turn = info.get("agreement_turn")

            print(f"[{scenario['item']}] vs {persona_name}  "
                  f"price={agreed}  turn={turn}  reward={reward:.3f}")
            print(f"{transcript}\n", flush=True)

            log_rows.append({
                "agent": label,
                "scenario": scenario["item"],
                "persona": persona_name,
                "agreed_price": agreed,
                "agreement_turn": turn,
                "reward": round(reward, 3),
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=int, required=True, help="Checkpoint update number to evaluate (e.g. 20)")
    parser.add_argument("--skip-base", action="store_true", help="Skip base model benchmark")
    parser.add_argument("--trial", action="store_true", help="Quick trial: 1 scenario, 2 personas")
    args = parser.parse_args()

    scenarios = SCENARIOS[:1] if args.trial else SCENARIOS
    personas = PERSONAS[:2] if args.trial else PERSONAS

    buyer_model_id = "microsoft/Phi-4-mini-instruct"
    buyer_tokenizer = AutoTokenizer.from_pretrained(buyer_model_id)
    if buyer_tokenizer.pad_token is None:
        buyer_tokenizer.pad_token = buyer_tokenizer.eos_token
    buyer_model = AutoModelForCausalLM.from_pretrained(buyer_model_id, torch_dtype=torch.float16)

    log_rows = []

    # Base model benchmark
    if not args.skip_base:
        base_agent = load_base_agent()
        buyer_model.to(base_agent.device)
        run_eval(base_agent, buyer_model, buyer_tokenizer, "base", scenarios, personas, log_rows)
        del base_agent
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Trained agent
    trained_agent = load_trained_agent(args.update)
    buyer_model.to(trained_agent.device)
    run_eval(trained_agent, buyer_model, buyer_tokenizer, f"grpo_{args.update}", scenarios, personas, log_rows)

    # Write CSV
    fields = ["agent", "scenario", "persona", "agreed_price", "agreement_turn", "reward"]
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
