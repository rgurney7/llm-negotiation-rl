# Reinforcement Learning for Multi-Turn LLM Negotiation

Training a 4B-parameter LLM to negotiate price in multi-turn conversations against diverse buyer opponents, using PPO, GRPO, and SFT. Two RL algorithms built from scratch in raw PyTorch; six PPO runs, four GRPO run series, one SFT baseline, and a five-agent evaluation sweep on Craigslist negotiations. Total compute spend: ~$250.

**See [analysis.md](analysis.md) for the full technical writeup**: motivation, prior work, algorithm design, results, and what I learned.

## Results

Evaluation on 50 Craigslist negotiations sampled from the enriched dataset, with the buyer played by Gemini/Groq API models. See the Reproducibility note under Limitations for caveats on the eval pipeline.

| Agent | Deals | Deal Rate | Mean Reward | Price Ratio |
|-------|-------|-----------|-------------|-------------|
| Base (no training) | 37/50 | 74% | +0.028 | 0.753 |
| SFT | 25/50 | 50% | **+0.087** | **0.793** |
| GRPO | 38/50 | 76% | -0.101 | 0.736 |
| PPO v5 (PPO+GRPO init) | 28/50 | 56% | -0.014 | 0.759 |
| PPO v6 (PPO from base) | 29/50 | 58% | -0.010 | 0.765 |

SFT generalized best; RL agents showed strong training performance but struggled on held-out data at this scale. Full discussion in [analysis.md](analysis.md).

## Architecture

```
                    Seller Agent                         Buyer Opponent
              ┌─────────────────────┐              ┌──────────────────┐
              │  Qwen3.5-4B (frozen)│              │  Gemini / Groq   │
              │  + LoRA (r=8, a=16) │◄────────────►│  API models      │
              │  + Critic (23M, PPO)│  multi-turn  │  (frozen)        │
              └─────────┬───────────┘  transcript   └──────────────────┘
                        │
                        ▼
              ┌─────────────────────┐              ┌──────────────────┐
              │  PPO / GRPO Update  │◄─────────────│  Gemini Judge    │
              │  (LoRA params only) │    reward     │  (price extract) │
              └─────────────────────┘              └──────────────────┘
```

Reward is piecewise: -1 below 70% of listing, 0 for no deal, and linear 0 to 1 between 70% and 100% of listing. No subjective quality score.

## Three Approaches

### PPO (`ppo/`)

Multi-turn PPO with token-level control. Free-form text across 8-turn episodes; 23M-parameter standalone critic; sparse terminal reward propagated via GAE.

- [ppo_agent.py](ppo/ppo_agent.py): LLM agent with LoRA + critic head
- [ppo_env.py](ppo/ppo_env.py): Multi-turn Gymnasium environment
- [ppo_train.py](ppo/ppo_train.py): PPO training loop
- [ppo_eval.py](ppo/ppo_eval.py): Held-out evaluation

### GRPO (`grpo/`)

Single-turn Group Relative Policy Optimization. For each prompt (transcript truncated to the final turn), generates G=8 completions and uses group-relative advantages.

- [grpo_agent.py](grpo/grpo_agent.py): GRPO agent with reference policy
- [grpo_env.py](grpo/grpo_env.py): Single-turn environment with concurrent buyer/judge calls
- [grpo_train.py](grpo/grpo_train.py): GRPO training loop
- [grpo_eval.py](grpo/grpo_eval.py): Multi-turn evaluation

### SFT (`sft/`)

Supervised fine-tuning baseline on curated positive-reward trajectories from synthetic data. Despite training on synthetic data only, SFT was the only method that transferred to held-out Craigslist scenarios.

- [sft_train.py](sft/sft_train.py): SFT with masked loss on seller turns

### Shared (`shared/`)

- [config.py](shared/config.py): Scenarios, personas, tactic taxonomy, prompt builders
- [models.py](shared/models.py): API model wrappers (OpenAI, Gemini, Groq) with retry/fallback
- [rollout_config.py](shared/rollout_config.py): Buyer roster selection
- [local_buyer.py](shared/local_buyer.py): Self-play buyer (base model with LoRA disabled)
- [reward.py](shared/reward.py): `price_reward()` used by all three approaches
- [gae.py](shared/gae.py): `compute_gae()` used by PPO

## Project Structure

```
README.md                       # This file
analysis.md                     # Full technical writeup
ppo/
  ppo_agent.py                  # Qwen3.5-4B + LoRA + 23M critic
  ppo_env.py                    # Multi-turn Gymnasium env
  ppo_train.py                  # PPO training loop
  ppo_eval.py                   # Held-out evaluation
grpo/
  grpo_agent.py                 # GRPO agent (no critic)
  grpo_env.py                   # Single-turn env
  grpo_train.py                 # GRPO training loop
  grpo_eval.py                  # Multi-turn evaluation
sft/
  sft_train.py                  # SFT baseline
shared/
  config.py                     # Scenarios, personas, tactics, prompt builders
  models.py                     # API model wrappers
  rollout_config.py             # Buyer roster selection
  local_buyer.py                # Self-play buyer
  reward.py                     # price_reward()
  gae.py                        # compute_gae()
data/
  craigslist.py                 # Craigslist parsing pipeline
  craigslist_parsed.csv         # Raw CraigslistBargain corpus (source for curation)
  craigslist_grpo_enriched.csv  # Training data (526 transcripts)
  craigslist_eval.csv           # Held-out evaluation
  craigslist_gold.csv           # Quality-reference set (curation calibration)
  synthetic_data.csv            # 200 synthetic scenarios
```

## Data

- **CraigslistBargains** (He et al. 2018): 526 real negotiation transcripts filtered through a two-pass quality pipeline, 460 of which reached a deal.
- **Synthetic scenarios**: 200 scenarios generated via Gemini across 40 item types and 10 buyer personas, spanning 5 price tiers ($50–$500K).
- **Held-out validation** (`craigslist_eval.csv`): 153 uuids with zero overlap with the 526 training uuids, enforced by `python data/check_splits.py`.
- **`craigslist_gold.csv`**: the gold-standard quality-reference set (top-tier transcripts with full rubric scores) used during data curation to calibrate the two-pass quality filter. Not a training or evaluation input; held-out eval uses `craigslist_eval.csv`.
- **`craigslist_parsed.csv`**: the normalized CraigslistBargain corpus (5,830 dialogues) produced by `data/craigslist.py`. This is the source the curated sets were built from, kept in the repo because the upstream CodaLab bundles can rot. Not used in training or evaluation.

## Limitations

- Roughly 500 training scenarios is small for RL generalization at this model scale.
- The buyer opponent distribution is narrow, and the learned behavior appeared to be opponent-specific.
- Reward captures agreed price only, not coherence or professionalism.
- Single seed per run; variance is unknown, and reward differences below roughly 0.05 should be treated as uncertain.
- Training and evaluation buyer rosters overlap, so reported behavior cannot be cleanly separated from opponent-style adaptation.
- Price extraction by the LLM judge was spot-checked during development rather than formally validated against ground-truth labels or across multiple judges.

### Reproducibility
Training ran on ephemeral GPU compute. LoRA and critic checkpoints and per-update training logs were not persisted; the numbers in the results table reflect run-time observation during the original experiments and are not independently re-verifiable from this repository. The code here is the training and evaluation pipeline, and reproducing the reported numbers would require re-running the full sweep. The evaluation scripts as committed diverged from the pipeline used during development (the default data path in `ppo_eval.py`, and the env constructor signature in `grpo_eval.py`), and should be read as the intended design rather than a direct replay. See the Implementation Notes in [analysis.md](analysis.md) for the held-out eval path and two other known issues in the PPO and GRPO code.

## References

- He et al. (2018). "Decoupling Strategy and Generation in Negotiation Dialogues." *EMNLP*.
- Lewis et al. (2017). "Deal or No Deal? End-to-End Learning for Negotiation Dialogues." *EMNLP*.
- Kolluri & Zhu (2025). "Training LLMs for Negotiation via Self-Play." *Stanford CS224N*.
