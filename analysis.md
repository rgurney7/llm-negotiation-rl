# Reinforcement Learning for Multi-Turn LLM Negotiation: Analysis

## Why Negotiation

Reinforcement Learning thrives on clear reward signals. Win or loss, survival or death, success or failure. The algorithms have done well in chess, Go, and video games, and recent work applying RL to language models has seen real success. But most of that work falls into one of two buckets: offline methods where the agent doesn't generate its own data, or online methods on single-turn tasks. Both produce good results, and both sidestep what RL is actually best at: optimizing over a long time horizon, beyond what human demonstrations contain.

Negotiation doesn't sidestep it. It's multi-turn, the reward is grounded in something real (the final agreed price), and credit assignment is hard. Which turn actually mattered? A concession that looks weak in isolation might set up a better final price. A firm counteroffer might win the deal or kill it. Chess has the same structure: moves that look sub-optimal in terms of material can turn out to be critical for the outcome.

The public CraigslistBargains dataset created by He et al. (2018) provides a unique opportunity for agents to operate in complex environments based on real-life situations and develop novel strategies in the pursuit of grounded rewards. The reward signal, agreed price divided by listing price, is concrete and provides a clear signal to the LLM on what success looks like. The agent either closed at 90% of asking or it didn't.

## Prior Work

RL for negotiation has a short but instructive history.

Lewis et al. (2017) at Facebook AI Research were the first to train end-to-end neural negotiation agents. They used REINFORCE with dialogue rollouts on a multi-issue bargaining game ("Deal or No Deal"), and the RL agent learned to negotiate, compromise, and even deceive, outperforming humans on deal value. But this was a constrained discrete-item game with a fixed action structure, not open-ended natural language price negotiation.

He et al. (2018) at Stanford created the CraigslistBargains dataset that I use and attempted RL on it directly. Their central finding: end-to-end RL on word-level generation degenerated. The model collapsed into repeating the same few sentences, with the top 3 accounting for 81-100% of all utterances. Their fix was to decouple strategy from generation, training RL on coarse dialogue acts (e.g., `propose(price=50)`) rather than raw text. This became the conventional wisdom: RL on free-form negotiation language doesn't work.

Kolluri & Zhu (2025) at Stanford more recently applied PPO self-play to fine-tune Qwen-2.5-1.5B on negotiation games. Without behavior cloning mixed throughout training, their agent couldn't maintain basic formatting, producing invalid deals and negative rewards. Their hybrid BC+PPO pipeline resolved this during training, though the paper does not evaluate on held-out opponents.

## What I Built and Why

My project tests whether these limitations still hold with modern LLMs and LoRA fine-tuning. The short answer: PPO on full natural language generation works now. Stable entropy, coherent output, real strategic learning across 8-turn episodes, without decoupling strategy from generation and without behavior cloning. The longer answer is that "works during training" and "generalizes to new data" turned out to be very different claims.

I tested three methods: two RL algorithms implemented from scratch in PyTorch, and a supervised fine-tuning baseline for comparison. GRPO isolates whether single-turn group-relative optimization can teach trajectory-level strategy. PPO tests whether multi-turn credit assignment with a learned critic outperforms it. SFT establishes whether either RL approach adds anything over straight imitation.

The GRPO implementation is a single-turn algorithm adapted for negotiation. For each prompt (a transcript truncated to the final turn), the model generates G=8 completions, and group-relative advantages amplify what worked. Training on only the final turn is analogous to showing a chess engine adversarial positions near the endgame: the hypothesis was that optimizing one decision point would teach the model features of the entire trajectory.

PPO runs in a multi-turn setup, with a 23M-parameter standalone critic evaluating each turn and sparse terminal reward propagated backward via GAE. I ran six training runs across two model families, producing over 3,600 negotiations in total. The SFT baseline is trained for three epochs on curated positive-reward trajectories from the synthetic dataset.

All three methods use Qwen3.5-4B with LoRA (r=8, α=16) as the base; earlier runs used Llama-3.2-3B. The core training data is 526 real Craigslist transcripts filtered through a two-pass quality pipeline, 460 of which reached a deal. GRPO additionally draws on 200 synthetic scenarios seeded off-loop by a stronger set of models (`gpt-5.4`, `gpt-5.4-mini`, `gemini-3.1-pro-preview`, and `gemini-3-flash-preview`), none of which run during training.

On the other side of each negotiation sits a frozen API buyer, sampled per episode from a weighted roster of `gemini-3.1-flash-lite-preview`, `gemini-2.5-flash`, and Groq's `openai/gpt-oss-120b`, so the agent doesn't overfit to a single opponent. A wrapper retries against another model in the roster on failure, which keeps a flaky API from crashing a run. The judge is a single `gemini-3.1-flash-lite-preview` call that reads the finished transcript and extracts the agreed price and the turn at which the price was agreed.

Reward is the agreed price over the listing price, mapped to [−1, 1]: −1 below 70% of listing, 0 for no deal, and a linear scale from 0 to 1 between 70% and 100% of listing.

## GRPO: A Cheap Signal That Transfers (To a Point)

The core idea: show the model full negotiation transcripts truncated to the final turn, have it generate 8 completions, reward the ones that close well. Each candidate gets scored independently: the buyer replies once, the judge extracts a price from (prefix + continuation + reply), and group-relative advantages do the rest. If it can learn what makes a good closing move from the full conversation context, maybe it internalizes the value of each position in the trajectory and develops a kind of implicit critic network.

On synthetic data, this worked. Mean reward improved by +0.451 for Llama-3.2-3B and +0.220 for Qwen3.5-4B when evaluated on 7-turn multi-turn negotiations, despite training only on single-turn completions. Bad deals were cut in half. The biggest gains were against aggressive (+0.310) and irrational (+0.456) buyer personas, exactly the cases where the untrained model tends to cave under pressure. Scoped honestly, this is a within-distribution single-turn-to-multi-turn transfer result: training on one decision point improved full-episode reward inside the same synthetic distribution.

One caveat keeps it scoped: training used 14 of the 50 synthetic scenarios and 5 of the 24 personas, and the committed code does not show the multi-turn eval drew from the held-back material, so item familiarity is a live alternative to genuine trajectory-level feature learning. The honest claim is within-distribution transfer from single-turn to multi-turn.

On real Craigslist data, the transfer didn't hold up the same way. GRPO learned during training, but in the multi-turn Craigslist comparison it landed at −0.101, below the untrained base model. Two things likely compounded here. The features it picked up from synthetic Gemini-vs-Gemini training didn't cleanly map onto real buyer behavior. And even after the two-pass quality pipeline, the Craigslist data at our scale probably didn't carry a rich enough signal for group-relative advantages to pick up general strategy rather than surface patterns. Single-turn training with sparse signal works when the underlying distribution is learnable from the amount of data you have; at 4B scale with a few hundred filtered transcripts, it likely wasn't.

The honest framing: single-turn GRPO transfer to multi-turn is real, but it's same-distribution transfer. The model learns features that help within the distribution it trained on. SFT trained on the same synthetic distribution and did transfer to Craigslist, so cross-distribution generalization is possible from this data. The difference is the learning signal: dense per-token imitation on curated trajectories versus sparse reward spread across a group of mostly-failing rollouts. At this scale, the sparse signal isn't rich enough for the model to extract distribution-invariant features.

## PPO: Real Learning

I ran six PPO experiments. The progression tells a clear story about what matters for multi-turn RL on language models.

Reward is sparse and terminal: episodes run for a full 8 turns with no early termination on agreement, and at turn 8 the judge extracts the final agreed price from the complete transcript. That single scalar propagates backward through the trajectory via GAE, which puts a lot of weight on the critic.

The critic was the single biggest lever. A 787K-parameter critic produced zero learning. Scaling to 23M with the same LLM and same task took win rate from 25% to 48%, with stable entropy throughout. Kolluri & Zhu (2025) are a useful reference point here. Their PPO setup on Qwen2.5-1.5B used a 2-layer MLP critic at roughly 4M parameters, and without behavior cloning mixed throughout training their agent couldn't hold basic formatting. That's consistent with what I saw at 787K. An undersized critic produces noisy advantage estimates, which feeds into unstable policy updates, which is the kind of format collapse BC usually gets brought in to patch. At 23M my PPO trained stably without BC. Separate learning rates also turned out to be essential. The critic needs roughly 30x the LoRA learning rate (3e-4 vs 1e-5) because it's training from scratch and has to learn V(s) before the policy shifts underneath it.

One non-standard detail worth flagging: the critic reads the hidden state at the last token of `context + generated`, so V is evaluated after the action rather than before. That's closer to Q(s,a) than V(s) in the strict sense, but since step-t and step-t+1 values are computed identically, the GAE signal `r + γV_{t+1} - V_t` still yields a meaningful relative estimate. Standard PPO implementations compute V(s_t) before the action; this one doesn't.

The best PPO results were encouraging. Run 5 (8-step episodes, 150 updates, 1,200 negotiations) peaked around updates 41-60 with reward at 2.4x baseline, deal rate at 72%, and price ratio at 0.86. The model learned to hold firm on price: a real strategic behavior emerging from sparse terminal reward passed through 8 turns of credit assignment. The format-collapse failure mode He et al. reported did not reappear in our setup during training; whether this generalizes beyond 4B LoRA + 23M critic is addressed below. LoRA keeps the pretrained language model frozen as a fluency backbone, and a properly sized critic provides clean advantage signal. The combination produces stable, coherent RL training on free-form negotiation text.

What the policy actually learned, and how it fared against the other agents, is taken up in the evaluation section below. The training-time result stands on its own: a sparse terminal reward, propagated through eight turns of credit assignment, produced real strategic behavior with stable entropy and no format collapse.

I also tested whether GRPO initialization helps PPO with an explicit controlled experiment: identical configs, one initialized from a GRPO checkpoint, one from the base model. The difference was +0.041 mean reward over 150 updates. The GRPO-initialized run actually lost entropy faster and plateaued earlier. Not worth the extra step.

## Reward Shape

Two consequences of the reward design are worth naming, because they show up in what the agents actually learned.

First, **walk-away dominates bad-deal.** Closing below 70% of listing pays −1, but walking away pays 0. Any policy that can't reliably hit 0.7× listing strictly prefers no deal. This is a structural incentive for conservative behavior, not an emergent strategy, and it's a big part of why SFT's "hold firm and walk away" looks so effective.

Second, **discontinuous gradient at 0.7.** The reward jumps from −1 at 0.69 to 0 at 0.70, so there's no smooth signal pulling the agent through the [0.5, 0.7] range. A shaped reward (clipped linear from −1 at 0.5 to +1 at 1.0) would likely produce smoother learning but would change what the agent is optimizing for. I kept the piecewise version because it matches how a person evaluates a deal: below threshold is bad, above is good, and the marginal difference inside "bad" doesn't matter much.

One note on the extractor: price extraction is a single Gemini Flash Lite call asking for one number or "NONE". I spot-checked ~50 extractions and found them accurate in every case, with no formal eval run. Regex parsing isn't feasible because final agreements surface in varied forms ("deal at 400", "let's split, 425", "sounds good, $400 it is"), and one-number extraction from messy prose is exactly what small LLMs are good at.

## SFT Wins, But the Comparison is Confounded


| Agent         | Deals | Deal Rate | Mean Reward | Price Ratio |
| ------------- | ----- | --------- | ----------- | ----------- |
| base          | 37/50 | 74%       | +0.028      | 0.753       |
| sft           | 25/50 | 50%       | +0.087      | 0.793       |
| grpo          | 38/50 | 76%       | −0.101      | 0.736       |
| v5 (PPO+GRPO) | 28/50 | 56%       | −0.014      | 0.759       |
| v6 (PPO base) | 29/50 | 58%       | −0.010      | 0.765       |


The five-agent comparison was evaluated on the training distribution. The held-out split could not be reconstructed, since the enrichment that produced its prompts was not persisted off the original compute, so these numbers do not establish held-out generalization for the RL agents. SFT trained on synthetic data, which makes Craigslist evaluation out-of-distribution for it regardless of the file used; its improvement over base is the project's one cross-distribution result. The training-time findings (critic-capacity ablation, stable multi-turn training with no format collapse) do not depend on this evaluation and stand independently.

Within that comparison SFT posted the highest price ratio (0.793), trading deal rate (50% vs 74% for base) by learning to hold firm and walk away from bad deals.

Before reading too much into the comparison, SFT had two structural advantages worth naming. First, curation: it trained only on positive-reward synthetic trajectories, so every gradient step pulled toward a known-good outcome. The RL agents had to discover what good looks like from scratch, with most rollouts producing zero or negative reward. Second, signal density: SFT got a per-token loss on every seller turn, while PPO had to propagate a single terminal scalar through 8 turns and GRPO had to find signal inside groups of mostly-failing completions. What SFT did not have was a data distribution advantage. It trained on the same synthetic scenarios as GRPO, against the same Gemini-generated buyers, and still transferred to Craigslist negotiations where GRPO did not.

The data regime matters too. Modern RL results showing generalization past SFT typically train on tens of thousands of samples with verifiable per-step rewards (math, code). Chu et al. (2025) found outcome-driven RL generalizes better than SFT, but their setup is rule-based reasoning with dense feedback at scale. At ~500 transcripts with sparse terminal rewards through 8 turns, that regime doesn't apply yet.

Reading through the actual transcripts, what PPO learned looks more like opponent-specific exploitation than negotiation strategy. The credit assignment machinery worked end-to-end: sparse terminal reward propagated through 8 turns, policy shifted measurably, training metrics improved. The shift just wasn't toward general negotiation skill. It was toward beating the specific buyer model in the loop. That's a meaningful negative result about what end-to-end PPO learns when the opponent distribution is narrow, separate from the question of whether the algorithm has the potential to effectively generalize on long-time horizon text-based tasks. The format-collapse failure mode did not reproduce in this setup; whether that generalizes beyond 4B LoRA + 23M critic is a further question.

## What I Learned

The most practically useful finding is that end-to-end RL on natural language negotiation no longer degenerates the way He et al. (2018) reported, and LoRA plus a properly sized critic solved that. Beyond that: the critic-capacity ablation moved win rate from 25% to 48%, single-turn GRPO transfers to multi-turn within the same synthetic distribution, and SFT — trained on synthetic data — was the one method to improve over base on Craigslist, the project's single cross-distribution result.

I went in believing RL would discover negotiation strategies beyond what SFT could teach. On the Craigslist evaluation, at 4B scale with 500 training scenarios and $250 of compute, it didn't beat SFT. But PPO's credit assignment machinery works. It propagated a sparse terminal reward through 8 turns and produced measurable strategic behavior without the degeneration or format collapse prior work reported. The bottleneck isn't the algorithm; it's dataset size for generalization and model capacity for separating fluency from strategy. Whether SFT followed by PPO at larger scale can break through SFT's ceiling is the open question, but proving that is a different project at a different budget. Long-horizon tasks with grounded rewards and genuine reasoning demands are exactly the regime RL is built for, and I'm more confident in that bet after this project, not less.

Total spend: ~$250 across RunPod GPU rentals and API calls, covering six PPO runs, four GRPO run series, one SFT baseline, and a five-agent evaluation sweep on Craigslist negotiations.

## A Note on Reproducibility

Training was done on ephemeral GPU compute. Model checkpoints, per-update training logs, and the sweep harness that produced the five-agent table were not persisted. The quantitative results above are run-time observations from the original experiments; the code in this repository is the training and evaluation pipeline, but reproducing the numbers would require re-running the full sweep.

## Implementation Notes and Known Issues

Three things in the committed code are worth calling out. The first two are algorithmic and left in place: the original runs used the versions described here, and since the checkpoints were not persisted they are documented rather than silently rewritten, so the diagnosis stays legible. The third was a divergence between the committed eval scripts and the intended experiment; the scripts have been corrected to document the intended design, though the held-out evaluation itself cannot be run from this repo.

**PPO value head is conditioned on the post-action hidden state.** In `ppo_agent.py`, the critic reads the hidden state at the final *generated* token, so the value estimate is closer to Q(s, a) than the state value V(s). A correct state-value baseline should read the last *context*-token hidden state, before generation. Because the baseline then depends on the action taken, the advantage estimate is biased — the GAE residual `r + γV(s') − V(s)` mixes a Q-like term into a recursion that assumes state values. Training still improved (the gradient stayed directionally useful), but a clean reimplementation would move the value read to the pre-action position.

**The GRPO objective is sequence-level, and the KL ratio is inverted.** The policy ratio in `grpo_train.py` is taken from the summed sequence log-prob (`exp(Σnew − Σold)`) rather than the token-level mean of `min(ratio_t·Â, clip(ratio_t)·Â)` in the DeepSeek formulation; the sequence-level form is higher variance and carries a length bias. The KL penalty uses `tr = π_θ/π_ref` inside `tr − log(tr) − 1`, the inverse of the `π_ref/π_θ` ratio the k3 estimator specifies. It still behaves as a valid pull toward the reference policy (non-negative, minimized at parity), but it is not the exact KL direction intended.

**The eval scripts were targeting the wrong split (now corrected to document intent).** As originally committed, `ppo_eval.py` read the enriched *training* file and `grpo_eval.py` ran the synthetic scenarios in `shared/config.py` through a stale env constructor and an off-roster Phi-4-mini buyer. Both now point at `data/craigslist_eval.csv` (153 validation uuids, zero training overlap, verified by `data/check_splits.py`) and sample the buyer from the production roster. The file is not runnable as committed — the enrichment that builds its `seller_prompt`/`buyer_prompt` columns was not persisted — so the scripts document the intended held-out design rather than a runnable replay, and the results table, which predates the correction, is unaffected.