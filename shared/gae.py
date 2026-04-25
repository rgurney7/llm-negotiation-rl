import torch


def compute_gae(rewards, values, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    lastgae = 0.0
    n = len(rewards)
    for t in reversed(range(n)):
        next_value = values[t + 1] if t < n - 1 else 0.0
        nonterminal = 1.0 if t < n - 1 else 0.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgae = delta + gamma * gae_lambda * nonterminal * lastgae
        advantages[t] = lastgae
    returns = advantages + values
    return advantages, returns
