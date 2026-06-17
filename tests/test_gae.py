import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from shared.gae import compute_gae


def test_terminal_done_masks_bootstrap_single_step():
    # One-step episode: the done flag on the last step must zero the V_{t+1}
    # bootstrap, so advantage = reward - value and return = reward.
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.4])
    adv, ret = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)
    assert torch.allclose(adv, torch.tensor([0.6]))
    assert torch.allclose(ret, torch.tensor([1.0]))


def test_last_step_has_no_future_value():
    # On the final step of a multi-step trajectory the bootstrap is masked:
    # advantage_T = r_T - V_T regardless of the earlier values.
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.2, 0.3, 0.5])
    adv, ret = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)
    assert torch.allclose(adv[-1], torch.tensor(1.0 - 0.5))
    # returns are defined as advantages + values
    assert torch.allclose(ret, adv + values)


def test_matches_manual_two_step():
    rewards = torch.tensor([0.0, 1.0])
    values = torch.tensor([0.5, 0.4])
    gamma, lam = 0.9, 0.8
    adv, _ = compute_gae(rewards, values, gamma, lam)
    # t=1 (terminal): delta = 1.0 - 0.4 = 0.6 ; adv1 = 0.6 (bootstrap masked)
    # t=0: delta = gamma*V1 - V0 = 0.9*0.4 - 0.5 = -0.14
    #      adv0 = delta + gamma*lam*adv1 = -0.14 + 0.72*0.6 = 0.292
    assert torch.allclose(adv[1], torch.tensor(0.6))
    assert torch.allclose(adv[0], torch.tensor(0.292), atol=1e-6)
