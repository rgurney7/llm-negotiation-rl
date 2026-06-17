import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.reward import price_reward

# reserve = 0.7 * listing, ceiling = listing. Tests use listing=100 so the
# 0.7 kink sits at an agreed price of 70.


def test_no_deal_is_zero():
    assert price_reward(None, 70.0, 100.0) == 0.0


def test_just_below_reserve_is_minus_one():
    # The kink: anything strictly under the reserve pays a flat -1.
    assert price_reward(69.99, 70.0, 100.0) == -1.0


def test_at_reserve_is_zero():
    # Exactly at 0.7 * listing is the bottom of the linear range, not the cliff.
    assert price_reward(70.0, 70.0, 100.0) == 0.0


def test_just_above_reserve_is_linear():
    # Just above the kink the reward follows the linear ramp, not the -1 cliff:
    # (73 - 70) / (100 - 70) = 0.1.
    assert price_reward(73.0, 70.0, 100.0) == 0.1


def test_linear_across_reserve_to_ceiling():
    # Linear 0 -> 1 from reserve (70) to ceiling (100).
    assert price_reward(85.0, 70.0, 100.0) == 0.5
    assert price_reward(100.0, 70.0, 100.0) == 1.0


def test_clipped_above_ceiling():
    assert price_reward(130.0, 70.0, 100.0) == 1.0
