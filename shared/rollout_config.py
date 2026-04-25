from shared.models import (
    GRPO_OPPONENT_ROSTER, PPO_OPPONENT_ROSTER,
    eval_model, pick_from_roster,
)


def get_grpo_buyer():
    return pick_from_roster(GRPO_OPPONENT_ROSTER)


def get_ppo_buyer():
    return pick_from_roster(PPO_OPPONENT_ROSTER)


def get_eval_model():
    return eval_model
