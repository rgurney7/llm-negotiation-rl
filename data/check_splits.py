"""Data-split integrity check.

Enforces the one invariant the generalization claim depends on: the held-out
evaluation set shares zero uuids with the training set. Also reports the known
overlap between craigslist_gold.csv and the training set (gold is the
quality-reference set used during data curation, not a training or evaluation
input; see README).

Run:  python data/check_splits.py
Exits non-zero if the eval/train invariant is violated.
"""

import os
import sys
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(DATA_DIR, "craigslist_grpo_enriched.csv")  # training data
EVAL_PATH = os.path.join(DATA_DIR, "craigslist_eval.csv")           # held-out validation
GOLD_PATH = os.path.join(DATA_DIR, "craigslist_gold.csv")           # held-back, unused for eval


def uuids(path):
    return set(pd.read_csv(path)["uuid"].unique())


def main():
    train = uuids(TRAIN_PATH)
    eval_ = uuids(EVAL_PATH)
    print(f"train uuids (enriched): {len(train)}")
    print(f"eval uuids (held-out):  {len(eval_)}")

    eval_overlap = eval_ & train
    print(f"eval ∩ train:           {len(eval_overlap)}")

    if os.path.exists(GOLD_PATH):
        gold = uuids(GOLD_PATH)
        gold_overlap = gold & train
        print(f"gold uuids:             {len(gold)}")
        print(f"gold ∩ train:           {len(gold_overlap)}  "
              f"(expected: gold is a curation reference, not a train/eval input)")

    # Hard invariant: the held-out eval set must not leak into training.
    if eval_overlap:
        print(f"\nFAIL: {len(eval_overlap)} eval uuid(s) overlap training: "
              f"{sorted(eval_overlap)[:10]}", file=sys.stderr)
        sys.exit(1)

    print("\nOK: held-out eval set has zero overlap with training set.")


if __name__ == "__main__":
    main()
