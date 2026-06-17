import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _uuids(name):
    return set(pd.read_csv(os.path.join(DATA_DIR, name))["uuid"].unique())


def test_eval_disjoint_from_train():
    # The one invariant the held-out generalization claim depends on. Verified
    # 0 overlap by hand; this test documents and enforces it (mirrors
    # data/check_splits.py).
    train = _uuids("craigslist_grpo_enriched.csv")
    held_out = _uuids("craigslist_eval.csv")
    assert train & held_out == set(), "held-out eval uuids leaked into the training set"


def test_split_sizes():
    assert len(_uuids("craigslist_grpo_enriched.csv")) == 526
    assert len(_uuids("craigslist_eval.csv")) == 153
