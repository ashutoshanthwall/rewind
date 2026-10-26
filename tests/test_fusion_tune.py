"""The pre-registered step 7.4 rule (scripts/fusion_tune.decide) on hand-made sweeps."""

from scripts.fusion_tune import decide


def s(r10, med):
    return {"R@10": r10, "median_seconds": med}


def test_highest_recall_wins():
    d, ranking = decide({0: s(0.5, 100), 0.5: s(0.667, 900), 1: s(0.5, 50)})
    assert d == "0.5" and ranking[0] == 0.5


def test_recall_tie_broken_by_lower_median_seconds():
    d, ranking = decide({0: s(0.5, 400), 1: s(0.5, 300), 2: s(0.333, 10)})
    assert d == "1" and ranking == [1, 0, 2]


def test_full_tie_is_not_decided():
    d, _ = decide({0: s(0.5, 300), 0.25: s(0.5, 300), 2: s(0.1, 1)})
    assert d == "NEEDS_USER"
