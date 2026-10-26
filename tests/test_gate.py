"""Paired gate rule (rewind/eval/gate.py) on toy per-query rows, decisions worked out by hand."""

import pytest

from rewind.eval.gate import paired_decision


def spec(q, seconds, hit=True):
    return {"query_id": q, "kind": "specific", "seconds": seconds, "hit": hit}


def cat(q, ap):
    return {"query_id": q, "kind": "category", "AP@50": ap}


def decide(a, b):
    return paired_decision(a, b, "AP@50", n_boot=200, seed=1, ci_level=0.95)


def test_adopt_when_mean_better_and_more_wins():
    # specific diffs (simple - complex): +100, -10, +5 -> 2 wins, 1 loss, mean +31.7; category +0.1 -> win
    d = decide([spec("a", 200), spec("b", 50), spec("c", 30), cat("d", 0.2)],
               [spec("a", 100), spec("b", 60), spec("c", 25), cat("d", 0.3)])
    assert (d["wins"], d["ties"], d["losses"]) == (3, 0, 1)
    assert d["families"]["specific_seconds"]["mean_diff"] == pytest.approx(95 / 3)
    assert d["mean_better"] and d["adopt"]


def test_keep_simple_when_one_big_win_but_more_losses():
    # +900, -10, -10: mean better (+293) but 1 win vs 2 losses -> keep the simpler variant
    d = decide([spec("a", 1000), spec("b", 50), spec("c", 50)], [spec("a", 100), spec("b", 60), spec("c", 60)])
    assert d["mean_better"] and not d["adopt"]


def test_keep_simple_when_families_disagree():
    # specific better (+50, +50), category worse (-0.2): 2 wins, 1 loss but mean not better overall
    d = decide([spec("a", 100), spec("b", 100), cat("c", 0.5)], [spec("a", 50), spec("b", 50), cat("c", 0.3)])
    assert (d["wins"], d["losses"]) == (2, 1) and not d["mean_better"] and not d["adopt"]


def test_both_missed_is_a_tie_but_still_counts_in_the_mean():
    # a: both miss, costs differ by list length (+40) -> tie; b: +10 -> win; mean +25, 1 win 0 losses
    d = decide([spec("a", 5040, hit=False), spec("b", 30)], [spec("a", 5000, hit=False), spec("b", 20)])
    assert (d["wins"], d["ties"], d["losses"]) == (1, 1, 0) and d["adopt"]


def test_equal_everything_keeps_simple():
    d = decide([spec("a", 10), cat("b", 0.5)], [spec("a", 10), cat("b", 0.5)])
    assert d["ties"] == 2 and not d["mean_better"] and not d["adopt"]


def test_different_queries_refused():
    with pytest.raises(ValueError, match="different queries"):
        decide([spec("a", 1)], [spec("b", 1)])


def test_gate44_tiebreak_reading():
    from scripts.gate44_tiebreak import tiebreak_decision
    # Conflict: specific diffs +300, +100, -10 (2 wins, 1 loss, mean +130), category -0.1 (1 loss) -> W/L 2/2.
    # Literal rule: category worse -> not mean better -> keep simple. Tie-break: conflict, seconds better, but
    # wins (2) > losses (2) fails -> keep simple.
    d = decide([spec("a", 400), spec("b", 200), spec("c", 50), cat("d", 0.3)],
               [spec("a", 100), spec("b", 100), spec("c", 60), cat("d", 0.2)])
    assert not d["adopt"]
    assert tiebreak_decision(d, "AP@50") == (True, False)
    # Add one more specific win (+50): W/L 3/2 -> tie-break adopts, the literal rule still does not.
    d = decide([spec("a", 400), spec("b", 200), spec("c", 50), spec("e", 100), cat("d", 0.3)],
               [spec("a", 100), spec("b", 100), spec("c", 60), spec("e", 50), cat("d", 0.2)])
    assert not d["adopt"]
    assert tiebreak_decision(d, "AP@50") == (True, True)
    # No conflict (both families worse) -> the literal rule's answer.
    d = decide([spec("a", 100), cat("d", 0.3)], [spec("a", 200), cat("d", 0.2)])
    assert tiebreak_decision(d, "AP@50") == (False, d["adopt"]) == (False, False)
