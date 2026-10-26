"""Fusion (rewind/search/fusion.py): z-scores, weights, pooling, grouping, "why it matched". Hand-computed cases."""

import math

import pandas as pd
import pytest

from rewind.search.core import SearchConfig
from rewind.search.fusion import check_weights, fuse, zscores

CFG = SearchConfig(k=200, merge_gap_s=5, max_clip_s=60, context_s=5)
DUR = {1: 300.0, 2: 300.0}


def hits(rows):
    """rows: [(video_id, t_s, score)]"""
    return pd.DataFrame(rows, columns=["video_id", "t_s", "score"])


def test_zscores_hand_computed():
    # [1, 2, 3]: mean 2, population sd sqrt(2/3) = 0.81650 -> -1.22474, 0, +1.22474
    assert zscores([1, 2, 3]) == pytest.approx([-1.224745, 0, 1.224745], abs=1e-6)
    assert zscores([0.7]).tolist() == [0.0]                  # one hit: no spread
    assert zscores([0.4, 0.4, 0.4]).tolist() == [0.0] * 3    # equal scores: no spread
    assert zscores([]).tolist() == []


def test_weights_rank_branches_hand_computed():
    # appearance z: [1.2247, 0, -1.2247] at (1, 0), (1, 100), (1, 200); action z: [1, -1] at (2, 0), (2, 100)
    h = {"appearance": hits([(1, 0, 0.3), (1, 100, 0.2), (1, 200, 0.1)]), "action": hits([(2, 0, 0.5), (2, 100, 0.4)])}
    c = fuse(h, {"appearance": 1.0, "action": 2.0}, DUR, CFG)
    # action x2 -> (2,0) 2.0 > (1,0) 1.2247 > (1,100) 0 > (1,200) -1.2247 > (2,100) -2.0
    assert [(x.video_id, x.first_hit_s) for x in c] == [(2, 0), (1, 0), (1, 100), (1, 200), (2, 100)]
    assert [x.score for x in c] == pytest.approx([2.0, 1.224745, 0.0, -1.224745, -2.0], abs=1e-6)
    c = fuse(h, {"appearance": 1.0, "action": 0.5}, DUR, CFG)
    # action x0.5 -> (1,0) 1.2247 > (2,0) 0.5 > (1,100) 0 > (2,100) -0.5 > (1,200) -1.2247
    assert [(x.video_id, x.first_hit_s) for x in c] == [(1, 0), (2, 0), (1, 100), (2, 100), (1, 200)]


def test_zero_weight_branch_is_excluded_not_zero_scored():
    h = {"appearance": hits([(1, 50, 0.3), (1, 150, 0.1)]), "action": hits([(2, 10, 0.9), (2, 20, 0.2)])}
    c = fuse(h, {"appearance": 1.0, "action": 0.0}, DUR, CFG)
    assert {x.video_id for x in c} == {1} and all(x.branches == ["appearance"] for x in c)


def test_branch_with_no_hits_and_no_hits_at_all():
    c = fuse({"appearance": hits([(1, 50, 0.3), (1, 150, 0.1)]), "action": hits([])},
             {"appearance": 1.0, "action": 1.0}, DUR, CFG)
    assert len(c) == 2 and all(x.branches == ["appearance"] for x in c)
    assert fuse({"appearance": hits([]), "action": hits([])}, {"appearance": 1.0, "action": 1.0}, DUR, CFG) == []


def test_overlapping_branches_merge_and_record_why():
    # appearance at 10 and 12, action window centred at 11 (same video): one clip, both branches, span 5..17
    h = {"appearance": hits([(1, 10, 0.30), (1, 12, 0.28), (1, 200, 0.10)]),
         "action": hits([(1, 11, 0.6), (1, 250, 0.2)])}
    c = fuse(h, {"appearance": 1.0, "action": 1.0}, DUR, CFG)
    top = c[0]
    assert (top.start_s, top.end_s, top.first_hit_s, top.last_hit_s) == (5.0, 17.0, 10, 12)
    assert top.branches == ["action", "appearance"]
    assert top.best["action"]["t_s"] == 11 and top.best["appearance"]["t_s"] == 10
    # a clip's score is the mean of its best 3 fused hits: appearance z of 0.30, 0.28 and action z of 0.6
    za = zscores([0.30, 0.28, 0.10])
    assert top.score == pytest.approx((za[0] + za[1] + 1.0) / 3)


def test_same_second_in_two_branches_counts_once():
    # both branches hit (1, 40): one hit with the better fused score, both branches recorded
    h = {"appearance": hits([(1, 40, 0.3), (1, 90, 0.1)]), "action": hits([(1, 40, 0.5), (1, 90, 0.1)])}
    c = fuse(h, {"appearance": 1.0, "action": 2.0}, DUR, CFG)
    top = c[0]
    assert (top.first_hit_s, top.last_hit_s) == (40, 40) and top.score == pytest.approx(2.0)
    assert top.branches == ["action", "appearance"]


def test_context_cut_at_video_start_and_end():
    c = fuse({"appearance": hits([(1, 2, 0.3), (1, 298, 0.2)])}, {"appearance": 1.0}, DUR, CFG)
    spans = sorted((x.start_s, x.end_s) for x in c)
    assert spans == [(0.0, 7.0), (293.0, 300.0)]


def test_mixed_branch_chain_is_capped():
    # alternating branches every 4 s from 0 to 72 in one video: no giant clip; first clip holds hits 0..60
    rows_a = [(1, t, 0.5) for t in range(0, 73, 8)]
    rows_b = [(1, t, 0.5) for t in range(4, 73, 8)]
    c = fuse({"appearance": hits(rows_a), "action": hits(rows_b)}, {"appearance": 1.0, "action": 1.0}, DUR, CFG)
    spans = sorted((x.first_hit_s, x.last_hit_s) for x in c)
    assert spans == [(0, 60), (64, 72)]
    assert all(x.last_hit_s - x.first_hit_s <= 60 for x in c)


def test_check_weights():
    assert check_weights({"appearance": 1.0, "action": 0, "events": 0}, ("appearance", "action")) == \
        {"appearance": 1.0, "action": 0.0}
    for bad in ("TBD", None, -1, True):
        with pytest.raises(ValueError, match="action"):
            check_weights({"appearance": 1.0, "action": bad}, ("appearance", "action"))
    with pytest.raises(ValueError, match="no weight"):
        fuse({"events": hits([(1, 0, 1.0)])}, {"appearance": 1.0}, DUR, CFG)
    assert not math.isnan(zscores([1, 1, 2])[0])


def test_top_n_returns_the_same_first_clips():
    h = {"appearance": hits([(1, t, 0.1 + t / 1000) for t in range(0, 300, 20)]),
         "action": hits([(2, t, 0.5 - t / 1000) for t in range(0, 300, 30)])}
    w = {"appearance": 1.0, "action": 2.0}
    full = fuse(h, w, DUR, CFG)
    top = fuse(h, w, DUR, CFG, top_n=3)
    assert len(top) == 3 and [(c.video_id, c.first_hit_s, c.score, c.branches) for c in top] == \
        [(c.video_id, c.first_hit_s, c.score, c.branches) for c in full[:3]]
