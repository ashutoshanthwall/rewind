"""Search core (rewind/search/core.py): top-k, hit grouping, mean-centering. Hand-computed toy cases."""

from datetime import datetime

import numpy as np
import pytest

from rewind.ingest.manifest import Video
from rewind.search.core import Clip, SearchConfig, group_hits, mean_center, top_k

CFG = SearchConfig(k=100, merge_gap_s=5, max_clip_s=60, context_s=5)


def run(hits, durations=None):
    """hits: [(video_id, t_s, score)] -> clips."""
    vids = np.array([h[0] for h in hits])
    ts = np.array([h[1] for h in hits], float)
    sc = np.array([h[2] for h in hits], float)
    return group_hits(np.arange(len(hits)), sc, vids, ts, durations or {1: 300.0, 2: 300.0}, CFG)


def test_top_k_order_and_ties():
    v = np.array([[1, 0], [0, 1], [0.6, 0.8], [1, 0]], np.float16)
    idx, s = top_k(v, np.array([1.0, 0.0]), 3)
    assert idx.tolist() == [0, 3, 2] and s.tolist() == pytest.approx([1, 1, 0.6], abs=1e-3)   # tie: lower row first


def test_merge_gap_splits_and_mean_top3():
    # 0, 3 merge (gap 3 <= 5); 9 is 6 s after 3: new clip; 12 joins it
    clips = run([(1, 0, 0.5), (1, 3, 0.7), (1, 9, 0.9), (1, 12, 0.1)])
    assert [(c.start_s, c.end_s) for c in clips] == [(0.0, 8.0), (4.0, 17.0)]   # context 5, cut at 0
    assert clips[0].score == pytest.approx(0.6) and clips[1].score == pytest.approx(0.5)
    assert run([(1, 0, 0.9), (1, 1, 0.8), (1, 2, 0.1), (1, 3, 0.7)])[0].score == pytest.approx((0.9 + 0.8 + 0.7) / 3)


def test_max_clip_cap_prevents_chain_merging():
    # hits every 4 s from 0 to 72: the clip 0..60 holds 16 hits (60 - 0 = 60 is allowed); 64 starts a new clip
    clips = run([(1, t, 0.5) for t in range(0, 73, 4)])
    spans = sorted((c.start_s, c.end_s, len(c.hits)) for c in clips)
    assert spans == [(0.0, 65.0, 16), (59.0, 77.0, 3)]


def test_same_frame_vectors_are_one_hit_and_videos_never_merge():
    clips = run([(1, 10, 0.3), (1, 10, 0.8), (2, 12, 0.4)])
    assert len(clips) == 2 and clips[0].video_id == 1 and len(clips[0].hits) == 1 and clips[0].score == 0.8
    assert clips[0].hits[0][2] == 1          # row of the best vector of that frame


def test_context_cut_at_video_end():
    clips = run([(1, 298, 0.5)], durations={1: 300.33})
    assert (clips[0].start_s, clips[0].end_s) == (293.0, 300.33)


def test_clip_span_wall_clock():
    v = Video(1, "c", "p", "G1", "rgb", datetime(2018, 3, 7, 16, 50, 0), 9000, 30.0)
    s = Clip(1, 4.5, 17.0, 0.5).span(v)
    assert s.camera_id == "G1" and s.start == datetime(2018, 3, 7, 16, 50, 4, 500000) and s.seconds == 12.5


def test_mean_center_per_group():
    x = np.array([[1, 0], [0, 1], [3, 4], [3, 4.5]], np.float32)
    out = mean_center(x, np.array(["A", "A", "B", "B"]))
    # A: mean (0.5, 0.5) -> (0.5, -0.5), (-0.5, 0.5) -> normalised +-(0.707, -0.707)
    assert out[0] == pytest.approx([2 ** -0.5, -2 ** -0.5]) and out[1] == pytest.approx([-2 ** -0.5, 2 ** -0.5])
    # B: mean (3, 4.25) -> (0, -0.25), (0, 0.25) -> (0, -1), (0, 1)
    assert out[2] == pytest.approx([0, -1]) and out[3] == pytest.approx([0, 1])
    with pytest.raises(ValueError, match="camera mean"):
        mean_center(np.array([[1.0, 0.0]]), np.array(["A"]))
