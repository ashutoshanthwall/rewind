"""Vendored ByteTrack (rewind/track/bytetrack_vendor) and rewind.track.tracks.track_clip on synthetic detections."""

import itertools

import numpy as np
import pandas as pd
import pytest

from rewind.track.bytetrack_vendor import BYTETracker
from rewind.track.bytetrack_vendor.matching import ious, linear_assignment
from rewind.track.tracks import TrackerConfig, merge_classes, track_clip

CFG = TrackerConfig(track_thresh=0.5, match_thresh=0.8, track_buffer=30, mot20=False, frame_rate=5,
                    classes=("car", "truck", "bus", "motorcycle"), merge_iou=0.65)
FPS = 30.0          # video frame rate; samples every 6th frame = 5 fps
STEP = 6


def brute_force(cost: np.ndarray, thresh: float) -> tuple[float, set]:
    """Minimum over all partial one-to-one matchings of sum(cost - thresh) over matched pairs (lap's objective)."""
    n, m = cost.shape
    best = (0.0, set())
    for choice in itertools.product(range(-1, m), repeat=n):
        cols = [c for c in choice if c >= 0]
        if len(cols) != len(set(cols)):
            continue
        pairs = {(i, c) for i, c in enumerate(choice) if c >= 0}
        val = sum(cost[i, c] - thresh for i, c in pairs)
        if val < best[0] - 1e-12:
            best = (val, pairs)
    return best


def test_linear_assignment_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n, m = rng.integers(1, 5, size=2)
        cost = rng.uniform(0, 1.2, size=(n, m))
        val, pairs = brute_force(cost, 0.8)
        matches, ua, ub = linear_assignment(cost, thresh=0.8)
        got = {(int(i), int(j)) for i, j in matches}
        assert got == pairs           # continuous random costs: the optimum is unique
        assert set(ua.tolist()) == set(range(n)) - {i for i, _ in got}
        assert set(ub.tolist()) == set(range(m)) - {j for _, j in got}


def test_linear_assignment_hand_cases():
    # 0.1 + 0.2 = 0.3 is the best total; both pairs are under 0.8 -> both matched
    m, ua, ub = linear_assignment(np.array([[0.1, 0.9], [0.85, 0.2]]), thresh=0.8)
    assert sorted(map(tuple, m.tolist())) == [(0, 0), (1, 1)] and len(ua) == len(ub) == 0
    # a single pair above the threshold stays unmatched
    m, ua, ub = linear_assignment(np.array([[0.9]]), thresh=0.8)
    assert len(m) == 0 and ua.tolist() == [0] and ub.tolist() == [0]
    # empty matrix
    m, ua, ub = linear_assignment(np.zeros((0, 3)), thresh=0.8)
    assert m.shape == (0, 2) and ub == (0, 1, 2)


def test_ious_plus_one_convention():
    # [0,0,9,9] and [5,5,14,14]: each 10 x 10 = 100 px (inclusive pixels); overlap 5 x 5 = 25; IoU = 25 / 175
    got = ious([[0, 0, 9, 9]], [[5, 5, 14, 14]])
    assert got.shape == (1, 1) and got[0, 0] == pytest.approx(25 / 175)


def test_trackers_do_not_share_id_counter():
    """The upstream global BaseTrack._count was replaced by a per-tracker counter: each tracker starts at id 1."""
    args = type("A", (), {"track_thresh": 0.5, "track_buffer": 30, "match_thresh": 0.8, "mot20": False})()
    det = np.array([[0, 0, 100, 50, 0.9]], dtype=float)
    a, b = BYTETracker(args, frame_rate=5), BYTETracker(args, frame_rate=5)
    assert [t.track_id for t in a.update(det.copy())] == [1]
    assert [t.track_id for t in b.update(det.copy())] == [1]


def dets_frame(f: int, boxes: list) -> list:
    return [(f, f / FPS, cls, score, *box) for cls, score, box in boxes]


def frame_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["frame", "t_s", "class", "score", "x1", "y1", "x2", "y2"])


def test_two_vehicles_two_tracks_indexed_by_first_appearance():
    # A (score 0.95) moves right 10 px per sample from sample 0; B (0.9) moves left, far away, from sample 3.
    # ByteTrack confirms tracks born on its first frame at once (A: all 20 samples output); a later track is
    # confirmed by its second match (B: born at sample 3, output from sample 4 -> 16 rows). A is index 0.
    rows = []
    frames = np.arange(0, 20 * STEP, STEP)
    for k, f in enumerate(frames):
        boxes = [("car", 0.95, (100 + 10 * k, 100, 200 + 10 * k, 150))]
        if k >= 3:
            boxes.append(("car", 0.90, (1500 - 10 * k, 600, 1600 - 10 * k, 650)))
        rows += dets_frame(int(f), boxes)
    out = track_clip(frame_df(rows), frames, FPS, CFG)
    assert sorted(out["track"].unique()) == [0, 1]
    assert out.groupby("track").size().tolist() == [20, 16]
    assert out[out["track"] == 1]["frame"].min() == 4 * STEP
    a = out[out["track"] == 0].sort_values("t_s")
    assert a["x1"].tolist() == [100 + 10 * k for k in range(20)]      # the detection's own box, not the Kalman box
    assert a["score"].iloc[0] == pytest.approx(0.95)


def test_dropout_shorter_than_buffer_keeps_track_and_class_flicker_does_not_split():
    # One vehicle, 20 samples, missing at samples 10..12 (3 samples < the 5-sample buffer); class alternates
    # car / truck. Expected: one track, 17 rows.
    rows = []
    frames = np.arange(0, 20 * STEP, STEP)
    for k, f in enumerate(frames):
        if 10 <= k <= 12:
            continue
        rows += dets_frame(int(f), [("car" if k % 2 == 0 else "truck", 0.9, (100 + 5 * k, 100, 300 + 5 * k, 200))])
    out = track_clip(frame_df(rows), frames, FPS, CFG)
    assert out["track"].unique().tolist() == [0]
    assert len(out) == 17
    assert set(out["class"]) == {"car", "truck"}


def test_car_truck_duplicate_boxes_are_merged_and_persons_ignored():
    rows = []
    frames = np.arange(0, 10 * STEP, STEP)
    for k, f in enumerate(frames):
        rows += dets_frame(int(f), [("car", 0.9, (100, 100, 300, 200)), ("truck", 0.6, (102, 101, 301, 200)),
                                    ("person", 0.95, (500, 100, 540, 200))])
    out = track_clip(frame_df(rows), frames, FPS, CFG)
    assert out["track"].unique().tolist() == [0] and len(out) == 10 and set(out["class"]) == {"car"}


def test_merge_classes_keeps_highest_score():
    # IoU([0,0,10,10], [1,0,11,10]) = 90 / 110 = 0.818 >= 0.65 -> merged; [50,50,60,60] is separate
    d = frame_df([(0, 0.0, "truck", 0.5, 0, 0, 10, 10), (0, 0.0, "car", 0.8, 1, 0, 11, 10),
                  (0, 0.0, "bus", 0.4, 50, 50, 60, 60)])
    kept = merge_classes(d, 0.65)
    assert kept["class"].tolist() == ["car", "bus"]


def test_off_grid_detection_frame_raises():
    frames = np.arange(0, 10 * STEP, STEP)
    with pytest.raises(ValueError, match="not on the sample grid"):
        track_clip(frame_df(dets_frame(7, [("car", 0.9, (0, 0, 10, 10))])), frames, FPS, CFG)


def test_empty_clip_gives_empty_frame():
    out = track_clip(frame_df([]), np.arange(0, 10 * STEP, STEP), FPS, CFG)
    assert out.empty and list(out.columns) == ["frame", "t_s", "track", "class", "score", "x1", "y1", "x2", "y2"]
