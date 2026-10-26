"""scripts/event_sweep.py helpers on toy cases (hand-computed expectations in comments)."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from rewind.ingest.manifest import Video
from rewind.track.tracks import RuleParams
from scripts.event_sweep import ClipTracks, counts, extend_at_edges, grid, prf, to_us

V = Video(1, "clip", "p", "A", "rgb", datetime(2018, 3, 7, 10, 0, 0), n_frames=60 * 30, fps=30.0)   # 60 s clip
P = RuleParams("px_per_s", stop_speed=20, min_stopped_s=1.0, min_moving_s=1.0, smooth_s=0.6)
SWEEP = {"speed_unit": ["px_per_s", "box_h_per_s"],
         "stop_speed": {"px_per_s": [5, 10, 20, 40], "box_h_per_s": [0.05, 0.1, 0.2, 0.4]},
         "min_stopped_s": [0.6, 1, 2], "min_moving_s": [0.6, 1, 2], "smooth_s": [0.6, 1.0],
         "min_p": 0.5, "min_r": 0.5, "shift_control_s": 30}


def stopping_track() -> pd.DataFrame:
    # same track as tests/test_track_rules.test_clip_events_end_to_end: one stop, span [2.0, 2.2) s in the clip
    x = np.minimum(np.arange(20) * 20.0, 200.0)
    t = np.round(np.arange(20) * 0.2, 6)
    return pd.DataFrame({"frame": np.arange(20) * 6, "t_s": t, "track": 0, "class": "car", "score": 0.5,
                         "x1": x - 50, "y1": 300.0, "x2": x + 50, "y2": 350.0})


def test_prf():
    assert prf(4, 5, 2) == pytest.approx((0.5, 0.4, 2 * 0.5 * 0.4 / 0.9))
    p, r, f = prf(0, 5, 0)
    assert p != p and r == 0 and f == 0          # P undefined (NaN) without detections
    assert prf(3, 3, 0) == (0.0, 0.0, 0.0)


def test_events_and_shift_control():
    c = ClipTracks(V, stopping_track())
    base = to_us(V.start_time)
    ev = c.events(P)
    assert ev["vehicle_stops"].tolist() == [[base + 2_000_000, base + 2_200_000]]
    assert len(ev["vehicle_starts"]) == 0
    # +30 s: [32.0, 32.2); +59 s: 61.0 >= 60 s duration -> wraps to [1.0, 1.2)
    assert c.events(P, shift_s=30)["vehicle_stops"].tolist() == [[base + 32_000_000, base + 32_200_000]]
    assert c.events(P, shift_s=59)["vehicle_stops"].tolist() == [[base + 1_000_000, base + 1_200_000]]


def test_counts_with_tolerance_and_shift():
    c = ClipTracks(V, stopping_track())
    base = to_us(V.start_time)
    # label [4.0, 6.0) s widened by 2 s -> [2.0, 8.0): the detection [2.0, 2.2) overlaps by 0.2 s -> matched
    ann = {("A", "vehicle_stops"): np.array([[base + 4_000_000, base + 6_000_000]])}
    got = counts([c], ann, P, tol_us=2_000_000)
    assert got[("A", "vehicle_stops")] == [1, 1, 1]
    assert got[("A", "vehicle_starts")] == [0, 0, 0]
    # tolerance 1 s -> [3.0, 7.0): no overlap with [2.0, 2.2)
    assert counts([c], ann, P, tol_us=1_000_000)[("A", "vehicle_stops")] == [1, 1, 0]
    # shifted +30 s: [32.0, 32.2) -> no match
    assert counts([c], ann, P, tol_us=2_000_000, shift_s=30)[("A", "vehicle_stops")] == [1, 1, 0]


def test_grid_size_and_order():
    g = grid(SWEEP)
    assert len(g) == 2 * 4 * 3 * 3 * 2 == 144
    assert g[0] == RuleParams("px_per_s", 5.0, 0.6, 0.6, 0.6)
    assert g[1] == RuleParams("px_per_s", 5.0, 0.6, 0.6, 1.0)      # smooth_s varies fastest
    assert g[-1] == RuleParams("box_h_per_s", 0.4, 2.0, 2.0, 1.0)


def test_edge_extension():
    # interior choice: nothing to extend
    new, notes = extend_at_edges(SWEEP, RuleParams("px_per_s", 10.0, 1.0, 1.0, 0.6))
    assert notes == ["smooth_s at low edge -> added 0.4"] and new["smooth_s"] == [0.4, 0.6, 1.0]
    # speed at the top (x2), min_stopped at the top (+1 s), min_moving at the bottom (-0.2 s), smooth at the top (+0.6)
    new, notes = extend_at_edges(SWEEP, RuleParams("box_h_per_s", 0.4, 2.0, 0.6, 1.0))
    assert new["stop_speed"]["box_h_per_s"] == [0.05, 0.1, 0.2, 0.4, 0.8]
    assert new["stop_speed"]["px_per_s"] == [5, 10, 20, 40]          # other unit untouched
    assert new["min_stopped_s"] == [0.6, 1.0, 2.0, 3.0] and new["min_moving_s"] == [0.4, 0.6, 1.0, 2.0]
    assert new["smooth_s"] == [0.6, 1.0, 1.6]
    assert SWEEP["stop_speed"]["box_h_per_s"] == [0.05, 0.1, 0.2, 0.4]   # the config dict is not modified
    # speed at the bottom (/2)
    new, _ = extend_at_edges(SWEEP, RuleParams("px_per_s", 5.0, 1.0, 1.0, 0.6))
    assert new["stop_speed"]["px_per_s"] == [2.5, 5, 10, 20, 40]
