"""Speed and vehicle_stops / vehicle_starts rules (rewind/track/tracks.py) on synthetic tracks, hand-computed."""

import numpy as np
import pandas as pd
import pytest

from rewind.track.tracks import (RuleParams, best_box, clip_events, detect_events, majority_class, runs,
                                 track_speed)

T = np.round(np.arange(20) * 0.2, 6)       # 20 samples at 5 fps: 0.0 .. 3.8 s


def boxes_at(cx: np.ndarray, h: float = 50.0, w: float = 100.0) -> np.ndarray:
    return np.stack([cx - w / 2, np.full_like(cx, 300.0), cx + w / 2, np.full_like(cx, 300.0 + h)], axis=1)


def test_speed_centred_window_and_units():
    # centre moves 10 px per 0.2 s = 50 px/s. smooth 0.6 -> half 0.3: sample 1 sees samples 0..2, dt 0.4,
    # 20 px / 0.4 s = 50 px/s. Sample 0 sees only 0..1 (dt 0.2 < 0.3) -> NaN; same for the last sample.
    t = T[:5]
    sp = track_speed(t, boxes_at(10.0 * np.arange(5), h=25.0), 0.6, "px_per_s")
    assert np.isnan(sp[0]) and np.isnan(sp[4])
    assert sp[1:4] == pytest.approx([50, 50, 50])
    # box height 25 px -> 50 / 25 = 2 box heights per second
    sp = track_speed(t, boxes_at(10.0 * np.arange(5), h=25.0), 0.6, "box_h_per_s")
    assert sp[1:4] == pytest.approx([2, 2, 2])
    # smooth 1.0 -> half 0.5: sample 2 sees samples 0..4, 40 px / 0.8 s = 50; samples 1 and 3 see 4 samples
    # (dt 0.6 >= 0.5) -> 30 px / 0.6 s = 50; samples 0 and 4 see 3 (dt 0.4 < 0.5) -> NaN
    sp = track_speed(t, boxes_at(10.0 * np.arange(5)), 1.0, "px_per_s")
    assert np.isnan(sp[0]) and np.isnan(sp[4]) and sp[1:4] == pytest.approx([50, 50, 50])


def test_speed_across_a_gap_uses_real_times():
    # samples at 0, 0.2, 1.2 (a 1 s gap), 1.4; x = 0, 10, 60, 70 (50 px/s throughout). smooth 0.6:
    # sample 1 sees 0..1 only (1.2 is 1.0 away) -> dt 0.2 < 0.3 -> NaN; no speed is computed across the gap.
    t = np.array([0.0, 0.2, 1.2, 1.4])
    sp = track_speed(t, boxes_at(np.array([0.0, 10.0, 60.0, 70.0])), 0.6, "px_per_s")
    assert np.isnan(sp).all()


def test_runs():
    assert runs(np.array([1, 1, 0, -1, 0, 0, 1])) == [(1, 0, 1), (0, 2, 2), (0, 4, 5), (1, 6, 6)]
    assert runs(np.array([-1, -1])) == []


P = RuleParams("px_per_s", stop_speed=20, min_stopped_s=1.0, min_moving_s=1.0, smooth_s=0.6)


def test_stop_rule():
    # moving at samples 0..9 (t 0..1.8, lasts 1.8 s), stopped 10..19 (t 2.0..3.8, 1.8 s) -> one stop, span [1.8, 2.0)
    speed = np.array([50.0] * 10 + [0.0] * 10)
    assert detect_events(T, speed, P) == [("vehicle_stops", pytest.approx(1.8), pytest.approx(2.0))]


def test_start_rule_is_the_reverse():
    speed = np.array([0.0] * 10 + [50.0] * 10)
    assert detect_events(T, speed, P) == [("vehicle_starts", pytest.approx(1.8), pytest.approx(2.0))]


def test_short_runs_give_no_event():
    # stopped run of samples 10..12: lasts 0.4 s < min_stopped_s 1.0 -> no stop; nor a start after it (0.4 s stop)
    speed = np.array([50.0] * 10 + [0.0] * 3 + [50.0] * 7)
    assert detect_events(T, speed, P) == []
    # exactly at the limit counts: stopped samples 10..15 last 1.0 s -> stop; moving 16..19 last 0.6 s < 1 -> no start
    speed = np.array([50.0] * 10 + [0.0] * 6 + [50.0] * 4)
    assert [e[0] for e in detect_events(T, speed, P)] == ["vehicle_stops"]


def test_none_sample_between_runs_blocks_the_transition():
    speed = np.array([50.0] * 9 + [np.nan] + [0.0] * 10)
    assert detect_events(T, speed, P) == []


def test_speed_equal_to_threshold_is_moving():
    speed = np.array([20.0] * 10 + [19.999] * 10)
    assert [e[0] for e in detect_events(T, speed, P)] == ["vehicle_stops"]


def track_df(x: np.ndarray, scores=None, classes=None) -> pd.DataFrame:
    b = boxes_at(x)
    n = len(x)
    return pd.DataFrame({"frame": np.arange(n) * 6, "t_s": T[:n], "track": 0,
                         "class": classes if classes is not None else ["car"] * n,
                         "score": scores if scores is not None else np.full(n, 0.5),
                         "x1": b[:, 0], "y1": b[:, 1], "x2": b[:, 2], "y2": b[:, 3]})


def test_clip_events_end_to_end():
    # x = 20 px per sample up to sample 10 (x = 200), then still. smooth 0.6 (neighbours i-1, i+1, dt 0.4 s):
    # samples 1..9: 40 / 0.4 = 100 px/s; sample 10: (200 - 180) / 0.4 = 50 -> moving; 11..18: 0 -> stopped;
    # 0 and 19: NaN. Moving run 1..10 (t 0.2..2.0, 1.8 s), stopped run 11..18 (t 2.2..3.6, 1.4 s) -> stop at [2.0, 2.2).
    x = np.minimum(np.arange(20) * 20.0, 200.0)
    scores = np.full(20, 0.5)
    scores[12] = 0.99                        # best box window [2.2 - 1.0, 2.2 + 1.0] = [1.2, 3.2] contains sample 12
    scores[2] = 1.0                          # higher, but outside the window
    ev = clip_events(track_df(x, scores), P)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert (e["event_type"], e["start_s"], e["end_s"], e["t_s"]) == ("vehicle_stops", pytest.approx(2.0),
                                                                      pytest.approx(2.2), pytest.approx(2.2))
    assert e["box_t_s"] == pytest.approx(2.4) and e["box_score"] == pytest.approx(0.99)
    assert e["box_frame"] == 12 * 6
    assert e["x1"] == pytest.approx(150.0)   # x = 200 at sample 12, box 100 px wide
    assert "track" not in ev.columns         # no track index leaves the tracks artifacts


def test_parked_car_with_jitter_gives_no_event():
    # +/- 2 px jitter: any displacement <= 4 px over 0.4 s = 10 px/s < 20 -> always stopped, no transition
    rng = np.random.default_rng(1)
    x = 500.0 + rng.uniform(-2, 2, size=20)
    assert clip_events(track_df(x), P).empty


def test_box_height_unit_makes_far_and_near_cars_equal():
    # the same motion in box heights per second: a near car (h 100 px) moving 40 px/s and a far one (h 20 px)
    # moving 8 px/s both move 0.4 h/s -> with threshold 0.2 h/s both are moving; with 20 px/s only the near one.
    t = T[:5]
    near = track_speed(t, boxes_at(8.0 * np.arange(5), h=100.0), 0.6, "box_h_per_s")
    far = track_speed(t, boxes_at(1.6 * np.arange(5), h=20.0), 0.6, "box_h_per_s")
    assert near[1:4] == pytest.approx([0.4] * 3) and far[1:4] == pytest.approx([0.4] * 3)
    assert track_speed(t, boxes_at(1.6 * np.arange(5), h=20.0), 0.6, "px_per_s")[2] == pytest.approx(8.0)


def test_best_box_tie_prefers_larger_area_and_majority_class_ties_alphabetical():
    tr = track_df(np.full(5, 100.0), scores=np.full(5, 0.7))
    tr.loc[3, ["x1", "x2"]] = [0.0, 300.0]    # same score, larger box
    assert best_box(tr, 0.4, P)["t_s"] == pytest.approx(0.6)
    assert majority_class(track_df(np.zeros(4), classes=["truck", "car", "truck", "car"])) == "car"


def test_rule_params_validation():
    with pytest.raises(ValueError):
        RuleParams("km_per_h", 1, 1, 1, 1)
    with pytest.raises(ValueError):
        RuleParams("px_per_s", 0, 1, 1, 1)
