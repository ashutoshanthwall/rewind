"""Step 2.3 metrics and baselines on toy cases. Every expected value is computed by hand in the comment next to it."""

import statistics
from datetime import datetime, timedelta

import pytest

from rewind.eval.baselines import (camera_orders, motion_ranked, segment_activity, segments, sequential, shuffled,
                                   witness)
from rewind.eval.metrics import (Event, Span, average_precision_at_k, bootstrap_ci, drop_dont_care, first_hit,
                                 instance_recall_at_k, match_events, overlaps, precision_recall, recall_at_k,
                                 seconds_watched)
from rewind.ingest.manifest import Video

T0 = datetime(2018, 3, 11, 10, 0, 0)


def sp(cam: str, a: float, b: float) -> Span:
    """Span on camera cam from T0 + a s to T0 + b s."""
    return Span(cam, T0 + timedelta(seconds=a), T0 + timedelta(seconds=b))


# --- spans and hits -----------------------------------------------------------------------------------------------

def test_overlap_is_half_open_same_camera_and_padded():
    assert overlaps(sp("A", 0, 60), sp("A", 59, 70))
    assert not overlaps(sp("A", 0, 60), sp("A", 60, 70))       # touching ends: no shared instant
    assert not overlaps(sp("A", 0, 60), sp("B", 10, 20))       # other camera never counts
    with pytest.raises(ValueError):
        sp("A", 5, 5)


def test_recall_at_k_and_first_hit():
    ranked = [sp("A", 0, 60), sp("B", 0, 60), sp("A", 60, 120)]
    gt = sp("A", 70, 75)                                        # only the third clip (rank 2) hits
    assert first_hit(ranked, gt) == 2
    assert [recall_at_k(ranked, gt, k) for k in (1, 2, 3)] == [0, 0, 1]
    assert first_hit(ranked, sp("C", 0, 10)) is None


def test_seconds_watched_counts_until_the_event_starts():
    ranked = [sp("B", 0, 60), sp("A", 0, 30), sp("A", 60, 120)]
    # GT on A at 80-90 s: watch B 60 s + A 30 s in full, then 80 - 60 = 20 s into the third clip -> 110 s.
    assert seconds_watched(ranked, sp("A", 80, 90), footage_s=1000) == 110
    # GT already under way when the clip starts (55-65 s overlaps A 60-120): 60 + 30 + 0 = 90 s.
    assert seconds_watched(ranked, sp("A", 55, 65), footage_s=1000) == 90


def test_seconds_watched_miss_rule():
    ranked = [sp("A", 0, 60), sp("A", 60, 90)]
    # No clip hits a GT on camera C: watched the whole list (60 + 30) + full review of 1000 s of footage = 1090.
    assert seconds_watched(ranked, sp("C", 10, 20), footage_s=1000) == 1090


# --- category queries -------------------------------------------------------------------------------------------

def test_instance_recall_and_average_precision():
    instances = [sp("A", 10, 20), sp("A", 100, 110), sp("B", 0, 10)]   # I1, I2, I3
    ranked = [sp("A", 0, 60),      # hits I1 (new)               -> rel 1, P = 1/1
              sp("B", 30, 60),     # nothing                     -> rel 0
              sp("A", 15, 30),     # hits I1 again (not new)     -> rel 0
              sp("A", 90, 120)]    # hits I2 (new)               -> rel 1, P = 2/4
    # AP@4 = (1 + 2/4) / min(4, 3) = 1.5 / 3 = 0.5;  AP@2 = 1 / min(2, 3) = 0.5;  AP@1 = 1 / 1 = 1
    assert average_precision_at_k(ranked, instances, 4) == pytest.approx(0.5)
    assert average_precision_at_k(ranked, instances, 2) == pytest.approx(0.5)
    assert average_precision_at_k(ranked, instances, 1) == pytest.approx(1.0)
    # Recall@1 = 1/3 (I1); Recall@4 = 2/3 (I1, I2)
    assert instance_recall_at_k(ranked, instances, 1) == pytest.approx(1 / 3)
    assert instance_recall_at_k(ranked, instances, 4) == pytest.approx(2 / 3)


def test_one_clip_covering_two_instances():
    instances = [sp("A", 10, 20), sp("A", 30, 40)]
    ranked = [sp("A", 0, 60)]
    # Recall counts both instances (2/2); AP: one relevant clip at rank 1, P = 1, divided by min(1, 2) = 1 -> 1.0
    assert instance_recall_at_k(ranked, instances, 1) == 1.0
    assert average_precision_at_k(ranked, instances, 1) == 1.0
    with pytest.raises(ValueError):
        average_precision_at_k(ranked, [], 1)


def test_drop_dont_care():
    instances = [sp("A", 10, 20)]
    dont_care = [sp("A", 100, 110), sp("A", 15, 18)]
    ranked = [sp("A", 90, 120),   # only don't-care -> dropped
              sp("A", 0, 60),     # instance (and a don't-care inside) -> kept
              sp("A", 200, 260)]  # nothing -> kept (a wrong answer)
    assert drop_dont_care(ranked, instances, dont_care) == [sp("A", 0, 60), sp("A", 200, 260)]


# --- detected vs annotated events --------------------------------------------------------------------------------

def test_event_matching_is_one_to_one_with_tolerance():
    ann = [Event(sp("A", 10, 14), "vehicle_stops"), Event(sp("A", 50, 54), "vehicle_stops"),
           Event(sp("A", 50, 54), "vehicle_starts")]
    det = [Event(sp("A", 11, 12), "vehicle_stops"),     # overlaps ann 0
           Event(sp("A", 12, 13), "vehicle_stops"),     # also only ann 0 -> already taken, unmatched
           Event(sp("A", 55, 56), "vehicle_stops"),     # ann 1 ends 54; with 2 s tolerance 56 > 55 -> matches
           Event(sp("B", 50, 54), "vehicle_starts")]    # other camera -> unmatched
    # matches: det0-ann0 and det2-ann1 = 2 (det1 loses ann0: both overlap 1 s, det0 comes first in the list)
    assert match_events(det, ann, tolerance_s=2.0) == 2
    # precision 2/4 = 0.5, recall 2/3
    assert precision_recall(det, ann, tolerance_s=2.0) == pytest.approx((0.5, 2 / 3))
    # without tolerance det2 (55-56) misses ann1 (50-54): 1 match -> precision 1/4, recall 1/3
    assert precision_recall(det, ann, tolerance_s=0.0) == pytest.approx((0.25, 1 / 3))
    with pytest.raises(ValueError):
        precision_recall([], ann, 0.0)


def test_tolerance_is_applied_once():
    ann = [Event(sp("A", 50, 54), "vehicle_stops")]
    # tolerance 2 s widens the label to 48-56: a detection at 55-56 matches (1 s inside), one at 57-58 does not
    # (3 s after the label's end > 2 s), and neither does 45-47.9 (ends before 48).
    assert match_events([Event(sp("A", 55, 56), "vehicle_stops")], ann, 2.0) == 1
    assert match_events([Event(sp("A", 57, 58), "vehicle_stops")], ann, 2.0) == 0
    assert match_events([Event(sp("A", 45, 47.9), "vehicle_stops")], ann, 2.0) == 0


def test_greedy_matching_prefers_larger_overlap():
    ann = [Event(sp("A", 0, 10), "x")]
    det = [Event(sp("A", 9, 12), "x"), Event(sp("A", 2, 8), "x")]   # overlaps 1 s and 6 s
    # the 6 s pair is taken first, so exactly 1 match either way; precision 1/2, recall 1/1
    assert precision_recall(det, ann, 0.0) == (0.5, 1.0)


# --- bootstrap ----------------------------------------------------------------------------------------------------

def test_bootstrap_ci():
    assert bootstrap_ci([3.0, 3.0, 3.0], statistics.median, 200, seed=1, level=0.95) == (3.0, 3.0)
    vals = [1.0, 2.0, 3.0, 4.0, 100.0]
    lo, hi = bootstrap_ci(vals, statistics.median, 2000, seed=1, level=0.95)
    # Medians of resamples of 5 values can only be one of the values; the 2.5 % and 97.5 % quantiles lie in [1, 100]
    # and must bracket the sample median 3.
    assert 1.0 <= lo <= 3.0 <= hi <= 100.0
    assert bootstrap_ci(vals, statistics.median, 2000, seed=1, level=0.95) == (lo, hi)   # seeded
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], statistics.median, 10, seed=1, level=0.95)


# --- baselines ------------------------------------------------------------------------------------------------------

def video(vid: int, cam: str, start_s: float, seconds: float) -> Video:
    return Video(vid, f"c{vid}", "", cam, "rgb", T0 + timedelta(seconds=start_s), round(seconds * 30), 30.0)


def test_segments_are_equal_pieces_without_slivers():
    # 180 s / 60 -> 3 pieces of 60 s; 150 s / 60 = 2.5 -> round half to even = 2 pieces of 75 s
    assert segments([video(1, "A", 0, 180)], 60) == [sp("A", 0, 60), sp("A", 60, 120), sp("A", 120, 180)]
    assert segments([video(1, "A", 0, 150)], 60) == [sp("A", 0, 75), sp("A", 75, 150)]
    # 300.333 s (9010 frames at 30 fps, as G330 clips) -> 5 pieces of 60.0667 s, the last ending at the clip end
    segs = segments([video(1, "A", 0, 9010 / 30)], 60)
    assert len(segs) == 5 and segs[-1].end == T0 + timedelta(seconds=9010 / 30)
    assert all(abs(s.seconds - 9010 / 150) < 1e-5 for s in segs)
    # 20 s clip -> max(1, round(0.33)) = 1 piece
    assert segments([video(1, "A", 0, 20)], 60) == [sp("A", 0, 20)]


def test_sequential_averaged_over_camera_orders():
    # Two cameras with 120 s each (2 segments). GT on B at 30-40 s.
    # Order A,B: 120 s of A + 30 s into B = 150.  Order B,A: 30.  Mean over both orders = 90.
    segs = segments([video(1, "A", 0, 120), video(2, "B", 0, 120)], 60)
    gt = sp("B", 30, 40)
    orders = camera_orders(segs)
    assert orders == [("A", "B"), ("B", "A")]
    watched = [seconds_watched(sequential(segs, o), gt, footage_s=240) for o in orders]
    assert watched == [150, 30] and sum(watched) / 2 == 90


def test_witness_window_first_then_the_rest():
    # Camera A 0-300 s (5 segments), B 0-300 s. Witness time t = 250 s, window 60 s -> [190, 310):
    # in-window segments in order (A, B): A 180-240, A 240-300, B 180-240, B 240-300; then the rest sequentially.
    segs = segments([video(1, "A", 0, 300), video(2, "B", 0, 300)], 60)
    ranked = witness(segs, T0 + timedelta(seconds=250), window_s=60, order=("A", "B"))
    assert ranked[:4] == [sp("A", 180, 240), sp("A", 240, 300), sp("B", 180, 240), sp("B", 240, 300)]
    assert ranked[4:7] == [sp("A", 0, 60), sp("A", 60, 120), sp("A", 120, 180)]
    assert len(ranked) == len(segs) == 10
    # GT on B at 250-255: watch A 180-300 (120 s) + B 180-240 (60 s) + 10 s into B 240-300 = 190 s
    assert seconds_watched(ranked, sp("B", 250, 255), footage_s=600) == 190


def test_segment_activity_weights_seconds_by_overlap():
    v = video(1, "A", 0, 4.5)              # 135 frames: seconds 0, 1, 2, 3 full, second 4 covers [4, 4.5)
    act = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert segment_activity(v, sp("A", 0, 2), act) == pytest.approx(0.5)            # (0 + 1) / 2
    # [1.5, 3.25): second 1 x 0.5, second 2 x 1, second 3 x 0.25 -> (0.5 + 2 + 0.75) / 1.75
    assert segment_activity(v, sp("A", 1.5, 3.25), act) == pytest.approx(3.25 / 1.75)
    # [4, 4.5): only the half second that exists -> 4.0
    assert segment_activity(v, sp("A", 4, 4.5), act) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="not inside clip"):
        segment_activity(v, sp("A", 3, 6), act)
    with pytest.raises(ValueError, match="not inside clip"):
        segment_activity(v, sp("B", 0, 1), act)
    with pytest.raises(ValueError, match="per-second scores"):
        segment_activity(v, sp("A", 0, 1), act[:4])


def test_motion_ranked_descending_with_deterministic_ties():
    a1, a2, b1 = sp("A", 0, 60), sp("A", 60, 120), sp("B", 0, 60)
    assert motion_ranked({a1: 0.1, a2: 0.5, b1: 0.3}) == [a2, b1, a1]
    assert motion_ranked({b1: 0.0, a2: 0.0, a1: 0.0}) == [a1, a2, b1]    # ties: camera, then time
    # GT on B 10-20 s: watch A 60-120 (60 s) + 10 s into B 0-60 = 70 s
    assert seconds_watched(motion_ranked({a1: 0.1, a2: 0.5, b1: 0.3}), sp("B", 10, 20), footage_s=180) == 70


def test_random_order_is_seeded_and_complete():
    segs = segments([video(1, "A", 0, 600), video(2, "B", 0, 600)], 60)
    a, b = shuffled(segs, "q1-0"), shuffled(list(reversed(segs)), "q1-0")
    assert a == b                                  # same seed, any input order -> same ranking
    assert sorted(a, key=lambda s: (s.camera_id, s.start)) == sorted(segs, key=lambda s: (s.camera_id, s.start))
    assert shuffled(segs, "q1-1") != a


# --- the baseline runner's ground truth and averaging ---------------------------------------------------------------

def test_category_truth_joins_pieces_and_separates_dont_care():
    import sqlite3
    from rewind.eval.splits import SCHEMA as SPLIT_SCHEMA
    from rewind.ingest.manifest import connect
    from scripts.run_baselines import category_truth
    conn = connect(":memory:")
    conn.executescript(SPLIT_SCHEMA)
    iso = lambda s: (T0 + timedelta(seconds=s)).isoformat()
    def ev(eid, group, cam, a, b, split, boundary=None, kind="vehicle_stops", source="kitware-eval"):
        conn.execute("INSERT INTO events VALUES (?, 1, ?, ?, 0, 1, 0.0, 1.0, ?, ?, ?, '[]', '[]', NULL, NULL, 1, 1, "
                     "0, 0, ?)", (eid, cam, kind, iso(a), iso(b), source, group))
        conn.execute("INSERT INTO event_splits VALUES (?, ?, ?)", (eid, split, boundary))
    ev(1, 1, "A", 0, 5, "dev")                                 # instance
    ev(2, 2, "A", 298, 300, "dev"); ev(3, 2, "A", 300, 303, "dev")   # one activity cut in 2 pieces -> one instance
    ev(4, 4, "B", 10, 20, "dev", boundary="overlaps X")        # boundary-flagged -> don't care
    ev(5, 5, "A", 50, 60, "tuning")                            # other split -> ignored
    ev(6, 6, "G474", 30, 40, "dev", source="contrib")          # IR third-party label -> ignored
    ev(7, 7, "A", 70, 80, "dev", kind="vehicle_starts")        # other type -> ignored
    instances, dont_care = category_truth(conn, "dev", "vehicle_stops")
    assert sorted(instances, key=lambda s: s.start) == [sp("A", 0, 5), sp("A", 298, 303)]
    assert dont_care == [sp("B", 10, 20)]
    with pytest.raises(ValueError, match="no vehicle_starts"):
        category_truth(conn, "tuning", "vehicle_starts")


def test_specific_metrics_average_over_rankings():
    from scripts.run_baselines import specific_metrics
    gt = sp("A", 10, 20)
    lists = [[sp("A", 0, 60)],                       # hit at rank 0, 10 s into the clip
             [sp("B", 0, 60), sp("A", 0, 60)]]        # hit at rank 1: 60 + 10 = 70 s
    # R@1 = mean(1, 0) = 0.5; R@2 = mean(1, 1) = 1; seconds = mean(10, 70) = 40
    assert specific_metrics(lists, gt, footage_s=1000, ks=[1, 2]) == {"R@1": 0.5, "R@2": 1.0, "seconds": 40.0}
