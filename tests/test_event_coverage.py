"""scripts/event_coverage.py helpers on toy cases (hand-computed)."""

import pandas as pd

from scripts.event_coverage import fires_near, frame_coverage


def boxes(rows):
    return pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2"])


def test_frame_coverage():
    # label on frames 0, 6, 12 (frame 6 has two actors). Track boxes: frame 0 identical (IoU 1); frame 6 overlaps the
    # second actor [100,0,110,10] with [102,0,112,10] -> IoU 80 / 120 = 0.667; frame 12 only [5,0,15,10] vs
    # [0,0,10,10] -> IoU 50 / 150 = 0.333 (>= 0.3 covered, < 0.4 not).
    label = boxes([(0, 0, 0, 10, 10), (6, 0, 0, 10, 10), (6, 100, 0, 110, 10), (12, 0, 0, 10, 10)])
    tracks = boxes([(0, 0, 0, 10, 10), (6, 102, 0, 112, 10), (12, 5, 0, 15, 10), (18, 0, 0, 10, 10)])
    assert frame_coverage(label, tracks, 0.3) == (3, 3)
    assert frame_coverage(label, tracks, 0.4) == (3, 2)
    assert frame_coverage(label, boxes([]), 0.3) == (3, 0)


def test_fires_near():
    det = pd.DataFrame({"event_type": ["vehicle_stops", "vehicle_starts"], "start_s": [10.0, 30.0],
                        "end_s": [10.2, 30.2]})
    # label [12, 15) widened by 2 s -> [10, 17): the stop [10.0, 10.2) overlaps
    assert fires_near(det, "vehicle_stops", 12.0, 15.0, 2.0)
    # widened by 1 s -> [11, 16): no overlap
    assert not fires_near(det, "vehicle_stops", 12.0, 15.0, 1.0)
    # same time, other type
    assert not fires_near(det, "vehicle_starts", 12.0, 15.0, 2.0)
