"""API helpers (rewind/api/app.py) without loading models: time parsing, selection checks, coverage (hand-computed).
The full app (encoders + engine + real indexes) is exercised by scripts/api_latency.py (step 7.3)."""

from datetime import datetime

import pytest
from fastapi import HTTPException

from rewind.api.app import Rewind, overlap_s, parse_time
from rewind.ingest.manifest import Video


def toy() -> Rewind:
    rw = Rewind.__new__(Rewind)                     # no models, no engine: only what coverage/selection use
    t = lambda h, m: datetime(2018, 3, 7, h, m)
    rw.videos = {1: Video(1, "a", "p", "G1", "rgb", t(10, 0), 9000, 30.0),     # 10:00-10:05
                 2: Video(2, "b", "p", "G1", "rgb", t(10, 5), 9000, 30.0),     # 10:05-10:10
                 3: Video(3, "c", "p", "G2", "ir", t(10, 0), 3000, 10.0)}      # 10:00-10:05
    rw.cameras = ["G1", "G2"]
    rw.branches = ("appearance", "action")
    rw.weights = {"appearance": 1.0, "action": 0.0}
    return rw


def test_overlap_hand_computed():
    a0, a1 = datetime(2018, 1, 1, 10, 0), datetime(2018, 1, 1, 10, 5)
    assert overlap_s(a0, a1, None, None) == 300
    assert overlap_s(a0, a1, datetime(2018, 1, 1, 10, 3), datetime(2018, 1, 1, 11)) == 120
    assert overlap_s(a0, a1, datetime(2018, 1, 1, 11), datetime(2018, 1, 1, 12)) == 0


def test_coverage_hand_computed():
    rw = toy()
    c = rw.coverage(["G1", "G2"], None, None)
    assert c["per_camera_hours"] == {"G1": round(600 / 3600, 2), "G2": round(300 / 3600, 2)}
    assert c["cameras"] == 2 and c["hours"] == round(900 / 3600, 2)
    assert c["branches"] == ["appearance"]                      # weight 0: not searched, not claimed
    # 10:03-10:07 on G1 only: 120 s of video 1 + 120 s of video 2 = 240 s
    c = rw.coverage(["G1"], datetime(2018, 3, 7, 10, 3), datetime(2018, 3, 7, 10, 7))
    assert c["per_camera_hours"] == {"G1": round(240 / 3600, 2)} and c["line"] == "Searched 0.1 h of footage across 1 camera."
    # a range with no footage: coverage says 0 h, 0 cameras (never a bare "no results")
    c = rw.coverage(["G1", "G2"], datetime(2018, 3, 8), datetime(2018, 3, 9))
    assert c["hours"] == 0 and c["cameras"] == 0 and c["line"].startswith("Searched 0.0 h")


def test_selection_errors():
    rw = toy()
    assert rw.selection(None, None, None) == (["G1", "G2"], None, None)
    assert rw.selection("G2", None, None)[0] == ["G2"]
    for args, msg in [(("G9", None, None), "unknown cameras"),
                      ((None, "2018-03-07T10:00:00", None), "both"),
                      ((None, "2018-03-07T11:00:00", "2018-03-07T10:00:00"), "after"),
                      ((None, "yesterday", "2018-03-07T10:00:00"), "ISO"),
                      ((None, "2018-03-07T10:00:00+01:00", "2018-03-07T11:00:00+01:00"), "time zone")]:
        with pytest.raises(HTTPException, match=msg):
            rw.selection(*args)
    assert parse_time(None, "start") is None
