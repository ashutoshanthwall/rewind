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


# --- step 9.1 endpoints (no models: a toy Rewind with real helper code) ---

from datetime import timedelta

import cv2
import numpy as np
from fastapi.testclient import TestClient

from rewind.api.app import create_app, draw_thumb, frame_at, merged_spans
from rewind.eval.metrics import Span


def test_merged_spans_hand_computed():
    rw = toy()
    t = lambda h, m: datetime(2018, 3, 7, h, m)
    # G1: 10:00-10:05 and 10:05-10:10 touch -> one span; G2: 10:00-10:05
    assert merged_spans(rw.videos.values(), 1.0) == {"G1": [[t(10, 0), t(10, 10)]], "G2": [[t(10, 0), t(10, 5)]]}
    gap = {**rw.videos, 4: Video(4, "d", "p", "G2", "ir", t(10, 6), 3000, 10.0)}     # 1 min after G2's end
    assert merged_spans(gap.values(), 1.0)["G2"] == [[t(10, 0), t(10, 5)], [t(10, 6), t(10, 11)]]
    assert merged_spans(gap.values(), 60.0)["G2"] == [[t(10, 0), t(10, 11)]]       # within join gap: merged


def motion_toy() -> "Rewind":
    rw = toy()
    t = lambda h, m: datetime(2018, 3, 7, h, m)
    seg = lambda cam, h, m: Span(cam, t(h, m), t(h, m) + timedelta(minutes=1))
    rw.motion = {seg("G1", 10, 0): 0.1, seg("G1", 10, 6): 0.9, seg("G2", 10, 2): 0.5, seg("G1", 10, 3): 0.5}
    rw.seg_video = {seg("G1", 10, 0): 1, seg("G1", 10, 3): 1, seg("G1", 10, 6): 2, seg("G2", 10, 2): 3}
    rw.proxies = {1: "a.mp4", 2: "b.mp4", 3: "c.mp4"}
    return rw


def test_motion_list_ranks_and_filters():
    rw = motion_toy()
    out = rw.motion_list(["G1", "G2"], None, None, 3)
    # by activity, ties by camera then time (step 3.2 rule): 0.9 G1 10:06, then 0.5 G1 10:03 before 0.5 G2 10:02
    assert [(s["camera"], s["wall_start"][11:16]) for s in out["segments"]] == [("G1", "10:06"), ("G1", "10:03"),
                                                                                ("G2", "10:02")]
    first = out["segments"][0]
    assert first["video_id"] == 2 and first["start_s"] == 60 and first["end_s"] == 120   # video 2 starts 10:05
    assert out["of"] == 4
    # window 10:02:30-10:04 on G1: only the 10:03 segment overlaps; G2 not selected
    out = rw.motion_list(["G1"], datetime(2018, 3, 7, 10, 2, 30), datetime(2018, 3, 7, 10, 4), 5)
    assert [s["wall_start"][11:16] for s in out["segments"]] == ["10:03"] and out["of"] == 1


def test_draw_thumb_scales_box_to_proxy():
    img = np.zeros((360, 640, 3), np.uint8)                   # proxy frame, half of a 1280-wide original
    jpg = draw_thumb(img, (200, 100, 400, 300), 0.5, 320)     # original box -> proxy (100, 50)-(200, 150)
    out = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (180, 320, 3)                         # resized to width 320, aspect kept
    # thumbnail scale 0.5 again: the box edge x = 100 * 0.5 = 50 is drawn (bright), the centre stays black
    assert out[50, 50].max() > 100 and out[50, 75].max() < 40
    assert draw_thumb(img, None, 0.5, 320)                    # no box: still a JPEG


def test_frame_at_returns_the_frame_at_t(tmp_path):
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(12):                                       # frame i: grey level 20 * i, at i / 10 s
        w.write(np.full((48, 64, 3), 20 * i, np.uint8))       # (20 apart: lossy mp4v moves levels by a few)
    w.release()
    # t = 0.73 s: the first frame with time >= 0.73 - 0.05 = 0.68 is frame 7 (0.7 s)
    for t, i in ((0.0, 0), (0.5, 5), (0.73, 7), (1.1, 11)):
        assert abs(int(frame_at(path, t).mean()) - 20 * i) <= 8, t
    with pytest.raises(ValueError, match="no frame"):
        frame_at(path, 3.0)


def test_endpoints_with_toy_rewind(tmp_path, monkeypatch):
    import rewind.api.app as appmod
    rw = motion_toy()
    rw.default_clips, rw.max_clips = 20, 100
    rw.ui = {"window_minutes": [5, 10], "default_window_minutes": 10, "fallback_motion_k": 2, "thumb_width": 320}
    rw.join_s = 1.0
    rw.modality = {"G1": "rgb", "G2": "ir"}
    clip = tmp_path / "a.mp4"
    clip.write_bytes(bytes(range(256)) * 4)                   # 1024 bytes: enough to check byte ranges
    monkeypatch.setattr(appmod, "REPO_ROOT", tmp_path)
    c = TestClient(create_app(rw=rw))
    f = c.get("/footage").json()
    assert [x["camera"] for x in f["cameras"]] == ["G1", "G2"] and f["cameras"][1]["modality"] == "ir"
    assert f["cameras"][0]["hours"] == round(600 / 3600, 2) and f["branches"] == ["appearance"]
    assert f["ui"] == {"window_minutes": [5, 10], "default_window_minutes": 10, "default_clips": 20, "max_clips": 100}
    m = c.get("/motion").json()
    assert len(m["segments"]) == 2 and m["coverage"]["line"].startswith("Searched")      # default k from config
    assert c.get("/motion?k=101").status_code == 400
    r = c.get("/proxy/1", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))                     # seeking works
    assert c.get("/proxy/99").status_code == 404
    assert c.get("/thumb?video_id=1&t=999").status_code == 400                            # outside the clip
    assert c.get("/thumb?video_id=1&t=1&box=1,2,3").status_code == 400
