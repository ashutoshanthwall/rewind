"""Phase 3 full pass: prefetch thread, settings hash, one clip end to end with a fake detector."""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from rewind.detect.detectors import Detections
from rewind.ingest.manifest import Video
from rewind.motion.mog2 import MotionConfig
from scripts.index_pass import load_detectors, prefetch, process_clip, settings_hash
from tests.test_manifest import write_avi

MOTION = {"fps": 5, "downscale_width": 32, "history_frames": 500, "var_threshold": 16, "detect_shadows": True,
          "warmup_s": 1, "open_kernel_px": 3, "merge_dilate_px": 3, "min_box_area_px": 4}


def test_prefetch_keeps_order_and_reraises():
    assert list(prefetch(iter(range(100)), maxsize=3)) == list(range(100))

    def broken():
        yield 1
        raise RuntimeError("decode failed")
    it = prefetch(broken(), maxsize=2)
    assert next(it) == 1
    with pytest.raises(RuntimeError, match="decode failed"):
        next(it)


def test_settings_hash_tracks_motion_and_detection_only():
    cfg = {"motion": {"fps": 5}, "detection": {"conf_thresh": 0.3}, "search": {"k": 1}}
    h = settings_hash(cfg)
    assert settings_hash({**cfg, "search": {"k": 2}}) == h
    assert settings_hash({**cfg, "detection": {"conf_thresh": 0.25}}) != h
    assert settings_hash({**cfg, "motion": {"fps": 1}}) != h


def test_load_detector_refuses_tbd_and_wrong_hash(tmp_path):
    assert load_detectors({"enabled": False}, {"rgb"}) == {}
    with pytest.raises(ValueError, match="enabled must be true or false"):
        load_detectors({"enabled": "TBD"}, {"rgb"})
    base = {"enabled": True, "backend": "yolox", "model": "m", "weights": "x", "weights_sha256": "TBD",
            "imgsz": {"rgb": 1280}, "conf_thresh": 0.3, "nms_iou": 0.65, "classes": ["person"]}
    with pytest.raises(ValueError, match="weights_sha256 is missing or TBD"):
        load_detectors(base, {"rgb"})
    w = tmp_path / "w.pth"
    w.write_bytes(b"abc")
    with pytest.raises(ValueError, match="no input size for \\['ir'\\]"):
        load_detectors({**base, "weights": str(w), "weights_sha256": "0" * 64}, {"rgb", "ir"})
    with pytest.raises(ValueError, match="does not match detection.weights_sha256"):
        load_detectors({**base, "weights": str(w), "weights_sha256": "0" * 64}, {"rgb"})


class FakeDetector:
    """Returns one box (2, 4, 10, 12) in the pixels of the frame it gets, class car, score 0.9; records sizes."""
    def __init__(self):
        self.sizes = []

    def __call__(self, frames):
        self.sizes += [f.shape for f in frames]
        return [Detections(np.array([[2, 4, 10, 12]], np.float32), np.array([0.9], np.float32), np.array([2]))
                for _ in frames]


def test_process_clip_end_to_end(tmp_path):
    # 64x48 video, 45 frames at 30 fps -> samples at frames 0, 6, ..., 42 (8 samples), 2 seconds (1.5 s of video).
    path = tmp_path / "v.avi"
    write_avi(path, n_frames=45, size=(64, 48))
    v = Video(1, "clip", str(path), "G1", "rgb", datetime(2018, 3, 7, 16, 50), 45, 30.0)
    cfg = {"motion": MOTION, "detection": {"enabled": True, "imgsz": {"rgb": 16}, "batch_size": 3}}
    det = FakeDetector()
    marker = process_clip(v, (64, 48), cfg, MotionConfig.from_config(MOTION), det, tmp_path / "art")
    assert marker["n_samples"] == 8 and marker["n_detections"] == 8 and marker["decode_width"] == 32
    # The detector gets frames at imgsz width (16 x 12), MOG2 at 32 wide; decode at the larger (32).
    assert set(det.sizes) == {(12, 16, 3)}
    d = pd.read_parquet(tmp_path / "art" / "detections" / "G1" / "clip.parquet")
    assert d["frame"].tolist() == [0, 6, 12, 18, 24, 30, 36, 42]
    np.testing.assert_allclose(d["t_s"], d["frame"] / 30.0)
    # Box scaled from 16-px-wide detector frame to the 64-px original: x4.
    assert d[["x1", "y1", "x2", "y2"]].iloc[0].tolist() == [8, 16, 40, 48]
    assert d["class"].unique().tolist() == ["car"]
    s = pd.read_parquet(tmp_path / "art" / "motion" / "G1" / "clip.seconds.parquet")
    assert s["second"].tolist() == [0, 1]
    done = json.loads((tmp_path / "art" / "motion" / "G1" / "clip.done.json").read_text())
    assert done["settings"] == settings_hash(cfg)
    assert not list((tmp_path / "art").rglob("*.tmp"))
