from datetime import datetime

import pytest

from rewind.ingest.manifest import Video
from scripts.throughput import decode, keep_step, pick_clips
from tests.test_manifest import write_avi


def test_keep_step():
    assert keep_step(30.0, 1) == 30
    assert keep_step(30.0, 5) == 6
    with pytest.raises(ValueError, match="not a whole-frame step"):
        keep_step(30.0, 7)  # 30/7 frames
    with pytest.raises(ValueError, match="not a whole-frame step"):
        keep_step(30.0, 60)


@pytest.mark.parametrize("thread_type", ["SLICE", "AUTO"])
def test_decode_keeps_every_kth_frame_from_zero(tmp_path, thread_type):
    path = tmp_path / "v.avi"
    write_avi(path, n_frames=20, size=(64, 48))
    seen = []
    kept, seconds = decode(path, 20, thread_type, step=6, width=32, on_keep=lambda i, a: seen.append((i, a.shape)))
    assert kept == 4 and seconds > 0
    assert seen == [(0, (24, 32, 3)), (6, (24, 32, 3)), (12, (24, 32, 3)), (18, (24, 32, 3))]  # 64x48 -> 32x24
    assert decode(path, 20, thread_type)[0] == 0  # decode only: nothing kept
    shapes = []
    decode(path, 20, thread_type, step=10, width=128, on_keep=lambda i, a: shapes.append(a.shape))
    assert shapes == [(48, 64, 3)] * 2  # never scaled up


def test_decode_refuses_wrong_frame_count(tmp_path):
    path = tmp_path / "v.avi"
    write_avi(path, n_frames=10)
    with pytest.raises(ValueError, match="decoded 10 frames, manifest says 11"):
        decode(path, 11, "AUTO")


def video(clip, camera, modality="rgb"):
    return Video(1, clip, f"raw/{clip}", camera, modality, datetime(2018, 3, 7), 9000, 30.0)


def test_pick_clips_round_robin_over_cameras():
    vids = [video(f"a{i}", "G330") for i in range(5)] + [video(f"b{i}", "G336") for i in range(5)] + \
           [video("c0", "G340")] + [video(f"ir{i}", "G474", "ir") for i in range(3)]
    picked = pick_clips(vids, "rgb", 5, seed=1)
    assert [v.camera_id for v in picked] == ["G330", "G336", "G340", "G330", "G336"]  # G340 has only 1 clip
    assert len({v.clip for v in picked}) == 5
    assert pick_clips(vids, "rgb", 5, seed=1) == picked  # seeded
    assert {v.modality for v in pick_clips(vids, "ir", 3, seed=1)} == {"ir"}
    with pytest.raises(ValueError, match="Only 3 ir clips available"):
        pick_clips(vids, "ir", 4, seed=1)
