from datetime import datetime

import pytest

from rewind.ingest.manifest import Video
import json

import av
import numpy as np

from scripts.throughput import NOTES_PREFIX, decode, history, hms, keep_step, pick_clips
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


def gray_avi(path, n_frames, step=12):
    """Frame i is a flat gray image of level step*i: the content identifies the frame."""
    with av.open(str(path), "w", format="avi") as out:
        stream = out.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.options = {"qscale": "2"}
        for i in range(n_frames):
            img = np.full((48, 64, 3), step * i, np.uint8)
            out.mux(stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
        out.mux(stream.encode())


@pytest.mark.parametrize("thread_type", ["SLICE", "AUTO"])
def test_kept_array_is_that_frame(tmp_path, thread_type):
    path = tmp_path / "g.avi"
    gray_avi(path, 20)
    seen = []
    decode(path, 20, thread_type, step=3, on_keep=lambda i, a: seen.append((i, round(float(a.mean()) / 12))))
    assert seen == [(i, i) for i in range(0, 20, 3)]  # index i carries gray level 12*i (+- half a step)


def test_hms_rounds_to_the_minute():
    assert hms(37.63 * 60) == "0 h 38 min"
    assert hms(3600 + 29) == "1 h 00 min"
    assert hms(2 * 3600 + 59.6 * 60) == "3 h 00 min"


def test_history_reads_old_and_new_metric_names(tmp_path):
    csv_path = tmp_path / "experiments.csv"
    rows = [("t1", "aaaaaaa1", {"decode_rgb_keep_5_fps_x_realtime": 34.0, "decode_ir_keep_5_fps_x_realtime": 97.0,
                                "clip_img_per_s": 299.0}, f"{NOTES_PREFIX}, seed 14"),
            ("t2", "bbbbbbb2", {"decode_rgb_keep_5fps_max640_x_realtime": 20.0,
                                "decode_ir_keep_5fps_max640_x_realtime": 273.0, "clip_img_per_s": 303.0},
             f"{NOTES_PREFIX}, seed 14"),
            ("t3", "ccccccc3", {"clip_img_per_s": 1.0}, "another experiment")]
    with csv_path.open("w") as f:
        f.write("timestamp,git_hash,config_hash,split,metrics,notes\n")
        for t, g, m, n in rows:
            f.write(f"{t},{g},x,none,\"{json.dumps(m).replace(chr(34), chr(34) * 2)}\",\"{n}\"\n")
    keys = ["decode_rgb_keep_5fps_max640_x_realtime", "decode_ir_keep_5fps_max640_x_realtime", "clip_img_per_s"]
    assert history(csv_path, keys) == [
        {"timestamp": "t1", "git": "aaaaaaa", "decode_rgb_keep_5fps_max640_x_realtime": 34.0,
         "decode_ir_keep_5fps_max640_x_realtime": None, "clip_img_per_s": 299.0},  # old IR 5 fps not comparable
        {"timestamp": "t2", "git": "bbbbbbb", "decode_rgb_keep_5fps_max640_x_realtime": 20.0,
         "decode_ir_keep_5fps_max640_x_realtime": 273.0, "clip_img_per_s": 303.0}]


def test_frames_at_yields_exactly_the_wanted_frames(tmp_path):
    from rewind.video import frames_at
    path = tmp_path / "g.avi"
    gray_avi(path, 20)
    got = [(i, round(float(a.mean()) / 12), a.shape) for i, a in frames_at(path, 20, {13, 0, 7})]
    assert got == [(0, 0, (48, 64, 3)), (7, 7, (48, 64, 3)), (13, 13, (48, 64, 3))]   # gray level 12*i -> index i
    with pytest.raises(ValueError, match="outside"):
        list(frames_at(path, 20, {20}))
    with pytest.raises(ValueError, match="decoded 20 frames, manifest says 21"):
        list(frames_at(path, 21, {1}))
