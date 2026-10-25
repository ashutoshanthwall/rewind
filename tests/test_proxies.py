import subprocess

import av
import numpy as np
import pytest

from rewind.ingest.manifest import connect
from scripts.make_proxies import (alignment_corr, check_alignment, check_recorded, ffmpeg_command, make_one,
                                  probe_proxy, record, verify_proxy)

CFG = {"max_height": 720, "codec": "libx264", "crf": 23, "preset": "veryfast", "keyint_frames": 60,
       "max_duration_diff_s": 1.0, "align_check_frames": 60, "align_check_width": 32, "align_max_offset": 3}


# --- alignment metric -----------------------------------------------------------------------------


def test_identical_sequences_correlate_perfectly_at_shift_zero():
    rng = np.random.default_rng(0)
    frames = rng.uniform(0, 255, (20, 4, 4)).astype(np.float32)
    corr = alignment_corr(frames, frames, max_offset=2)
    assert corr[0] == pytest.approx(1.0)  # each difference image with itself: cosine 1
    assert all(v < 0.5 for d, v in corr.items() if d != 0)  # independent random differences


def test_hand_computed_two_pixel_case():
    # 1x2-pixel frames. A mean-centred 2-pixel difference image is +-(a, -a), so the correlation of two of them
    # is the product of the signs of (left - right). Frame values chosen so the differences have signs
    # s = [+, +, -, +, -, -, +, -] (8 difference images from 9 frames).
    s = [1, 1, -1, 1, -1, -1, 1, -1]
    frames = np.zeros((9, 1, 2), np.float32)
    for k, sign in enumerate(s):
        frames[k + 1] = frames[k] + np.array([[sign, -sign]], np.float32)
    corr = alignment_corr(frames, frames, max_offset=1)
    # i runs over 1..6 for every shift.
    # d=0:  all products +1 -> 1.0
    # d=+1: s1*s2.. s6*s7 = (+)(-)=-1, (-)(+)=-1, (+)(-)=-1, (-)(-)=+1, (-)(+)=-1, (+)(-)=-1 -> -4/6
    # d=-1: s1*s0.. s6*s5 = (+)(+)=+1, (-)(+)=-1, (+)(-)=-1, (-)(+)=-1, (-)(-)=+1, (+)(-)=-1 -> -2/6
    assert corr == pytest.approx({-1: -2 / 6, 0: 1.0, 1: -4 / 6})


def test_static_frames_contribute_zero():
    frames = np.ones((10, 3, 3), np.float32)
    assert alignment_corr(frames, frames, max_offset=1) == {-1: 0.0, 0: 0.0, 1: 0.0}
    with pytest.raises(ValueError, match="not aligned"):  # no motion: alignment cannot be shown
        check_alignment(alignment_corr(frames, frames, max_offset=1), "static")


def test_shifted_proxy_is_rejected():
    rng = np.random.default_rng(1)
    frames = rng.uniform(0, 255, (30, 4, 4)).astype(np.float32)
    shifted = np.concatenate([frames[1:], frames[-1:]])  # proxy lost frame 0, last frame duplicated
    corr = alignment_corr(frames, shifted, max_offset=2)
    assert max(corr, key=corr.get) == -1
    with pytest.raises(ValueError, match="not aligned"):
        check_alignment(corr, "shifted")


def test_too_few_frames_raises():
    with pytest.raises(ValueError, match="Need more than"):
        alignment_corr(np.zeros((5, 2, 2)), np.zeros((5, 2, 2)), max_offset=3)


# --- real ffmpeg on generated videos --------------------------------------------------------------


def moving_square_avi(path, n_frames=60, size=(64, 48), fps=30):
    """mpeg4 AVI with a bright square moving 1 px per frame, so every frame differs from its neighbours."""
    with av.open(str(path), "w", format="avi") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        stream.options = {"qscale": "2"}
        for i in range(n_frames):
            img = np.full((size[1], size[0], 3), 40, np.uint8)
            x = i % (size[0] - 12)
            img[10:22, x:x + 12] = 230
            out.mux(stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
        out.mux(stream.encode())


def video_row(tmp_path, n_frames=60):
    return {"video_id": 1, "clip": "2018-03-07.10-00-00.10-00-02.school.G336", "path": "raw/toy.avi",
            "n_frames": n_frames, "fps": 30.0, "duration_s": n_frames / 30, "width": 64, "height": 48}


def test_make_one_keeps_every_frame_and_alignment(tmp_path):
    (tmp_path / "raw").mkdir()
    moving_square_avi(tmp_path / "raw" / "toy.avi")
    proxy_dir = tmp_path / "proxy"
    proxy_dir.mkdir()
    final, info, corr = make_one(video_row(tmp_path), tmp_path, proxy_dir, CFG)
    assert final.name == "2018-03-07.10-00-00.10-00-02.school.G336.mp4"
    assert not final.with_name(final.name + ".part").exists()
    assert (info["n_frames"], info["fps"], info["width"], info["height"]) == (60, 30.0, 64, 48)  # no upscaling
    assert max(corr, key=corr.get) == 0
    conn = connect(tmp_path / "m.sqlite")
    conn.execute("INSERT INTO videos VALUES (1, 'c', 'raw/toy.avi', 1, 'x', 'G336', 'school', 'rgb', "
                 "'2018-03-07T10:00:00', '2018-03-07T10:00:02', 2.0, 60, 30.0, 64, 48, 'mpeg4', 'now')")
    record(conn, video_row(tmp_path), final, tmp_path, info, corr, CFG)
    assert check_recorded(conn, tmp_path) == 1
    final.write_bytes(b"truncated")
    with pytest.raises(FileNotFoundError, match="missing or changed size"):
        check_recorded(conn, tmp_path)


def test_scaling_to_720_keeps_aspect(tmp_path):
    src, dst = tmp_path / "big.avi", tmp_path / "big.mp4"
    moving_square_avi(src, n_frames=10, size=(1920, 1080))
    subprocess.run(ffmpeg_command(src, dst, 30.0, CFG), check=True)
    info = probe_proxy(dst, 30.0)
    assert (info["width"], info["height"], info["n_frames"]) == (1280, 720, 10)


def test_frame_dropping_proxy_is_rejected(tmp_path):
    # A proxy that lost frame 0 but kept the frame count (last frame cloned) must fail the alignment check.
    (tmp_path / "raw").mkdir()
    src = tmp_path / "raw" / "toy.avi"
    moving_square_avi(src)
    proxy_dir = tmp_path / "proxy"
    proxy_dir.mkdir()
    bad = proxy_dir / "bad.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf",
                    "trim=start_frame=1,setpts=N/(30*TB),tpad=stop=1:stop_mode=clone", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(bad)], check=True)
    assert probe_proxy(bad, 30.0)["n_frames"] == 60
    with pytest.raises(ValueError, match="not aligned"):
        verify_proxy(video_row(tmp_path), bad, src, CFG)


def test_unrecorded_finished_proxy_is_not_adopted(tmp_path):
    (tmp_path / "raw").mkdir()
    moving_square_avi(tmp_path / "raw" / "toy.avi")
    proxy_dir = tmp_path / "proxy"
    proxy_dir.mkdir()
    (proxy_dir / "2018-03-07.10-00-00.10-00-02.school.G336.mp4").write_bytes(b"left over")
    with pytest.raises(FileExistsError, match="not recorded"):
        make_one(video_row(tmp_path), tmp_path, proxy_dir, CFG)


def test_wrong_frame_count_is_rejected(tmp_path):
    (tmp_path / "raw").mkdir()
    moving_square_avi(tmp_path / "raw" / "toy.avi")
    proxy_dir = tmp_path / "proxy"
    proxy_dir.mkdir()
    with pytest.raises(ValueError, match="original has 61"):
        make_one(video_row(tmp_path, n_frames=61), tmp_path, proxy_dir, CFG)


def test_timestamp_gap_is_rejected(tmp_path):
    # Same frame count, but frames 30.. are shown 1/30 s late: playback time would no longer be index / fps.
    src, bad = tmp_path / "toy.avi", tmp_path / "gap.mp4"
    moving_square_avi(src)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", r"setpts=(N+gte(N\,30))/(30*TB)",
                    "-fps_mode", "passthrough", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(bad)], check=True)
    with pytest.raises(ValueError, match=r"frame 30 has pts 1\.033333 s, expected 1\.000000 s"):
        probe_proxy(bad, 30.0)


def test_proxy_timestamps_are_index_over_fps(tmp_path):
    src, dst = tmp_path / "toy.avi", tmp_path / "toy.mp4"
    moving_square_avi(src)
    subprocess.run(ffmpeg_command(src, dst, 30.0, CFG), check=True)
    with av.open(str(dst)) as c:
        times = [f.time for f in c.decode(video=0)]
    assert times == pytest.approx([i / 30 for i in range(60)], abs=1e-9)  # decoded in display order 0, 1/30, ...


def test_non_integer_fps_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not an integer"):
        ffmpeg_command(tmp_path / "a.avi", tmp_path / "a.mp4", 29.97, CFG)
