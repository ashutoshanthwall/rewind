"""Step 3.1: MOG2 scoring, per-second aggregation, frame sampling and the check's measures (hand-computed cases)."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from rewind.ingest.manifest import Video
from rewind.motion.mog2 import FrameMotion, MotionConfig, MotionScorer, per_second, score_clip
from rewind.video import sampled_frames, scaled_size
from scripts.check_motion import auc, coverage, event_seconds, lanes, pick_clips
from tests.test_manifest import write_avi

CFG = MotionConfig(fps=5, downscale_width=640, history_frames=500, var_threshold=16, detect_shadows=True,
                   warmup_s=1, open_kernel_px=3, merge_dilate_px=9, min_box_area_px=50)


def scene(n: int, square_from: int, size=(160, 120), square=20) -> list[tuple[int, np.ndarray]]:
    """Grey static background with mild noise; from sample `square_from` on, a white square moves right 4 px per
    sample starting at x=20, y=50."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        img = np.clip(100 + rng.normal(0, 2, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
        if i >= square_from:
            x = 20 + 4 * (i - square_from)
            img[50:50 + square, x:x + square] = 255
        out.append((6 * i, img))   # frame indices on a 6-frame grid (5 fps sampling of 30 fps video)
    return out


def test_static_scene_has_no_activity_and_no_boxes():
    frames = score_clip(scene(30, square_from=10**9), CFG, (160, 120))
    assert len(frames) == 30
    assert max(f.activity for f in frames) < 0.001   # warm-up: no first-frame burst
    assert all(f.boxes == [] for f in frames)


def test_moving_square_is_found_and_boxed_in_original_pixels():
    # Analysis size = original size here (160 < 640), so boxes are in the same pixels.
    frames = score_clip(scene(30, square_from=15), CFG, (160, 120))
    before, after = frames[:15], frames[16:]
    assert max(f.activity for f in before) < 0.001
    # The square is 400 px of 19,200 = 0.021 of the frame; MOG2 sees it (and some of where it just was).
    assert min(f.activity for f in after) > 0.01
    last = frames[-1]                       # sample 29: square at x = 20 + 4 * 14 = 76 .. 96, y = 50 .. 70
    assert len(last.boxes) == 1
    x1, y1, x2, y2 = last.boxes[0]
    assert x1 <= 76 and x2 >= 96 and y1 <= 50 and y2 >= 70
    assert x2 - x1 < 60 and y2 - y1 < 40     # one tight-ish box, not the whole frame


def test_boxes_are_scaled_back_to_original_size():
    scorer = MotionScorer(CFG, orig_size=(1920, 1080))
    fg = np.zeros((360, 640), np.uint8)
    fg[100:120, 200:240] = 1                 # 40 x 20 analysis px; dilation 9 grows it by 4 px per side
    (x1, y1, x2, y2), = scorer.boxes(fg)
    assert (x1, y1, x2, y2) == (196 * 3, 96 * 3, 244 * 3, 124 * 3)


def test_small_blobs_are_dropped():
    # One foreground pixel dilated by the 9x9 ellipse covers exactly the kernel's pixels (< 81); a 10x10 square
    # grows to more than 100. With min_box_area_px 100 only the square is kept.
    cfg = MotionConfig(**{**CFG.__dict__, "min_box_area_px": 100})
    scorer = MotionScorer(cfg, orig_size=(160, 120))
    assert int(scorer.dilate_kernel.sum()) < 100
    fg = np.zeros((120, 160), np.uint8)
    fg[10, 10] = 1
    assert scorer.boxes(fg) == []
    fg[60:70, 60:70] = 1
    assert scorer.boxes(fg) == [(56, 56, 74, 74)]           # square 60..69 grown by 4 px per side, x2/y2 exclusive


def test_per_second_mean_and_short_last_second():
    # 30 fps, 5 fps samples at frames 0, 6, ..., 66 for a 70-frame clip: seconds 0 (frames 0-29), 1 (30-59), 2 (60-69)
    frames = [FrameMotion(f, a, []) for f, a in [(0, 0.1), (6, 0.1), (12, 0.1), (18, 0.1), (24, 0.1),
                                                  (30, 0.2), (36, 0.4), (42, 0.0), (48, 0.0), (54, 0.4),
                                                  (60, 0.9), (66, 0.3)]]
    np.testing.assert_allclose(per_second(frames, 30.0, 70), [0.1, 0.2, 0.6])


def test_per_second_refuses_a_second_without_samples():
    with pytest.raises(ValueError, match="seconds without any motion sample"):
        per_second([FrameMotion(0, 0.1, [])], 30.0, 60)


def test_config_refuses_tbd_and_missing():
    good = {k: getattr(CFG, k) for k in CFG.__dataclass_fields__}
    assert MotionConfig.from_config(good) == CFG
    with pytest.raises(ValueError, match="min_box_area_px"):
        MotionConfig.from_config({**good, "min_box_area_px": "TBD"})
    del good["warmup_s"]
    with pytest.raises(ValueError, match="warmup_s"):
        MotionConfig.from_config(good)


def test_sampled_frames_grid_and_scaling(tmp_path):
    path = tmp_path / "v.avi"
    write_avi(path, n_frames=20, size=(64, 48))
    out = list(sampled_frames(path, 20, step=6, max_width=32))
    assert [i for i, _ in out] == [0, 6, 12, 18]
    assert out[0][1].shape == (24, 32, 3)
    # write_avi paints frame i with grey level i * 8: the sampled frames are the right ones (mpeg4 is lossy: +-3)
    assert all(abs(a.mean() - 8 * i) < 4 for i, a in out)


def test_sampled_frames_checks_frame_count(tmp_path):
    path = tmp_path / "v.avi"
    write_avi(path, n_frames=10)
    with pytest.raises(ValueError, match="decoded 10 frames, manifest says 11"):
        list(sampled_frames(path, 11, step=5, max_width=640))


def test_scaled_size():
    assert scaled_size(1920, 1080, 640) == (640, 360)
    assert scaled_size(352, 240, 640) == (352, 240)     # never up
    assert scaled_size(1000, 333, 500) == (500, 166)    # 166.5 -> even


def test_auc_hand_computed():
    # pos 0.5 beats 0.1, 0.2 and ties 0.5 -> 2.5 of 3; pos 0.1 ties 0.1 -> 0.5 of 3. (2.5 + 0.5) / 6 = 0.5
    assert auc([0.5, 0.1], [0.1, 0.2, 0.5]) == pytest.approx(0.5)
    assert auc([1, 2], [0]) == 1.0 and auc([0], [1, 2]) == 0.0


def test_coverage_hand_computed():
    actor = (0, 0, 10, 10)                                    # area 100
    assert coverage(actor, []) == 0.0
    assert coverage(actor, [(5, 0, 20, 10)]) == 0.5           # 5 x 10 overlap
    assert coverage(actor, [(5, 0, 20, 10), (0, 0, 8, 10)]) == 0.8   # best single box, not the union
    assert coverage((3, 3, 3, 9), [(0, 0, 10, 10)]) == 0.0    # zero-area box


def test_event_seconds_and_lanes():
    assert list(event_seconds(1.5, 3.0, 10)) == [1, 2]        # [1.5, 3.0) overlaps seconds 1 and 2
    assert list(event_seconds(1.5, 3.2, 10)) == [1, 2, 3]
    assert list(event_seconds(8.5, 12.0, 10)) == [8, 9]       # clipped to the clip
    assert lanes([(0, 5), (1, 3), (5, 8), (3, 4)]) == [0, 1, 0, 1]


def test_pick_clips_needs_back_to_back():
    t = datetime(2018, 3, 7, 16, 50)
    v = lambda i, start: Video(i, f"c{i}", "p", "G1", "rgb", start, 9000, 30.0)
    clips = [v(1, t), v(2, t + timedelta(minutes=10)), v(3, t + timedelta(minutes=15))]
    assert [c.video_id for c in pick_clips(clips, "G1", 10, 2.0)] == [2, 3]   # 1 is not followed back to back
    with pytest.raises(ValueError, match="no back-to-back run"):
        pick_clips(clips, "G1", 15, 2.0)
