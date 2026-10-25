"""Step 4.1 dedup rules (rewind/embed/dedup.py) on toy frames with hand-computed answers."""

import numpy as np
import pytest

from rewind.embed.dedup import (Candidate, DedupConfig, analysis_gray, boxes_mask, candidate_step, changed_share,
                                detections_unchanged, keep_flags)
from scripts.dedup_sweep import recalled

CFG = DedupConfig(fps=1, max_gap_s=10, pixel_thresh=0.01, pixel_delta=25, det_iou=0.5, det_min_score=0.5,
                  analysis_width=640)


def cand(t, gray, mask=None, dets=()):
    """Candidate at t seconds (30 fps video); dets = [(class, (x1, y1, x2, y2), score)]."""
    gray = np.asarray(gray, dtype=np.uint8)
    return Candidate(frame=30 * t, t_s=float(t), gray=gray,
                     motion_mask=np.ones(gray.shape, bool) if mask is None else mask,
                     det_classes=np.array([d[0] for d in dets], dtype=str),
                     det_boxes=np.array([d[1] for d in dets], dtype=float).reshape(-1, 4),
                     det_scores=np.array([d[2] for d in dets], dtype=float))


def test_boxes_mask_scales_and_clips():
    # 1920x1080 -> 640x360 is /3: box (0, 0, 960, 540) -> columns 0..320, rows 0..180 = 57,600 pixels
    m = boxes_mask(np.array([[0, 0, 960, 540]]), (1920, 1080), (640, 360))
    assert m.sum() == 320 * 180 and m[179, 319] and not m[180, 0] and not m[0, 320]
    # a box hanging past the frame is clipped: (1800, 1000, 2400, 1500) -> cols 600..640, rows 333.3..360 -> 333..360
    m = boxes_mask(np.array([[1800, 1000, 2400, 1500]]), (1920, 1080), (640, 360))
    assert m.sum() == 40 * 27
    assert boxes_mask(np.zeros((0, 4)), (1920, 1080), (640, 360)).sum() == 0


def test_changed_share_counts_only_inside_motion_regions_and_above_delta():
    a = np.zeros((10, 10), np.uint8)
    b = a.copy()
    b[0, 0:3] = 30          # 3 pixels changed by 30 > 25, inside the region
    b[9, 9] = 200           # changed, but outside both regions
    b[1, 0] = 25            # changed by exactly 25: not > delta
    region = np.zeros((10, 10), bool)
    region[0:5, 0:5] = True
    assert changed_share(cand(0, a, region), cand(1, b, np.zeros((10, 10), bool)), 25) == pytest.approx(3 / 100)
    # the union of both frames' regions counts: region only on the second frame gives the same answer
    assert changed_share(cand(0, a, np.zeros((10, 10), bool)), cand(1, b, region), 25) == pytest.approx(3 / 100)
    # no motion region on either frame: no change, whatever the pixels do
    none = np.zeros((10, 10), bool)
    assert changed_share(cand(0, a, none), cand(1, b, none), 25) == 0.0


def test_detections_unchanged():
    car = ("car", (0, 0, 10, 10), 0.9)
    car_moved = ("car", (2, 0, 12, 10), 0.9)      # IoU = 80 / 120 = 0.667 >= 0.5
    car_far = ("car", (6, 0, 16, 10), 0.9)        # IoU = 40 / 160 = 0.25 < 0.5
    g = np.zeros((2, 2))
    assert detections_unchanged(cand(0, g, dets=[car]), cand(1, g, dets=[car_moved]), 0.5, 0.5)
    assert not detections_unchanged(cand(0, g, dets=[car]), cand(1, g, dets=[car_far]), 0.5, 0.5)
    # same box, other class: changed
    assert not detections_unchanged(cand(0, g, dets=[car]), cand(1, g, dets=[("truck", car[1], 0.9)]), 0.5, 0.5)
    # a low-score box that appears (flicker) is ignored; a confident one that appears is a change
    weak = ("person", (50, 50, 60, 70), 0.35)
    assert detections_unchanged(cand(0, g, dets=[car]), cand(1, g, dets=[car, weak]), 0.5, 0.5)
    assert not detections_unchanged(cand(0, g, dets=[car]), cand(1, g, dets=[car, ("person", weak[1], 0.8)]), 0.5, 0.5)
    # a confident box matched by a weak box of the same class still counts as present
    assert detections_unchanged(cand(0, g, dets=[car]), cand(1, g, dets=[("car", car[1], 0.31)]), 0.5, 0.5)
    assert detections_unchanged(cand(0, g), cand(1, g), 0.5, 0.5)


def test_keep_flags_static_scene_keeps_only_every_max_gap():
    g = np.zeros((10, 10))
    cands = [cand(t, g) for t in range(23)]
    flags = keep_flags(cands, CFG, use_detections=True)
    assert [t for t, f in enumerate(flags) if f] == [0, 10, 20]


def test_keep_flags_compares_with_last_kept_not_previous():
    # grey level rises 10 per second: vs the previous frame always 10 (<= 25), vs the last kept frame it passes 25
    # after 3 s, so frames 0, 3, 6, 9, ... are kept (100 % of pixels changed >= thresh 0.01)
    cands = [cand(t, np.full((10, 10), 10 * t)) for t in range(10)]
    assert [t for t, f in enumerate(keep_flags(cands, CFG, True)) if f] == [0, 3, 6, 9]


def test_keep_flags_threshold_zero_keeps_all_and_detections_force_keeps():
    g = np.zeros((10, 10))
    cands = [cand(t, g) for t in range(5)]
    assert all(keep_flags(cands, DedupConfig(**{**CFG.__dict__, "pixel_thresh": 0.0}), True))
    # a confident person appears at t=2 and stays: kept at 2 (changed vs t=0), dropped at 3, 4 (same as t=2)
    p = ("person", (1, 1, 5, 9), 0.8)
    cands = [cand(0, g), cand(1, g), cand(2, g, dets=[p]), cand(3, g, dets=[p]), cand(4, g, dets=[p])]
    assert keep_flags(cands, CFG, use_detections=True) == [True, False, True, False, False]
    assert keep_flags(cands, CFG, use_detections=False) == [True, False, False, False, False]


def test_keep_flags_refuses_unordered():
    g = np.zeros((2, 2))
    with pytest.raises(ValueError, match="increasing"):
        keep_flags([cand(1, g), cand(0, g)], CFG, True)


def test_candidate_step_and_config():
    assert candidate_step(30, 1, 5) == 30
    with pytest.raises(ValueError, match="grid"):
        candidate_step(30, 2, 5)      # step 15 is not a multiple of the 5 fps step 6
    cfg = {"sampling": {"appearance_fps": 1, "max_gap_s": 10, "dedup_motion_pixel_thresh": "TBD",
                        "dedup_pixel_delta": 25, "dedup_det_iou": 0.5, "dedup_det_min_score": 0.5},
           "motion": {"downscale_width": 640}}
    with pytest.raises(ValueError, match="TBD"):
        DedupConfig.from_config(cfg)
    assert DedupConfig.from_config(cfg, pixel_thresh=0.001).pixel_thresh == 0.001
    with pytest.raises(ValueError, match="share"):
        DedupConfig.from_config(cfg, pixel_thresh=3)


def test_analysis_gray_scales_down_only():
    big = np.zeros((1080, 1920, 3), np.uint8)
    assert analysis_gray(big, 640).shape == (360, 640)
    small = np.zeros((240, 352, 3), np.uint8)
    assert analysis_gray(small, 640).shape == (240, 352)


def test_recalled_pads_both_sides_inclusive():
    kept = np.array([0, 300, 600])
    assert recalled(kept, 330, 400, 0) is False          # no kept frame in 330..400
    assert recalled(kept, 330, 400, 30) is True          # 300 >= 330 - 30
    assert recalled(kept, 540, 570, 30) is True          # 600 <= 570 + 30
    assert recalled(kept, 540, 569, 30) is False         # 600 > 599
    assert recalled(kept, 601, 700, 0) is False
