"""Step 4.2 crop selection (rewind/embed/crops.py) on toy boxes with hand-computed answers."""

import numpy as np
import pytest

from rewind.embed.crops import CropConfig, covered_share, select_crops, variant

# 100x100 frame, masks at full resolution (mask_width >= frame width)
CFG = CropConfig(max_crops=2, min_crop_px=10, rank="score_area", moving_share=0.5, dup_cover=0.5, mask_width=640)
SIZE = (100, 100)


def test_covered_share_hand_computed():
    mask = np.zeros((100, 100), bool)
    mask[0:50, 0:100] = True                     # top half
    boxes = np.array([[0, 0, 10, 10], [0, 40, 10, 60], [0, 60, 10, 80], [-5, -5, 10, 10]])
    # inside / half inside / outside / clipped to (0, 0, 10, 10) -> inside
    assert covered_share(boxes, mask, SIZE).tolist() == [1.0, 0.5, 0.0, 1.0]


def test_score_area_ranks_by_score_times_area_and_skips_small_boxes():
    cls = np.array(["car", "car", "person", "car"])
    boxes = np.array([[0, 0, 40, 40],        # area 1600 x 0.5 = 800
                      [50, 50, 70, 70],      # 400 x 0.9 = 360
                      [80, 0, 95, 30],       # 450 x 0.9 = 405
                      [0, 90, 5, 100]])      # shorter side 5 < 10: skipped
    scores = np.array([0.5, 0.9, 0.9, 0.99])
    crops = select_crops(cls, boxes, scores, np.zeros((0, 4)), SIZE, CFG)
    det = [c for c in crops if c.source == "detection"]
    assert [(c.cls, c.box) for c in det] == [("car", (0, 0, 40, 40)), ("person", (80, 0, 95, 30))]
    assert [c.rank for c in det] == [0, 1]


def test_moving_first_puts_detections_under_motion_first():
    cls = np.array(["car", "person"])
    boxes = np.array([[0, 0, 60, 60], [70, 70, 90, 90]])   # big parked car, small walking person
    scores = np.array([0.9, 0.6])
    motion = np.array([[68, 68, 92, 92]])                     # covers the person fully
    one = CropConfig(**{**CFG.__dict__, "max_crops": 1})
    assert select_crops(cls, boxes, scores, motion, SIZE, one)[0].cls == "car"
    moving = CropConfig(**{**one.__dict__, "rank": "moving_first"})
    assert select_crops(cls, boxes, scores, motion, SIZE, moving)[0].cls == "person"


def test_motion_crops_dup_flag_and_variants():
    cls = np.array(["person"])
    boxes = np.array([[10, 10, 30, 50]])
    scores = np.array([0.8])
    motion = np.array([[10, 10, 30, 50],      # area 800, the same object: dup
                       [60, 60, 80, 80],      # area 400
                       [0, 70, 15, 85],       # area 225
                       [90, 90, 95, 95]])     # too small
    crops = select_crops(cls, boxes, scores, motion, SIZE, CFG)
    mot = [c for c in crops if c.source == "motion"]
    assert [(c.box, c.dup) for c in mot] == [((10, 10, 30, 50), True), ((60, 60, 80, 80), False)]   # top 2 by area
    assert [c.box for c in variant(crops, "motion", 2)] == [(10, 10, 30, 50), (60, 60, 80, 80)]
    assert [(c.source, c.box) for c in variant(crops, "detection_then_motion", 2)] == [
        ("detection", (10, 10, 30, 50)), ("motion", (60, 60, 80, 80))]
    assert variant(crops, "none", 2) == []
    with pytest.raises(ValueError, match="crop_source"):
        variant(crops, "tiles", 2)


def test_crop_config_refuses_tbd_rank():
    cfg = {"appearance": {"max_crops_per_frame": 4, "min_crop_px": 32, "crop_rank": "TBD", "crop_moving_share": 0.5,
                          "crop_dup_cover": 0.5}, "motion": {"downscale_width": 640}}
    with pytest.raises(ValueError, match="TBD"):
        CropConfig.from_config(cfg)
    assert CropConfig.from_config(cfg, rank="moving_first").rank == "moving_first"
