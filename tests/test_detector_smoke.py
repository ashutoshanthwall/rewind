"""Step 3.3: box matching and helpers of the detector smoke test (hand-computed cases; no detector needed)."""

import numpy as np
import pytest

from rewind.detect.detectors import class_ids
from scripts.detector_smoke import centre_match, dedup_tagged, greedy_match, iou_matrix, parse_spec, size_bucket


def test_iou_matrix_hand_computed():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[5, 0, 15, 10], [0, 0, 10, 10], [20, 20, 30, 30]], float)
    # overlap 50, union 150 -> 1/3; identical -> 1; disjoint -> 0
    np.testing.assert_allclose(iou_matrix(a, b), [[1 / 3, 1.0, 0.0]])
    assert iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)


def test_greedy_match_is_one_to_one_and_prefers_best_iou():
    gt = np.array([[0, 0, 10, 10], [1, 0, 11, 10]], float)
    det = np.array([[0, 0, 10, 10]], float)            # IoU 1.0 with gt0, 9/11 with gt1
    assert greedy_match(gt, det, 0.5).tolist() == [True, False]
    det2 = np.array([[0, 0, 10, 10], [1, 0, 11, 10]], float)
    assert greedy_match(gt, det2, 0.5).tolist() == [True, True]
    assert greedy_match(gt, det, 1.01).tolist() == [False, False]
    assert greedy_match(gt, np.zeros((0, 4)), 0.5).tolist() == [False, False]


def test_greedy_match_threshold_is_inclusive():
    gt = np.array([[0, 0, 10, 10]], float)
    det = np.array([[0, 0, 10, 20]], float)            # IoU 100 / 200 = 0.5 exactly
    assert greedy_match(gt, det, 0.5).tolist() == [True]


def test_centre_match_loose_box_and_one_to_one():
    loose = np.array([[0, 0, 100, 50]], float)
    tight = np.array([[40, 20, 50, 30]], float)        # IoU 0.02, centre (45, 25) inside
    assert greedy_match(loose, tight, 0.3).tolist() == [False]
    assert centre_match(loose, tight).tolist() == [True]
    # two GT boxes, one detection inside both: only one GT is found (the smaller first)
    gts = np.array([[0, 0, 100, 50], [40, 20, 60, 40]], float)
    assert centre_match(gts, tight).tolist() == [False, True]
    assert centre_match(gts, np.zeros((0, 4))).tolist() == [False, False]


def test_dedup_tagged_clips_drops_empty_and_duplicates():
    items = [((-10, 0, 10, 10), "A"), ((0, 0, 10, 10), "B"), ((50, 50, 60, 60), "C"), ((200, 5, 220, 9), "D")]
    boxes, tags = dedup_tagged(items, w=100, h=100, dup_iou=0.9)
    # A clipped to (0,0,10,10) == B -> B dropped as duplicate; D lies outside the 100 px frame -> empty, dropped
    assert boxes.tolist() == [[0, 0, 10, 10], [50, 50, 60, 60]]
    assert tags == ["A", "C"]


def test_size_bucket():
    edges = [0.05, 0.15]
    assert size_bucket(np.array([0, 0, 5, 4]), 100, edges) == "<0.05"
    assert size_bucket(np.array([0, 0, 5, 5]), 100, edges) == "0.05-0.15"
    assert size_bucket(np.array([0, 0, 5, 15]), 100, edges) == ">=0.15"


def test_parse_spec():
    assert parse_spec("a=yolox:w/x.pth:640:yolox-s") == {"name": "a", "backend": "yolox", "weights": "w/x.pth",
                                                         "imgsz": 640, "yolox_exp": "yolox-s"}
    assert parse_spec("b=ultralytics:w/y.pt:1280")["yolox_exp"] is None
    with pytest.raises(ValueError, match="bad detector spec"):
        parse_spec("c=ultralytics:w/y.pt")


def test_class_ids():
    assert class_ids(["person", "car", "truck", "bus", "motorcycle", "bicycle"]) == [0, 2, 7, 5, 3, 1]
    with pytest.raises(ValueError, match="not COCO class names"):
        class_ids(["van"])
