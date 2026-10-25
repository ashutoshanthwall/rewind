"""Step 4.2 frame views and crop squares (rewind/embed/clip_model.py), on CPU tensors with hand-computed answers."""

import numpy as np
import pytest
import torch

from rewind.embed.clip_model import crop_square, frame_tensor, frame_view, square_region, view_box


def ramp(h, w):
    """(3, h, w) image whose pixel value encodes its column (channel 0) and row (channel 1)."""
    img = torch.zeros((3, h, w))
    img[0] = torch.arange(w).float()[None, :].expand(h, w)
    img[1] = torch.arange(h).float()[:, None].expand(h, w)
    return img


def test_view_boxes_rgb_and_ir():
    assert view_box("tile0", 1920, 1080) == (0, 0, 1080, 1080)
    assert view_box("tile1", 1920, 1080) == (840, 0, 1920, 1080)   # PROGRESS item 4: x 0-1080 and 840-1920
    assert view_box("tile1", 352, 240) == (112, 0, 352, 240)
    assert view_box("pad", 352, 240) == (0, 0, 352, 240)
    with pytest.raises(ValueError, match="landscape"):
        view_box("tile0", 240, 352)


def test_frame_views_are_the_right_pixels():
    img = ramp(4, 6)
    t0, t1 = frame_view(img, "tile0"), frame_view(img, "tile1")
    assert t0.shape == t1.shape == (3, 4, 4)
    assert t0[0, 0].tolist() == [0, 1, 2, 3] and t1[0, 0].tolist() == [2, 3, 4, 5]
    pad = frame_view(img, "pad")                    # 6x6, the 4 image rows in the middle (1 black row above/below)
    assert pad.shape == (3, 6, 6)
    assert pad[1, :, 0].tolist() == [0, 0, 1, 2, 3, 0] and pad[0, 0].abs().sum() == 0 and pad[0, 5].abs().sum() == 0
    assert pad[0, 1].tolist() == [0, 1, 2, 3, 4, 5]


def test_crop_square_pads_and_squares_around_centre():
    # box 100x40 at (10, 20): side = 100 * (1 + 2 * 0.2) = 140, centre (60, 40) -> x1 = -10, y1 = -30
    assert crop_square((10, 20, 110, 60), 0.2) == (-10, -30, 140)
    assert crop_square((0, 0, 10, 30), 0.0) == (-10, 0, 30)


def test_square_region_fills_outside_with_black():
    img = ramp(4, 6)
    sq = square_region(img, -1, -1, 3)              # rows/cols -1..1: first row and column outside the image
    assert sq[0].tolist() == [[0, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert sq[1].tolist() == [[0, 0, 0], [0, 0, 0], [0, 1, 1]]
    assert square_region(img, 10, 10, 2).abs().sum() == 0   # fully outside


def test_frame_tensor_converts_bgr_to_rgb():
    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[..., 0] = 10   # blue
    bgr[..., 2] = 200  # red
    t = frame_tensor(bgr, device="cpu")
    assert t.shape == (3, 2, 2) and t[0, 0, 0] == 200 and t[2, 0, 0] == 10
