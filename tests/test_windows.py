"""Action windows (rewind/embed/windows.py). Hand-computed toy cases."""

import numpy as np
import pytest

from rewind.embed.windows import pooled_windows, window_frames, window_starts


def test_window_starts_whole_windows_only():
    # 10 s clip, 4 s windows, stride 2: starts 0, 2, 4, 6 (6 + 4 = 10 fits); 8 would end at 12
    assert window_starts(10.0, 4, 2).tolist() == [0, 2, 4, 6]
    # 300.33 s (a real clip length): last start 296 (296 + 4 = 300 <= 300.33), 149 windows
    s = window_starts(9010 / 30, 4, 2)
    assert len(s) == 149 and s[-1] == 296
    assert window_starts(3.9, 4, 2).tolist() == []          # shorter than one window: none
    assert window_starts(4.0, 4, 2).tolist() == [0]


def test_window_frames_8_every_half_second_at_30fps():
    f = window_frames(np.array([0.0, 2.0]), 4, 8, 30.0)
    # times 0, 0.5, ..., 3.5 -> frames 0, 15, ..., 105; window 2: +60
    assert f[0].tolist() == [0, 15, 30, 45, 60, 75, 90, 105]
    assert f[1].tolist() == [60, 75, 90, 105, 120, 135, 150, 165]


def test_pooled_windows_carry_forward_dropped_frames():
    # kept 1 fps frames at 0, 1, 5 s (2, 3, 4 dropped by dedup) with 2-d vectors
    kept_t = np.array([0.0, 1.0, 5.0])
    v = np.array([[1, 0], [0, 1], [1, 1]], np.float16) / np.array([[1], [1], [np.sqrt(2)]], np.float16)
    out = pooled_windows(kept_t, v, np.array([0.0, 2.0]), 4, 1.0)
    # window 0: samples 0,1,2,3 -> kept 0,1,1,1 -> mean (0.25, 0.75) -> normalised (0.3162, 0.9487)
    assert out[0] == pytest.approx([0.25 / np.sqrt(0.625), 0.75 / np.sqrt(0.625)], abs=1e-3)
    # window 2: samples 2,3,4,5 -> kept 1,1,1,5 -> mean (0.1768, 0.9268) -> normalised (0.1874, 0.9823)
    m = np.array([0.7071 / 4, 3.7071 / 4])
    assert out[1] == pytest.approx(m / np.linalg.norm(m), abs=1e-3)
    assert np.linalg.norm(out, axis=1) == pytest.approx([1, 1], abs=1e-6)


def test_pooled_windows_errors_loudly():
    with pytest.raises(ValueError, match="precedes"):
        pooled_windows(np.array([1.0]), np.ones((1, 2)), np.array([0.0]), 4, 1.0)
    with pytest.raises(ValueError, match="increasing"):
        pooled_windows(np.array([1.0, 1.0]), np.ones((2, 2)), np.array([0.0]), 4, 1.0)
    with pytest.raises(ValueError, match="zero"):
        pooled_windows(np.array([0.0, 1.0]), np.array([[1, 0], [-1, 0]], float), np.array([0.0]), 2, 1.0)
