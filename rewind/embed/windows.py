"""Step 5.1: short sliding windows for the action branch (config section `action`), shared by every candidate.

Windows of one clip: [s, s + window_s) for s = 0, stride_s, 2 * stride_s, ... while s + window_s <= the clip's
duration (n_frames / fps). A window's search time is its centre, s + window_s / 2: grouping (rewind/search/core.py)
then shows [first centre - context_s, last centre + context_s], which contains every hit window.

Two ways to fill a window:
  - frames_per_window frames at s + i * window_s / frames_per_window (i = 0 .. n-1), each the decoded frame
    round(t * fps) (video models, e.g. 8 frames every 0.5 s);
  - mean-pooled CLIP (candidate a, built from the step 4.2 appearance vectors, no new embedding run): the 1 fps
    sample times inside [s, s + window_s) (4 at 1 fps); a sample time whose frame dedup dropped (step 4.1) takes the
    vector of the last kept frame at or before it, which dedup judged essentially unchanged; the mean of those
    vectors, L2-normalised again.
"""

import numpy as np


def window_starts(duration_s: float, window_s: float, stride_s: float) -> np.ndarray:
    """Start times (s) of every whole window of a clip."""
    if window_s <= 0 or stride_s <= 0:
        raise ValueError(f"window_s {window_s} and stride_s {stride_s} must be > 0")
    n = int(np.floor((duration_s - window_s) / stride_s + 1e-9)) + 1
    return np.arange(max(n, 0)) * float(stride_s)


def window_frames(starts: np.ndarray, window_s: float, n_frames_per_window: int, fps: float) -> np.ndarray:
    """(n_windows, n_frames_per_window) decoded-frame indices: frame round(t * fps) at the window's sample times."""
    offsets = np.arange(n_frames_per_window) * (window_s / n_frames_per_window)
    return np.rint((starts[:, None] + offsets[None, :]) * fps).astype(np.int64)


def pooled_windows(kept_t: np.ndarray, kept_vecs: np.ndarray, starts: np.ndarray, window_s: float,
                   sample_fps: float) -> np.ndarray:
    """Mean-pooled window vectors (float32, L2-normalised), one per start.

    kept_t: times (s) of the kept 1 fps frames of one clip, increasing; kept_vecs: their vectors, same order.
    Sample times are multiples of 1 / sample_fps in [s, s + window_s); each takes the last kept frame at or before it."""
    kept_t = np.asarray(kept_t, float)
    if len(kept_t) == 0 or (np.diff(kept_t) <= 0).any():
        raise ValueError("kept frame times must be non-empty and strictly increasing")
    step = 1.0 / sample_fps
    out = np.empty((len(starts), kept_vecs.shape[1]), np.float32)
    for i, s in enumerate(starts):
        samples = np.arange(np.ceil(s / step - 1e-9), np.ceil((s + window_s) / step - 1e-9)) * step
        j = np.searchsorted(kept_t, samples + 1e-6, side="right") - 1
        if (j < 0).any():
            raise ValueError(f"window at {s} s: a sample time precedes the first kept frame ({kept_t[0]} s)")
        m = kept_vecs[j].astype(np.float32).mean(axis=0)
        norm = np.linalg.norm(m)
        if norm == 0:
            raise ValueError(f"window at {s} s: mean vector is zero, cannot normalise")
        out[i] = m / norm
    return out
