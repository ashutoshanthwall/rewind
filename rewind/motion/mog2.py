"""MOG2 motion pass (step 3.1): per-second activity score + merged motion boxes for one clip.

Per sampled frame (config `motion`, ~5 fps, frame scaled down to `downscale_width`, grayscale):
  1. OpenCV MOG2 foreground mask; shadow pixels (value 127 when detect_shadows) are NOT foreground.
  2. Morphological opening (open_kernel_px) removes speckle noise.
  3. Activity = fraction of foreground pixels after opening (0..1; comparable between frame sizes).
  4. Motion boxes: the opened mask is dilated (merge_dilate_px) so that nearby blobs of one object merge, then
     connected components with area >= min_box_area_px (pixels of the analysis frame) become boxes, scaled back
     to ORIGINAL video pixels.
Per-second activity = mean over the sampled frames whose time falls in that second (frame f is in second
floor(f / fps)); a short last second uses the samples it has.

Warm-up: a fresh MOG2 model marks the whole first frame as background and needs a few seconds to settle. The first
warmup_s seconds of samples are fed once to train the model, then the clip is processed from frame 0 again with the
trained model. Each clip starts a fresh model (clips are independent units; resuming needs no model state).
"""

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

SHADOW_VALUE = 127      # OpenCV MOG2 marks shadows 127, foreground 255
FOREGROUND_VALUE = 255


@dataclass(frozen=True)
class MotionConfig:
    fps: float
    downscale_width: int
    history_frames: int
    var_threshold: float
    detect_shadows: bool
    warmup_s: float
    open_kernel_px: int
    merge_dilate_px: int
    min_box_area_px: int

    @classmethod
    def from_config(cls, cfg: dict) -> "MotionConfig":
        """From the `motion` section of config.yaml; a missing key or a TBD value raises."""
        values = {}
        for name in cls.__dataclass_fields__:
            if name not in cfg or cfg[name] == "TBD":
                raise ValueError(f"config motion.{name} is missing or TBD")
            values[name] = cfg[name]
        return cls(**values)


@dataclass
class FrameMotion:
    frame: int
    activity: float
    boxes: list[tuple[int, int, int, int]]   # x1, y1, x2, y2 in original video pixels (x2, y2 exclusive)


class MotionScorer:
    """MOG2 state for one clip. Call warm_up() once with the first samples, then score() each sample in order."""

    def __init__(self, cfg: MotionConfig, orig_size: tuple[int, int]):
        self.cfg = cfg
        self.orig_w, self.orig_h = orig_size
        self.mog = cv2.createBackgroundSubtractorMOG2(history=cfg.history_frames, varThreshold=cfg.var_threshold,
                                                      detectShadows=cfg.detect_shadows)
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel_px, cfg.open_kernel_px))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.merge_dilate_px, cfg.merge_dilate_px))

    @staticmethod
    def gray(frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def warm_up(self, frames_bgr: Iterable[np.ndarray]) -> None:
        for frame in frames_bgr:
            self.mog.apply(self.gray(frame))

    def score(self, frame_index: int, frame_bgr: np.ndarray) -> FrameMotion:
        mask = self.mog.apply(self.gray(frame_bgr))
        fg = (mask == FOREGROUND_VALUE).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.open_kernel)
        activity = float(fg.mean())
        return FrameMotion(frame_index, activity, self.boxes(fg))

    def boxes(self, fg: np.ndarray) -> list[tuple[int, int, int, int]]:
        merged = cv2.dilate(fg, self.dilate_kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        h, w = fg.shape
        sx, sy = self.orig_w / w, self.orig_h / h
        out = []
        for x, y, bw, bh, area in stats[1:]:   # row 0 is the background
            if area < self.cfg.min_box_area_px:
                continue
            out.append((int(x * sx), int(y * sy), min(self.orig_w, round((x + bw) * sx)),
                        min(self.orig_h, round((y + bh) * sy))))
        return out


def score_clip(samples: Iterable[tuple[int, np.ndarray]], cfg: MotionConfig,
               orig_size: tuple[int, int]) -> list[FrameMotion]:
    """MOG2 over one clip's samples (index, BGR frame at analysis size), with the warm-up described above.

    Samples are buffered only for the warm-up window (warmup_s * motion fps frames at 640 px: a few MB)."""
    scorer = MotionScorer(cfg, orig_size)
    n_warm = max(1, round(cfg.warmup_s * cfg.fps))
    buffer, out = [], []
    for index, frame in samples:
        if buffer is not None:
            buffer.append((index, frame))
            if len(buffer) < n_warm:
                continue
            scorer.warm_up(f for _, f in buffer)
            out.extend(scorer.score(i, f) for i, f in buffer)
            buffer = None
            continue
        out.append(scorer.score(index, frame))
    if buffer:  # clip shorter than the warm-up window
        scorer.warm_up(f for _, f in buffer)
        out.extend(scorer.score(i, f) for i, f in buffer)
    return out


def per_second(frames: list[FrameMotion], video_fps: float, n_frames: int) -> np.ndarray:
    """Mean activity per clip second; second s covers frames [s * fps, (s + 1) * fps). Every second of the clip
    must have at least one sample (else the sampling step is wrong): raises otherwise."""
    n_seconds = int(np.ceil(n_frames / video_fps))
    sums, counts = np.zeros(n_seconds), np.zeros(n_seconds, dtype=int)
    for fm in frames:
        s = int(fm.frame // video_fps)
        sums[s] += fm.activity
        counts[s] += 1
    if (counts == 0).any():
        raise ValueError(f"seconds without any motion sample: {np.flatnonzero(counts == 0)[:5].tolist()}")
    return sums / counts
