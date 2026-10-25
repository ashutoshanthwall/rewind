"""Step 4.1: 1 fps sampling with motion-aware dedup (CLAUDE.md 1.6: global embeddings barely change when something
small moves, so dedup decides on pixels inside motion regions, not on embeddings).

Per clip, frames 0, k, 2k, ... (k = video fps / sampling.appearance_fps) are candidates. The first is always kept.
Every later candidate is compared with the LAST KEPT frame (not the previous candidate, so slow drift can't drop a
whole scene) and dropped only if all three hold:
  1. pixel change inside motion regions is below sampling.dedup_motion_pixel_thresh: the share of the analysis
     frame's pixels (grayscale, motion.downscale_width wide) whose grey level changed by more than
     sampling.dedup_pixel_delta, counted only inside the union of the motion boxes (scripts/index_pass.py, MOG2) of
     both frames. A share of the frame, so RGB (640 px wide) and IR (352 px) use the same threshold;
  2. the detections are essentially unchanged (if detection is enabled): every detection with score
     >= sampling.dedup_det_min_score in either frame has a same-class detection (any score) in the other frame with
     IoU >= sampling.dedup_det_iou. Weaker detections are left out of the test because on busy outdoor cameras
     (14-33 detections per frame, mostly parked cars) low-score boxes flicker in and out between frames;
  3. it is less than sampling.max_gap_s after the last kept frame, so kept frames are never more than max_gap_s
     apart (at 1 fps and 10 s: at most 9 candidates dropped in a row).
A threshold of 0 keeps every candidate (no share is below 0): the no-dedup reference.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from rewind.video import keep_step


@dataclass(frozen=True)
class DedupConfig:
    fps: float               # sampling.appearance_fps
    max_gap_s: float
    pixel_thresh: float      # share of analysis-frame pixels; 0 keeps everything
    pixel_delta: int         # grey levels
    det_iou: float
    det_min_score: float
    analysis_width: int      # motion.downscale_width

    @classmethod
    def from_config(cls, cfg: dict, pixel_thresh: float | None = None) -> "DedupConfig":
        """From the full config; pixel_thresh overrides sampling.dedup_motion_pixel_thresh (the 4.1 sweep)."""
        s = cfg["sampling"]
        keys = ("appearance_fps", "max_gap_s", "dedup_pixel_delta", "dedup_det_iou", "dedup_det_min_score")
        if pixel_thresh is None:
            keys += ("dedup_motion_pixel_thresh",)
            pixel_thresh = s.get("dedup_motion_pixel_thresh")
        for key in keys:
            if s.get(key) in (None, "TBD"):
                raise ValueError(f"config sampling.{key} is missing or TBD")
        if not isinstance(pixel_thresh, (int, float)) or isinstance(pixel_thresh, bool) or not 0 <= pixel_thresh <= 1:
            raise ValueError(f"dedup pixel threshold must be a share in [0, 1], got {pixel_thresh!r}")
        return cls(s["appearance_fps"], s["max_gap_s"], float(pixel_thresh), s["dedup_pixel_delta"],
                   s["dedup_det_iou"], s["dedup_det_min_score"], cfg["motion"]["downscale_width"])


@dataclass
class Candidate:
    """One 1 fps candidate frame of a clip."""
    frame: int
    t_s: float                # frame / video fps: seconds in the clip (= proxy time)
    gray: np.ndarray          # uint8, analysis size
    motion_mask: np.ndarray   # bool, analysis size: union of this frame's motion boxes
    det_classes: np.ndarray   # (n,) str
    det_boxes: np.ndarray     # (n, 4) x1 y1 x2 y2, original pixels
    det_scores: np.ndarray    # (n,)


def candidate_step(video_fps: float, appearance_fps: float, motion_fps: float) -> int:
    """Frame step of the 1 fps grid; it must lie on the 5 fps grid of the motion/detection pass."""
    step = keep_step(video_fps, appearance_fps)
    if step % keep_step(video_fps, motion_fps):
        raise ValueError(f"{appearance_fps} fps frames are not on the {motion_fps} fps motion/detection grid")
    return step


def analysis_gray(frame_bgr: np.ndarray, width: int) -> np.ndarray:
    """Grayscale frame scaled down to `width` (never up), area interpolation."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if w <= width:
        return gray
    return cv2.resize(gray, (width, 2 * round(h * width / w / 2)), interpolation=cv2.INTER_AREA)


def boxes_mask(boxes: np.ndarray, frame_size: tuple[int, int], mask_size: tuple[int, int]) -> np.ndarray:
    """Union of boxes (original pixels, (n, 4)) as a bool mask of mask_size (w, h); boxes clipped to the frame."""
    (fw, fh), (mw, mh) = frame_size, mask_size
    mask = np.zeros((mh, mw), dtype=bool)
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        c1, c2 = int(np.floor(max(x1, 0) * mw / fw)), int(np.ceil(min(x2, fw) * mw / fw))
        r1, r2 = int(np.floor(max(y1, 0) * mh / fh)), int(np.ceil(min(y2, fh) * mh / fh))
        mask[r1:r2, c1:c2] = True
    return mask


def changed_share(a: Candidate, b: Candidate, pixel_delta: int) -> float:
    """Share of the analysis frame's pixels inside either frame's motion regions whose grey level changed by more
    than pixel_delta between a and b."""
    if a.gray.shape != b.gray.shape:
        raise ValueError(f"frames of different sizes: {a.gray.shape} vs {b.gray.shape}")
    region = a.motion_mask | b.motion_mask
    if not region.any():
        return 0.0
    diff = cv2.absdiff(a.gray, b.gray) > pixel_delta
    return float(np.count_nonzero(diff & region)) / diff.size


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (n, 4) and (m, 4) boxes."""
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _covered(a: Candidate, b: Candidate, iou: float, min_score: float) -> bool:
    """Every detection of a with score >= min_score has a same-class detection in b with IoU >= iou."""
    for cls in np.unique(a.det_classes[a.det_scores >= min_score]):
        mine = a.det_boxes[(a.det_classes == cls) & (a.det_scores >= min_score)]
        theirs = b.det_boxes[b.det_classes == cls]
        if not len(theirs) or (iou_matrix(mine, theirs).max(axis=1) < iou).any():
            return False
    return True


def detections_unchanged(a: Candidate, b: Candidate, iou: float, min_score: float) -> bool:
    return _covered(a, b, iou, min_score) and _covered(b, a, iou, min_score)


class Deduper:
    """Online keep/drop decisions for the candidates of ONE clip, offered in frame order."""

    def __init__(self, cfg: DedupConfig, use_detections: bool):
        self.cfg, self.use_detections, self.last = cfg, use_detections, None

    def offer(self, c: Candidate) -> bool:
        last, cfg = self.last, self.cfg
        if last is not None and c.frame <= last.frame:
            raise ValueError("candidates must be in increasing frame order")
        keep = (last is None
                or c.t_s - last.t_s >= cfg.max_gap_s
                or changed_share(last, c, cfg.pixel_delta) >= cfg.pixel_thresh
                or (self.use_detections and not detections_unchanged(last, c, cfg.det_iou, cfg.det_min_score)))
        if keep:
            self.last = c
        return keep


def keep_flags(cands: Sequence[Candidate], cfg: DedupConfig, use_detections: bool) -> list[bool]:
    """Keep/drop decision for each candidate of one clip, in frame order (rules in the module docstring)."""
    if any(b.frame <= a.frame for a, b in zip(cands, cands[1:])):
        raise ValueError("candidates must be in increasing frame order")
    d = Deduper(cfg, use_detections)
    return [d.offer(c) for c in cands]
