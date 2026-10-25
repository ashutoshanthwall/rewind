"""Step 4.2: which object crops of a kept frame get embedded (config section `appearance`).

Two crop sources, chosen independently and stored separately so step 4.4 can compare crop_source none / motion /
detection_then_motion from one embedding run:
  - detection crops: detections clipped to the frame, at least appearance.min_crop_px on their shorter side, ranked
    by appearance.crop_rank, at most appearance.max_crops_per_frame;
      score_area   = score x box area, descending ("largest / most confident first", CLAUDE.md step 4.2);
      moving_first = detections whose box is at least appearance.crop_moving_share covered by motion boxes first,
                     then the rest, each group by score x area. On busy outdoor cameras most detections are parked
                     cars that would fill every slot on every frame.
  - motion crops: MOG2 motion boxes (clipped, same minimum size) by area, descending, at most max_crops_per_frame.
    A motion crop is marked `dup` when selected detection crops cover at least appearance.crop_dup_cover of its area
    (the same object twice).
Variants: none = no crops; motion = the motion crops; detection_then_motion = the detection crops, then non-dup
motion crops in their rank order, up to max_crops_per_frame in total (only the stored top motion crops can fill).
"""

from dataclasses import dataclass

import numpy as np

from rewind.embed.dedup import boxes_mask

VARIANTS = ("none", "motion", "detection_then_motion")


@dataclass(frozen=True)
class CropConfig:
    max_crops: int
    min_crop_px: int
    rank: str              # score_area | moving_first
    moving_share: float
    dup_cover: float
    mask_width: int        # motion.downscale_width: resolution of the coverage masks

    @classmethod
    def from_config(cls, cfg: dict, rank: str | None = None) -> "CropConfig":
        """From the full config; rank overrides appearance.crop_rank (the 4.2 ranking check)."""
        a = cfg["appearance"]
        keys = ("max_crops_per_frame", "min_crop_px", "crop_moving_share", "crop_dup_cover")
        if rank is None:
            keys += ("crop_rank",)
            rank = a.get("crop_rank")
        for key in keys:
            if a.get(key) in (None, "TBD"):
                raise ValueError(f"config appearance.{key} is missing or TBD")
        if rank not in ("score_area", "moving_first"):
            raise ValueError(f"appearance.crop_rank must be score_area or moving_first, got {rank!r}")
        return cls(a["max_crops_per_frame"], a["min_crop_px"], rank, a["crop_moving_share"],
                   a["crop_dup_cover"], cfg["motion"]["downscale_width"])


@dataclass(frozen=True)
class Crop:
    source: str            # detection | motion
    rank: int              # 0-based within its source
    box: tuple             # x1, y1, x2, y2 clipped to the frame, original pixels
    cls: str               # COCO class for detections, '' for motion
    score: float           # detector score; nan for motion
    dup: bool = False      # motion crop covered by selected detection crops


def clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b


def big_enough(boxes: np.ndarray, min_px: int) -> np.ndarray:
    return np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= min_px


def mask_size(frame_size: tuple[int, int], width: int) -> tuple[int, int]:
    w, h = frame_size
    return (w, h) if w <= width else (width, max(1, round(h * width / w)))


def covered_share(boxes: np.ndarray, mask: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """Share of each box's mask cells (same rounding as boxes_mask) inside a bool mask of the whole frame."""
    (fw, fh), (mh, mw) = frame_size, mask.shape
    sat = np.zeros((mh + 1, mw + 1), dtype=np.int64)
    sat[1:, 1:] = mask.cumsum(0).cumsum(1)
    out = np.zeros(len(boxes))
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, dtype=float).reshape(-1, 4)):
        c1, c2 = int(np.floor(max(x1, 0) * mw / fw)), int(np.ceil(min(x2, fw) * mw / fw))
        r1, r2 = int(np.floor(max(y1, 0) * mh / fh)), int(np.ceil(min(y2, fh) * mh / fh))
        n = max(0, r2 - r1) * max(0, c2 - c1)
        if n:
            out[i] = (sat[r2, c2] - sat[r1, c2] - sat[r2, c1] + sat[r1, c1]) / n
    return out


def select_crops(det_classes: np.ndarray, det_boxes: np.ndarray, det_scores: np.ndarray, motion_boxes: np.ndarray,
                 frame_size: tuple[int, int], cfg: CropConfig) -> list[Crop]:
    """Detection crops then motion crops of one frame (rules in the module docstring)."""
    w, h = frame_size
    msize = mask_size(frame_size, cfg.mask_width)
    db = clip_boxes(det_boxes, w, h)
    ok = big_enough(db, cfg.min_crop_px)
    db, dcls, dsc = db[ok], np.asarray(det_classes)[ok], np.asarray(det_scores, dtype=float)[ok]
    area = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])
    key = dsc * area
    if cfg.rank == "moving_first":
        moving = covered_share(db, boxes_mask(motion_boxes, frame_size, msize), frame_size) >= cfg.moving_share
        order = sorted(range(len(db)), key=lambda i: (not moving[i], -key[i], i))
    else:
        order = sorted(range(len(db)), key=lambda i: (-key[i], i))
    crops = [Crop("detection", r, tuple(db[i]), str(dcls[i]), float(dsc[i]))
             for r, i in enumerate(order[:cfg.max_crops])]

    mb = clip_boxes(motion_boxes, w, h)
    mb = mb[big_enough(mb, cfg.min_crop_px)]
    m_area = (mb[:, 2] - mb[:, 0]) * (mb[:, 3] - mb[:, 1])
    m_order = sorted(range(len(mb)), key=lambda i: (-m_area[i], i))[:cfg.max_crops]
    det_mask = boxes_mask(np.array([c.box for c in crops]).reshape(-1, 4), frame_size, msize)
    dup = covered_share(mb[m_order], det_mask, frame_size) >= cfg.dup_cover if m_order else []
    crops += [Crop("motion", r, tuple(mb[i]), "", float("nan"), bool(d)) for r, (i, d) in enumerate(zip(m_order, dup))]
    return crops


def variant(crops: list[Crop], name: str, max_crops: int) -> list[Crop]:
    """The crops a crop_source variant uses, in slot order."""
    if name == "none":
        return []
    motion = [c for c in crops if c.source == "motion"]
    if name == "motion":
        return motion[:max_crops]
    if name == "detection_then_motion":
        det = [c for c in crops if c.source == "detection"][:max_crops]
        return det + [c for c in motion if not c.dup][:max_crops - len(det)]
    raise ValueError(f"unknown crop_source {name!r}; expected one of {VARIANTS}")
