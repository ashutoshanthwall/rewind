"""Within-clip vehicle tracks and the vehicle_stops / vehicle_starts rules (step 6.1; PROGRESS "Phase 6 —
PRE-REGISTRATION"). Pure functions; no I/O.

Tracks (CLAUDE.md 1.2): one ByteTrack tracker per clip, created fresh, so nothing links tracks across clips or
cameras. The `track` column is a within-clip index (0..n-1 by first appearance), used only to compute speed; it is
never an identity and never leaves the tracks artifacts (no SQLite table, report or UI shows it).

Tracker input: the clip's detections of events.vehicle_classes, merged class-agnostically (YOLOX's per-class NMS can
box one vehicle as both car and truck; such pairs with IoU >= detection.nms_iou keep the higher score), on every
sample of the detection grid (frames 0, step, 2*step, ...; frames without detections are empty updates). Rows are
written for samples where a track was matched to a detection, with that detection's own box, class and score.

Speed of a track at sample i: the track's samples within +/- smooth_s / 2 of t_i; distance between the box centres
of the first and last of them / their time difference; NaN if that difference is < smooth_s / 2. Unit px_per_s, or
box_h_per_s (divided by the median box height of those samples).

Rules: label each sample moving (speed >= stop_speed), stopped (< stop_speed) or none (NaN); runs = maximal
consecutive samples of one label, a none sample ends a run; duration = last time - first time.
  vehicle_stops  = a moving run lasting >= min_moving_s, then (next sample) a stopped run lasting >= min_stopped_s;
  vehicle_starts = a stopped run lasting >= min_stopped_s, then a moving run lasting >= min_moving_s.
Detected span = [last sample of the first run, first sample of the second run); event time = its end.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rewind.track.bytetrack_vendor import BYTETracker

EPS = 1e-6           # float tolerance on times (seconds); times are frame / fps
SPEED_UNITS = ("px_per_s", "box_h_per_s")
EVENT_TYPES = ("vehicle_stops", "vehicle_starts")
TRACK_COLUMNS = ["frame", "t_s", "track", "class", "score", "x1", "y1", "x2", "y2"]


@dataclass(frozen=True)
class TrackerConfig:
    track_thresh: float
    match_thresh: float
    track_buffer: int        # frames at the tracker's nominal 30 fps (ByteTrack convention)
    mot20: bool
    frame_rate: float        # samples per second fed to the tracker (= detection.fps)
    classes: tuple
    merge_iou: float         # class-agnostic merge of vehicle detections (= detection.nms_iou)

    @classmethod
    def from_config(cls, cfg: dict) -> "TrackerConfig":
        b = cfg["events"]["bytetrack"]
        return cls(b["track_thresh"], b["match_thresh"], b["track_buffer"], b["mot20"], cfg["detection"]["fps"],
                   tuple(cfg["events"]["vehicle_classes"]), cfg["detection"]["nms_iou"])


@dataclass(frozen=True)
class RuleParams:
    speed_unit: str
    stop_speed: float
    min_stopped_s: float
    min_moving_s: float
    smooth_s: float

    def __post_init__(self):
        if self.speed_unit not in SPEED_UNITS:
            raise ValueError(f"speed_unit {self.speed_unit!r} not in {SPEED_UNITS}")
        if min(self.stop_speed, self.min_stopped_s, self.min_moving_s, self.smooth_s) <= 0:
            raise ValueError(f"rule parameters must be > 0: {self}")

    @classmethod
    def from_config(cls, cfg: dict) -> "RuleParams":
        e = cfg["events"]
        return cls(e["speed_unit"], e["stop_speed"], e["min_stopped_s"], e["min_moving_s"], e["smooth_s"])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of (n, 4) and (m, 4) x1, y1, x2, y2 boxes (continuous coordinates)."""
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def merge_classes(frame_dets: pd.DataFrame, iou: float) -> pd.DataFrame:
    """Class-agnostic NMS within one frame: highest score first; drop boxes with IoU >= iou to a kept one."""
    d = frame_dets.sort_values("score", ascending=False, kind="stable")
    boxes = d[["x1", "y1", "x2", "y2"]].to_numpy(float)
    keep = []
    for i in range(len(d)):
        if not keep or iou_matrix(boxes[i:i + 1], boxes[keep]).max() < iou:
            keep.append(i)
    return d.iloc[keep]


class _TrackerArgs:
    def __init__(self, c: TrackerConfig):
        self.track_thresh, self.track_buffer, self.match_thresh, self.mot20 = (c.track_thresh, c.track_buffer,
                                                                               c.match_thresh, c.mot20)


def track_clip(dets: pd.DataFrame, frames: np.ndarray, fps: float, cfg: TrackerConfig) -> pd.DataFrame:
    """dets: one clip's detections (frame, t_s, class, score, x1..y2); frames: the clip's full sample grid (sorted);
    fps: the video's frame rate (t_s = frame / fps). Returns TRACK_COLUMNS rows (module docstring)."""
    if len(frames) == 0 or np.any(np.diff(frames) <= 0):
        raise ValueError("frames must be a non-empty increasing sample grid")
    off_grid = set(dets["frame"].unique()) - set(frames.tolist())
    if off_grid:
        raise ValueError(f"{len(off_grid)} detection frames are not on the sample grid, e.g. {sorted(off_grid)[:3]}")
    veh = dets[dets["class"].isin(cfg.classes)]
    by_frame = {f: merge_classes(g, cfg.merge_iou) for f, g in veh.groupby("frame")}
    tracker = BYTETracker(_TrackerArgs(cfg), frame_rate=cfg.frame_rate)
    rows = []
    empty = np.empty((0, 5))
    for f in frames.tolist():
        g = by_frame.get(f)
        arr = empty if g is None else g[["x1", "y1", "x2", "y2", "score"]].to_numpy(float)
        for st in tracker.update(arr):
            if st.frame_id != tracker.frame_id:
                raise AssertionError("ByteTrack returned a track not matched in this frame")
            # det_tlbr went tlbr -> tlwh -> tlbr inside ByteTrack (float rounding): match within 1e-3 px, uniquely
            hit = np.flatnonzero(np.all(np.abs(arr[:, :4] - st.det_tlbr) < 1e-3, axis=1))
            if len(hit) != 1:
                raise AssertionError(f"frame {f}: track box matches {len(hit)} detections")
            r = g.iloc[hit[0]]
            rows.append((f, f / fps, st.track_id, r["class"], float(r["score"]), *arr[hit[0], :4]))
    out = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    if len(out):   # within-clip index by first appearance: 0..n-1
        first = out.groupby("track")["frame"].min().sort_values(kind="stable")
        out["track"] = out["track"].map({t: i for i, t in enumerate(first.index)})
    return out.astype({"frame": "int64", "t_s": "float64", "track": "int64", "score": "float32", "x1": "float32",
                       "y1": "float32", "x2": "float32", "y2": "float32"})


def track_speed(t: np.ndarray, boxes: np.ndarray, smooth_s: float, unit: str) -> np.ndarray:
    """Speed per sample of one track (module docstring). t: (n,) increasing seconds; boxes: (n, 4) x1, y1, x2, y2."""
    if unit not in SPEED_UNITS:
        raise ValueError(f"unit {unit!r} not in {SPEED_UNITS}")
    if np.any(np.diff(t) <= 0):
        raise ValueError("track times must increase")
    half = smooth_s / 2
    lo = np.searchsorted(t, t - half - EPS, side="left")
    hi = np.searchsorted(t, t + half + EPS, side="right") - 1
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    dt = t[hi] - t[lo]
    ok = dt >= half - EPS
    speed = np.full(len(t), np.nan)
    speed[ok] = np.hypot(cx[hi] - cx[lo], cy[hi] - cy[lo])[ok] / dt[ok]
    if unit == "box_h_per_s":
        h = boxes[:, 3] - boxes[:, 1]
        med = np.array([np.median(h[a:b + 1]) for a, b in zip(lo, hi)])
        speed[ok] = speed[ok] / med[ok]
    return speed


def runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Maximal runs of equal labels as (label, first index, last index); label -1 (none) ends a run, no run."""
    out = []
    i = 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        if labels[i] != -1:
            out.append((int(labels[i]), i, j))
        i = j + 1
    return out


def detect_events(t: np.ndarray, speed: np.ndarray, p: RuleParams) -> list[tuple[str, float, float]]:
    """(event type, span start s, span end s = event time) of one track, in time order (module docstring)."""
    labels = np.where(np.isnan(speed), -1, (speed >= p.stop_speed).astype(int))
    out = []
    rs = runs(labels)
    for (la, a0, a1), (lb, b0, b1) in zip(rs, rs[1:]):
        if b0 != a1 + 1:      # a none sample in between: not one transition
            continue
        dur_a, dur_b = t[a1] - t[a0], t[b1] - t[b0]
        if la == 1 and lb == 0 and dur_a >= p.min_moving_s - EPS and dur_b >= p.min_stopped_s - EPS:
            out.append(("vehicle_stops", float(t[a1]), float(t[b0])))
        elif la == 0 and lb == 1 and dur_a >= p.min_stopped_s - EPS and dur_b >= p.min_moving_s - EPS:
            out.append(("vehicle_starts", float(t[a1]), float(t[b0])))
    return out


def best_box(track: pd.DataFrame, t_event: float, p: RuleParams) -> pd.Series:
    """The track's highest-score row in [t_event - min_moving_s, t_event + min_stopped_s] (ties: larger area)."""
    w = track[(track["t_s"] >= t_event - p.min_moving_s - EPS) & (track["t_s"] <= t_event + p.min_stopped_s + EPS)]
    if w.empty:
        raise ValueError(f"no track sample near the event at {t_event:.2f} s")
    area = (w["x2"] - w["x1"]) * (w["y2"] - w["y1"])
    order = np.lexsort((-area.to_numpy(), -w["score"].to_numpy()))
    return w.iloc[order[0]]


def majority_class(track: pd.DataFrame) -> str:
    """Most frequent class of a track's boxes (ties: alphabetical, so the result is deterministic)."""
    counts = track["class"].value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def clip_events(tracks: pd.DataFrame, p: RuleParams) -> pd.DataFrame:
    """All detected events of one clip's tracks: event_type, start_s, end_s, t_s (= end_s), class, best box (x1..y2,
    box_frame, box_t_s, box_score). No track index in the output."""
    rows = []
    for _, tr in tracks.groupby("track", sort=True):
        tr = tr.sort_values("t_s")
        t = tr["t_s"].to_numpy(float)
        sp = track_speed(t, tr[["x1", "y1", "x2", "y2"]].to_numpy(float), p.smooth_s, p.speed_unit)
        events = detect_events(t, sp, p)
        if events:
            cls = majority_class(tr)
        for etype, s0, s1 in events:
            b = best_box(tr, s1, p)
            rows.append((etype, s0, s1, s1, cls, float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]),
                         int(b["frame"]), float(b["t_s"]), float(b["score"])))
    cols = ["event_type", "start_s", "end_s", "t_s", "class", "x1", "y1", "x2", "y2", "box_frame", "box_t_s",
            "box_score"]
    return pd.DataFrame(rows, columns=cols).sort_values(["t_s", "event_type"], kind="stable").reset_index(drop=True)
