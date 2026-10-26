"""Evaluation metrics (step 2.3, CLAUDE.md 1.10). Pure functions over ranked lists of clips; no I/O.

A clip (and a ground-truth event) is a span of one camera's footage in local wall-clock time, end exclusive.
Times are compared, never subtracted across clips, except within one clip (same date, so DST can't interfere).

Definitions (decided at step 2.3, see PROGRESS):
  - Hit: a clip overlaps the ground-truth span on the SAME camera with positive length (MEVA has no cross-camera
    links, so the same moment on another camera is not counted).
  - Specific queries (one correct event):
      recall_at_k = 1 if a hit is in the top k, else 0.
      seconds_watched = full duration of every clip ranked before the first hit + the time from the hit clip's
        start to the event's start (0 if the event was already under way). Miss rule: if no clip in the list hits,
        the cost is the whole list's duration + a full review of all footage in scope (footage_s).
  - Category queries (every labelled activity of a type): instances are the ground-truth spans.
      instance_recall_at_k = share of instances overlapped by at least one of the top k clips.
      average_precision_at_k = sum over i <= k of P(i) * rel(i) / min(k, n_instances), where rel(i) = 1 if clip i
        overlaps an instance no earlier clip overlapped (repeats of a found instance don't count) and
        P(i) = (relevant clips in the top i) / i. Clips that overlap only don't-care spans are removed first.
  - Detected events vs annotated events: one-to-one greedy matching of same camera and type by temporal overlap
    with the ANNOTATED span widened by tolerance_s on both sides (so a detection up to tolerance_s outside the label
    still matches, never more), largest overlap first; precision = matched / detected, recall = matched / annotated.
  - Bootstrap: percentile confidence interval of a statistic over queries, resampled with a seeded RNG.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class Span:
    camera_id: str
    start: datetime
    end: datetime  # exclusive

    def __post_init__(self):
        if not self.start < self.end:
            raise ValueError(f"empty or reversed span {self}")

    @property
    def seconds(self) -> float:
        return (self.end - self.start).total_seconds()


def overlaps(a: Span, b: Span) -> bool:
    """Same camera and positive-length overlap of [start, end) spans."""
    return a.camera_id == b.camera_id and a.start < b.end and b.start < a.end


def widen(span: Span, seconds: float) -> Span:
    pad = timedelta(seconds=seconds)
    return Span(span.camera_id, span.start - pad, span.end + pad)


def first_hit(ranked: Sequence[Span], gt: Span) -> int | None:
    """0-based rank of the first clip that hits gt, or None."""
    return next((i for i, clip in enumerate(ranked) if overlaps(clip, gt)), None)


def recall_at_k(ranked: Sequence[Span], gt: Span, k: int) -> int:
    rank = first_hit(ranked, gt)
    return int(rank is not None and rank < k)


def seconds_watched(ranked: Sequence[Span], gt: Span, footage_s: float) -> float:
    """Seconds of footage watched before the event starts on screen (miss rule in the module docstring)."""
    rank = first_hit(ranked, gt)
    if rank is None:
        return sum(c.seconds for c in ranked) + footage_s
    hit = ranked[rank]
    into = max(0.0, (gt.start - hit.start).total_seconds())  # same camera and overlapping: same date
    return sum(c.seconds for c in ranked[:rank]) + into


def drop_dont_care(ranked: Sequence[Span], instances: Sequence[Span], dont_care: Sequence[Span]) -> list[Span]:
    """Remove clips that overlap a don't-care span and no instance (neither right nor wrong)."""
    if not dont_care:
        return list(ranked)
    return [c for c in ranked
            if any(overlaps(c, i) for i in instances) or not any(overlaps(c, d) for d in dont_care)]


def instance_recall_at_k(ranked: Sequence[Span], instances: Sequence[Span], k: int) -> float:
    if not instances:
        raise ValueError("no ground-truth instances")
    found = {j for c in ranked[:k] for j, inst in enumerate(instances) if overlaps(c, inst)}
    return len(found) / len(instances)


def average_precision_at_k(ranked: Sequence[Span], instances: Sequence[Span], k: int) -> float:
    if not instances:
        raise ValueError("no ground-truth instances")
    found: set[int] = set()
    hits, total = 0, 0.0
    for i, clip in enumerate(ranked[:k], start=1):
        new = {j for j, inst in enumerate(instances) if j not in found and overlaps(clip, inst)}
        if new:
            found |= new
            hits += 1
            total += hits / i
    return total / min(k, len(instances))


@dataclass(frozen=True)
class Event:
    span: Span
    activity_type: str


def match_events(detected: Sequence[Event], annotated: Sequence[Event], tolerance_s: float) -> int:
    """Number of one-to-one matches (same camera and type, overlap with the annotated span widened by
    tolerance_s), greedy by overlap length, largest first; ties broken by list order."""
    pairs = []
    for j, a in enumerate(annotated):
        target = widen(a.span, tolerance_s)
        for i, d in enumerate(detected):
            if d.activity_type == a.activity_type and overlaps(d.span, target):
                lo, hi = max(d.span.start, target.start), min(d.span.end, target.end)
                pairs.append((-(hi - lo).total_seconds(), i, j))  # overlap > 0 by the test above
    used_d, used_a = set(), set()
    for _, i, j in sorted(pairs):
        if i not in used_d and j not in used_a:
            used_d.add(i)
            used_a.add(j)
    return len(used_d)


def match_count_us(det: np.ndarray, ann: np.ndarray, tolerance_us: int) -> int:
    """match_events for ONE camera and ONE type, on integer microseconds (fast enough for grid sweeps).
    det, ann: (n, 2) int64 [start, end) in microseconds; same rule and tie-breaking as match_events (overlap with the
    annotated span widened by tolerance_us, largest overlap first, then detected index, then annotated index)."""
    if len(det) == 0 or len(ann) == 0:
        return 0
    lo = np.maximum(det[:, None, 0], ann[None, :, 0] - tolerance_us)
    hi = np.minimum(det[:, None, 1], ann[None, :, 1] + tolerance_us)
    i, j = np.nonzero(hi > lo)
    order = np.lexsort((j, i, -(hi - lo)[i, j]))
    used_d, used_a = set(), set()
    for k in order:
        if i[k] not in used_d and j[k] not in used_a:
            used_d.add(i[k])
            used_a.add(j[k])
    return len(used_d)


def precision_recall(detected: Sequence[Event], annotated: Sequence[Event], tolerance_s: float) -> tuple[float, float]:
    """(precision, recall). Raises on an empty side: 0/0 has no honest value. IR (incomplete labels): use recall only."""
    if not detected or not annotated:
        raise ValueError(f"{len(detected)} detected, {len(annotated)} annotated: precision/recall undefined")
    m = match_events(detected, annotated, tolerance_s)
    return m / len(detected), m / len(annotated)


def bootstrap_ci(values: Sequence[float], stat: Callable[[Sequence[float]], float], n_boot: int, seed: int,
                 level: float) -> tuple[float, float]:
    """Percentile bootstrap interval of stat(values), resampling values with replacement."""
    if len(values) < 2:
        raise ValueError(f"bootstrap needs >= 2 values, got {len(values)}")
    rng = random.Random(seed)
    boots = [stat(rng.choices(values, k=len(values))) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])  # linear interpolation between order stats
    return float(lo), float(hi)


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve: probability that a random positive scores above a random negative, ties count
    1/2 (Mann-Whitney). Raises if either class is empty."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"AUC needs both classes: {len(pos)} positive, {len(neg)} negative")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))
