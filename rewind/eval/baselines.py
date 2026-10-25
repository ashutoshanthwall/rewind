"""Baselines (step 2.3, CLAUDE.md 1.10): how an investigator would go through the footage without Rewind.

All baselines rank the same unit: each clip of the query's split cut into max(1, round(duration / segment_s))
equal segments (a 300.33 s clip -> 5 x 60.07 s; fixed-length cutting would leave 0.33 s slivers that each cost a
rank). Every baseline ranks ALL segments, so it always reaches the event (no miss).

  - Sequential: one camera at a time, each camera's footage in time order. The camera order is arbitrary, so the
    metrics are averaged exactly over every camera order (`camera_orders`), not taken from one chosen order.
  - Random: segments in a seeded random order; metrics averaged over `repeats` seeded shuffles.
  - Witness +-w: a witness gives a time t; first every segment overlapping [t - w, t + w] (one camera at a time,
    time order), then the rest as in Sequential. Averaged over camera orders. The witness time is the event's true
    middle (a perfect witness: the baseline's best case). Specific queries only (a category has no single time).
Segments only come from the split's own clips, so no window reaches into another split's footage.
"""

import itertools
import random
from collections.abc import Sequence
from datetime import datetime, timedelta

from rewind.eval.metrics import Span
from rewind.ingest.manifest import Video


def segments(videos: Sequence[Video], segment_s: float) -> list[Span]:
    """Each clip cut into max(1, round(duration / segment_s)) equal pieces; the last ends exactly at the clip end."""
    if segment_s <= 0:
        raise ValueError(f"segment_s must be positive, got {segment_s}")
    out = []
    for v in videos:
        n = max(1, round((v.end_time - v.start_time).total_seconds() / segment_s))
        cuts = [v.start_time + (v.end_time - v.start_time) * i / n for i in range(n)] + [v.end_time]
        out += [Span(v.camera_id, a, b) for a, b in zip(cuts, cuts[1:])]
    return out


def by_camera(segs: Sequence[Span]) -> dict[str, list[Span]]:
    out: dict[str, list[Span]] = {}
    for s in sorted(segs, key=lambda s: (s.camera_id, s.start)):
        out.setdefault(s.camera_id, []).append(s)
    return out


def camera_orders(segs: Sequence[Span]) -> list[tuple[str, ...]]:
    return list(itertools.permutations(sorted({s.camera_id for s in segs})))


def sequential(segs: Sequence[Span], order: Sequence[str]) -> list[Span]:
    cams = by_camera(segs)
    return [s for cam in order for s in cams[cam]]


def witness(segs: Sequence[Span], t: datetime, window_s: float, order: Sequence[str]) -> list[Span]:
    w = timedelta(seconds=window_s)
    ranked = sequential(segs, order)
    first = [s for s in ranked if s.start < t + w and t - w < s.end]
    return first + [s for s in ranked if not (s.start < t + w and t - w < s.end)]


def shuffled(segs: Sequence[Span], seed: str) -> list[Span]:
    out = sorted(segs, key=lambda s: (s.camera_id, s.start))  # fixed starting order: the seed alone decides
    random.Random(seed).shuffle(out)
    return out
