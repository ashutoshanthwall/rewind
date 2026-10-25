"""Search core (steps 4.4 / 7.4): vectors -> scored hits -> clips (config section `search`).

Hits: the top search.k_per_index vectors by cosine with the query. Several vectors of one frame (frame views and
crops) are one hit, scored by the best of them. Grouping, per video (a clip never spans two files):
  1. hits sorted by time; a hit joins the current clip if it is at most search.merge_gap_s after the clip's last hit
     AND the clip (first to last hit) stays within search.max_clip_s; otherwise it starts a new clip (no chain-merged
     giant clips);
  2. clip score = mean of its best 3 hit scores (search.clip_rank mean_top3; fewer hits: mean of those);
  3. the clip shown is [first hit - context_s, last hit + context_s], cut to the video.
Clips are ranked by score, highest first; ties by video id, then start time.
"""

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from rewind.eval.metrics import Span
from rewind.ingest.manifest import Video


@dataclass(frozen=True)
class SearchConfig:
    k: int
    merge_gap_s: float
    max_clip_s: float
    context_s: float
    top_n: int = 3          # mean_top3

    @classmethod
    def from_config(cls, cfg: dict) -> "SearchConfig":
        s = cfg["search"]
        if s["clip_rank"] != "mean_top3":
            raise ValueError(f"search.clip_rank {s['clip_rank']!r} is not implemented (mean_top3 only)")
        return cls(s["k_per_index"], s["merge_gap_s"], s["max_clip_s"], s["context_s"])


@dataclass
class Clip:
    video_id: int
    start_s: float               # shown span in clip seconds (with context)
    end_s: float
    score: float
    hits: list = field(default_factory=list)   # (t_s, score, row index of the best vector)

    def span(self, v: Video) -> Span:
        return Span(v.camera_id, v.start_time + timedelta(seconds=self.start_s),
                    v.start_time + timedelta(seconds=self.end_s))


def top_k(vectors: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine scores of the k best rows (rows and query L2-normalised), best first."""
    scores = vectors.astype(np.float32) @ query.astype(np.float32)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.lexsort((idx, -scores[idx]))]
    return idx, scores[idx]


def group_hits(idx: np.ndarray, scores: np.ndarray, video_ids: np.ndarray, t_s: np.ndarray,
               durations: dict[int, float], cfg: SearchConfig) -> list[Clip]:
    """Hits (row indices into video_ids / t_s, with scores) -> ranked clips (rules in the module docstring)."""
    best: dict[tuple[int, float], tuple[float, int]] = {}
    for i, s in zip(idx.tolist(), scores.tolist()):
        key = (int(video_ids[i]), float(t_s[i]))
        if key not in best or s > best[key][0]:
            best[key] = (s, i)
    by_video: dict[int, list] = {}
    for (vid, t), (s, i) in sorted(best.items()):
        by_video.setdefault(vid, []).append((t, s, i))
    clips = []
    for vid, hits in by_video.items():
        current: list = []
        for hit in hits:
            if current and (hit[0] - current[-1][0] > cfg.merge_gap_s or hit[0] - current[0][0] > cfg.max_clip_s):
                clips.append(_clip(vid, current, durations[vid], cfg))
                current = []
            current.append(hit)
        clips.append(_clip(vid, current, durations[vid], cfg))
    return sorted(clips, key=lambda c: (-c.score, c.video_id, c.start_s))


def _clip(vid: int, hits: list, duration: float, cfg: SearchConfig) -> Clip:
    top = sorted((s for _, s, _ in hits), reverse=True)[:cfg.top_n]
    return Clip(vid, max(0.0, hits[0][0] - cfg.context_s), min(duration, hits[-1][0] + cfg.context_s),
                float(np.mean(top)), list(hits))


def camera_means(vectors: np.ndarray, groups: np.ndarray) -> dict:
    """Mean vector (float32) of each group (camera)."""
    x = vectors.astype(np.float32)
    return {g: x[groups == g].mean(axis=0) for g in np.unique(groups)}


def mean_center(vectors: np.ndarray, groups: np.ndarray, means: dict | None = None) -> np.ndarray:
    """Step 4.3: subtract each group's (camera's) mean vector, then L2-normalise again. No labels used.
    means: precomputed per-group means (e.g. over all footage); default: the means of these vectors."""
    x = vectors.astype(np.float32)
    means = camera_means(x, groups) if means is None else means
    out = np.empty_like(x)
    for g in np.unique(groups):
        if g not in means:
            raise ValueError(f"no mean for group {g!r}")
        m = groups == g
        out[m] = x[m] - means[g]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("a vector equals its camera mean: cannot normalise after centering")
    return out / norms
