"""Fusion of the active branches into ranked clips (step 7.4, CLAUDE.md 1.6 "QUERY").

Per branch (appearance, action; the event branch was dropped at Gate 6.2) the engine returns the top
search.k_per_index hits for the query (rewind/search/engine.py). Then:
  1. z-score per branch, over that branch's hits for this query: z = (s - mean) / std (population std). Raw cosine
     scores of different encoders are not comparable; z-scores are. A branch with fewer than 2 hits, or hits that
     all have the same score, gets z = 0 (no spread to normalise; such a hit ranks as an average hit).
  2. weighted fusion: fused = search.weights[branch] * z. A branch with weight 0 contributes no hits at all
     (excluded, not zero-scored, so it cannot form clips); appearance is fixed at 1.0 (PROGRESS, pre-agreed rule).
  3. hits of all branches are pooled and grouped into clips by rewind/search/core.group_hits: a hit's time is its
     t_s (frame time; window centre for the action branch); several hits at the same (video, second) count once, by
     the best fused score; merge within search.merge_gap_s, cap search.max_clip_s, clip score = mean of the best 3
     fused hit scores, shown span = hits +- search.context_s cut to the video.
  4. each clip records the branches with at least one pooled hit (weight > 0) in the same video between the clip's
     first and last hit time ("why it matched"), and its best hit per branch.
"""

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from rewind.eval.metrics import Span
from rewind.ingest.manifest import Video
from rewind.search.core import SearchConfig, group_hits


def zscores(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if len(s) < 2 or s.max() == s.min():     # exact test: s.std() of equal floats can be ~1e-17, not 0
        return np.zeros(len(s))
    return (s - s.mean()) / s.std()


def check_weights(weights: dict, branches) -> dict[str, float]:
    """Weights of the active branches; every one must be a number >= 0 (a TBD weight stops the run)."""
    out = {}
    for b in branches:
        w = weights.get(b)
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
            raise ValueError(f"search.weights.{b} must be a number >= 0, got {w!r}")
        out[b] = float(w)
    return out


@dataclass
class FusedClip:
    video_id: int
    start_s: float
    end_s: float
    score: float
    first_hit_s: float
    last_hit_s: float
    branches: list[str]
    best: dict = field(default_factory=dict)   # branch -> its best pooled hit in the clip (a dict of the hit row)

    def span(self, v: Video) -> Span:
        """The shown span (with context) as wall-clock time on the video's camera."""
        return Span(v.camera_id, v.start_time + timedelta(seconds=self.start_s),
                    v.start_time + timedelta(seconds=self.end_s))


def pool(hits: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Per-branch hits (columns video_id, t_s, score, ...) -> one table with branch, z and fused score.
    Branches with weight 0 or no hits are left out."""
    parts = []
    for b, df in hits.items():
        if b not in weights:
            raise ValueError(f"hits for branch {b!r}, which has no weight")
        if weights[b] == 0 or df.empty:
            continue
        p = df.reset_index(drop=True).copy()
        p["branch"] = b
        p["z"] = zscores(p["score"].to_numpy())
        p["fused"] = weights[b] * p["z"]
        parts.append(p)
    if not parts:
        return pd.DataFrame(columns=["video_id", "t_s", "score", "branch", "z", "fused"])
    return pd.concat(parts, ignore_index=True)


def fuse(hits: dict[str, pd.DataFrame], weights: dict[str, float], durations: dict[int, float],
         cfg: SearchConfig, top_n: int | None = None) -> list[FusedClip]:
    """Rules in the module docstring. durations: video_id -> seconds (clips are cut to the video). top_n: return only
    the best top_n clips (the ranking is computed over all clips either way; evaluation needs the full list)."""
    pooled = pool(hits, weights)
    if pooled.empty:
        return []
    vids = pooled["video_id"].to_numpy()
    ts = pooled["t_s"].to_numpy(np.float64)
    fused = pooled["fused"].to_numpy(np.float64)
    branch = pooled["branch"].to_numpy()
    clips = group_hits(np.arange(len(pooled)), fused, vids, ts, durations, cfg)
    if top_n is not None:
        clips = clips[:top_n]
    by_video = pooled.groupby("video_id").indices          # video_id -> row positions, computed once
    records = pooled.to_dict("records")
    out = []
    for c in clips:
        t_first, t_last = c.hits[0][0], c.hits[-1][0]
        rows = by_video[c.video_id]
        rows = rows[(ts[rows] >= t_first) & (ts[rows] <= t_last)]
        best = {}
        for b in sorted(set(branch[rows])):
            rb = rows[branch[rows] == b]
            best[b] = records[rb[np.argmax(fused[rb])]]      # first of equal maxima, in pooled order
        out.append(FusedClip(c.video_id, c.start_s, c.end_s, c.score, t_first, t_last, sorted(best), best))
    return out
