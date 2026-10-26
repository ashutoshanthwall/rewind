"""Event branch of search (step 6.2; PROGRESS "Phase 6 — PRE-REGISTRATION", Gate 6.2 part 2).

Intent: a query asks for an event type if its lowercased text contains one of that type's keywords
(events.intent_keywords) as a whole word or whole phrase; no stemming ("pulling away" does not match "pulls away").
Only enabled types (events.types) count. Rewind never infers an event the rules did not produce: the branch only
returns detected events (scripts/build_vehicle_events.py) of the intended types.

Ranking: each candidate event is one hit at its event time, scored by cosine(CLIP vector of its best-box crop, CLIP
query vector); the top search.k_per_index become clips through rewind/search/core.group_hits (same merge, cap, top-3
mean and context rules as appearance).
"""

import re

import numpy as np
import pandas as pd

from rewind.search.core import Clip, SearchConfig, group_hits, top_k


def intent(text: str, keywords: dict[str, list[str]], enabled: list[str]) -> list[str]:
    """Enabled event types whose keywords occur in text (in the order of `enabled`)."""
    unknown = set(enabled) - set(keywords)
    if unknown:
        raise ValueError(f"enabled event types without keywords: {sorted(unknown)}")
    low = text.lower()
    return [t for t in enabled
            if any(re.search(r"\b" + re.escape(k.lower()) + r"\b", low) for k in keywords[t])]


def event_clips(q: np.ndarray, vecs: np.ndarray, meta: pd.DataFrame, types: list[str], video_ids: set[int],
                rgb_only: bool, durations: dict[int, float], cfg: SearchConfig) -> list[Clip]:
    """Clips from the detected events of `types` in these videos (RGB only if rgb_only), best first.
    meta: one row per vector with video_id, t_s, event_type, modality."""
    keep = meta["event_type"].isin(types).to_numpy() & meta["video_id"].isin(video_ids).to_numpy()
    if rgb_only:
        keep &= (meta["modality"] == "rgb").to_numpy()
    rows = np.flatnonzero(keep)
    if len(rows) == 0:
        return []
    idx, sc = top_k(vecs[rows], q, cfg.k)
    return group_hits(rows[idx], sc, meta["video_id"].to_numpy(), meta["t_s"].to_numpy(), durations, cfg)
