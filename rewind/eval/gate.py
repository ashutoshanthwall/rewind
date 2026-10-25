"""Paired gate rule (Gates 4.4, 5.1, 6.2, 8.1; PROGRESS open-issue item 8, confirmed by Ashutosh 25 Oct).

"Adopt the more complex variant only if its mean is better AND it wins on more queries than it loses; otherwise keep
the simpler one." Operational form (written in code before any dev result of step 4.4 existed):
  - Per query, each query is scored by its own metric: specific -> seconds watched (lower is better); category ->
    AP@k (higher is better, k = max(metrics.category_k)). A specific query that both variants miss (no hit in either
    list) is a tie: the miss rule's cost then differs only by list length, which says nothing about finding it.
  - wins / ties / losses are counted over all queries of the comparison.
  - "Mean is better" per family: mean seconds (specific) lower, mean AP@k (category) higher. With both families
    present the complex variant must be strictly better in at least one and not worse in the other (a better
    specific mean bought with a worse category mean is not "better").
  - adopt = mean better AND wins > losses. Always reported: wins / ties / losses and the paired bootstrap CI of each
    family's mean difference (not required to exclude 0).
"""

import statistics
from collections.abc import Sequence

from rewind.eval.metrics import bootstrap_ci


def paired_decision(simple: Sequence[dict], complex_: Sequence[dict], ap_key: str, n_boot: int, seed: int,
                    ci_level: float) -> dict:
    """simple / complex_: per-query rows with query_id, kind, and seconds + hit (specific) or ap_key (category)."""
    a = {r["query_id"]: r for r in simple}
    b = {r["query_id"]: r for r in complex_}
    if set(a) != set(b):
        raise ValueError("the two variants were run on different queries")
    wins = ties = losses = 0
    spec_diff, cat_diff = [], []      # positive = complex better
    for q in sorted(a):
        ra, rb = a[q], b[q]
        if ra["kind"] == "specific":
            d = ra["seconds"] - rb["seconds"]
            spec_diff.append(d)
            if not ra["hit"] and not rb["hit"]:
                d = 0.0
        else:
            d = rb[ap_key] - ra[ap_key]
            cat_diff.append(d)
        wins, ties, losses = wins + (d > 0), ties + (d == 0), losses + (d < 0)
    fam = {}
    for name, diffs in (("specific_seconds", spec_diff), ("category_" + ap_key, cat_diff)):
        if diffs:   # a CI needs >= 2 queries; with one it is undefined (None), reported as such
            ci = bootstrap_ci(diffs, statistics.mean, n_boot, seed, ci_level) if len(diffs) >= 2 else None
            fam[name] = {"n": len(diffs), "mean_diff": statistics.mean(diffs), "ci": ci}
    better = [f["mean_diff"] > 0 for f in fam.values()]
    worse = [f["mean_diff"] < 0 for f in fam.values()]
    mean_better = any(better) and not any(worse)
    return {"wins": wins, "ties": ties, "losses": losses, "families": fam, "mean_better": mean_better,
            "adopt": mean_better and wins > losses}
