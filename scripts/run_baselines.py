"""Step 2.3: run the sequential, random and witness +-10 min baselines on the tuning and dev queries.

Never the test split: footage, events and queries come through the guarded accessors (split_video_ids,
split_event_ids, load_queries), which refuse test. Metrics: rewind/eval/metrics.py; baselines:
rewind/eval/baselines.py; settings: config sections `metrics` and `baselines`.

Ground truth:
  - specific query: the frozen event span on its camera;
  - category query: every Kitware (RGB) activity of the type in the split, one instance per activity (pieces cut at
    clip boundaries joined); boundary-flagged activities are don't-care; IR segments are left out of the ranking
    (IR labels are incomplete: no precision/mAP on IR, Gate 0.3).
Writes one experiments.csv row per (baseline, split) and results/reports/baselines_2.3.md (summary, breakdowns by
qtype / modality / specificity, per-query tables). A short run (about a minute).

Run from the repo root:
  uv run python -m scripts.run_baselines
  uv run python -m scripts.run_baselines --no-log   re-render the report only (all inputs are seeded: same numbers)
"""

import argparse
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from rewind.eval.baselines import camera_orders, segments, sequential, shuffled, witness
from rewind.eval.metrics import (Span, average_precision_at_k, bootstrap_ci, drop_dont_care, instance_recall_at_k,
                                 recall_at_k, seconds_watched)
from rewind.eval.queries import load_queries
from rewind.eval.splits import split_event_ids, split_video_ids
from rewind.experiment_log import log_experiment
from rewind.ingest.manifest import load_videos
from scripts.select_footage import markdown_table

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_DIR = REPO_ROOT / "eval" / "queries"
SPLITS_RUN = ("tuning", "dev")  # never test before step 8.2
SYSTEMS = ("sequential", "random", "witness")


def category_truth(conn, split: str, activity_type: str) -> tuple[list[Span], list[Span]]:
    """(instances, don't-care spans) of one activity type in a split: Kitware activities, pieces joined by group."""
    keep = set(split_event_ids(conn, split))
    every = set(split_event_ids(conn, split, include_boundary=True))
    groups: dict[int, list] = defaultdict(list)
    for r in conn.execute("SELECT event_id, group_id, camera_id, start_time, end_time FROM events "
                          "WHERE activity_type = ? AND label_source != 'contrib'", (activity_type,)):
        if r[0] in every:
            groups[r[1]].append(r)
    instances, dont_care = [], []
    for pieces in groups.values():
        if len({p[2] for p in pieces}) != 1:
            raise ValueError(f"group {pieces[0][1]} spans cameras")
        span = Span(pieces[0][2], min(datetime.fromisoformat(p[3]) for p in pieces),
                    max(datetime.fromisoformat(p[4]) for p in pieces))
        (instances if all(p[0] in keep for p in pieces) else dont_care).append(span)
    if not instances:
        raise ValueError(f"{split}: no {activity_type} instances")
    return instances, dont_care


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def specific_metrics(lists: list[list[Span]], gt: Span, footage_s: float, ks: list[int]) -> dict:
    """Metrics averaged over several rankings (camera orders or random repeats)."""
    out = {f"R@{k}": mean(recall_at_k(r, gt, k) for r in lists) for k in ks}
    out["seconds"] = mean(seconds_watched(r, gt, footage_s) for r in lists)
    return out


def category_metrics(lists: list[list[Span]], inst: list[Span], dont_care: list[Span], ks: list[int]) -> dict:
    lists = [drop_dont_care(r, inst, dont_care) for r in lists]
    out = {f"AP@{k}": mean(average_precision_at_k(r, inst, k) for r in lists) for k in ks}
    out |= {f"IR@{k}": mean(instance_recall_at_k(r, inst, k) for r in lists) for k in ks}  # instance recall
    return out


def run_split(conn, split: str, cfg: dict) -> list[dict]:
    m, b = cfg["metrics"], cfg["baselines"]
    ids = set(split_video_ids(conn, split))
    videos = [v for v in load_videos(conn) if v.video_id in ids]
    segs = segments(videos, b["segment_s"])
    rgb_cams = {v.camera_id for v in videos if v.modality == "rgb"}
    rgb_segs = [s for s in segs if s.camera_id in rgb_cams]
    footage_s = sum(v.n_frames / v.fps for v in videos)
    orders, rgb_orders = camera_orders(segs), camera_orders(rgb_segs)
    rows = []
    for q in load_queries(QUERIES_DIR, split):
        base = {k: q[k] for k in ("query_id", "kind", "qtype", "specificity", "modality")} | {"split": split}
        reps = [shuffled(segs if q["kind"] == "specific" else rgb_segs, f"{b['random_seed']}-{q['query_id']}-{r}")
                for r in range(b["random_repeats"])]
        if q["kind"] == "specific":
            gt = Span(q["camera_id"], datetime.fromisoformat(q["start_time"]), datetime.fromisoformat(q["end_time"]))
            t = gt.start + (gt.end - gt.start) / 2
            lists = {"sequential": [sequential(segs, o) for o in orders], "random": reps,
                     "witness": [witness(segs, t, b["witness_window_s"], o) for o in orders]}
            for system, ls in lists.items():
                rows.append(base | {"system": system} | specific_metrics(ls, gt, footage_s, m["recall_k"]))
        else:
            inst, dont_care = category_truth(conn, split, q["activity_type"])
            lists = {"sequential": [sequential(rgb_segs, o) for o in rgb_orders], "random": reps}
            for system, ls in lists.items():
                rows.append(base | {"system": system, "instances": len(inst)}
                            | category_metrics(ls, inst, dont_care, m["category_k"]))
    return rows


def summarise(rows: list[dict], cfg: dict) -> dict:
    """Numeric summary of one (system, split) for experiments.csv."""
    m = cfg["metrics"]
    boot = lambda vals, stat: bootstrap_ci(vals, stat, m["bootstrap_n"], m["bootstrap_seed"], m["ci_level"])
    spec = [r for r in rows if r["kind"] == "specific"]
    cat = [r for r in rows if r["kind"] == "category"]
    out = {"n_specific": len(spec), "n_category": len(cat)}
    if spec:
        secs = [r["seconds"] for r in spec]
        lo, hi = boot(secs, statistics.median)
        out |= {"median_seconds": statistics.median(secs), "median_seconds_ci_lo": lo, "median_seconds_ci_hi": hi,
                "mean_seconds": mean(secs), "worst_seconds": max(secs)}
        for k in m["recall_k"]:
            vals = [r[f"R@{k}"] for r in spec]
            lo, hi = boot(vals, mean)
            out |= {f"R@{k}": mean(vals), f"R@{k}_ci_lo": lo, f"R@{k}_ci_hi": hi}
    if cat:
        for k in m["category_k"]:
            vals = [r[f"AP@{k}"] for r in cat]
            lo, hi = boot(vals, mean)
            out |= {f"mAP@{k}": mean(vals), f"mAP@{k}_ci_lo": lo, f"mAP@{k}_ci_hi": hi,
                    f"IR@{k}": mean(r[f"IR@{k}"] for r in cat)}
    return out


def report(df: pd.DataFrame, summaries: dict, cfg: dict) -> str:
    m, b = cfg["metrics"], cfg["baselines"]
    fmt = lambda x: x if not isinstance(x, (int, float)) else "—" if pd.isna(x) else \
        str(int(x)) if float(x).is_integer() else f"{x:.3f}"  # strings, so pandas can't turn 12 back into 12.0
    lines = [
        "# Baselines (step 2.3)", "",
        "Generated by `uv run python -m scripts.run_baselines`. Tuning and dev only (test is read once, at 8.2).",
        f"Units: {b['segment_s']} s segments of the split's own footage. Sequential and witness are averaged over every",
        f"camera order; random over {b['random_repeats']} seeded shuffles per query. Witness time = the event's true",
        f"middle, window ±{b['witness_window_s'] // 60} min (a perfect witness: the baseline's best case); specific",
        "queries only. Hit = overlap on the same camera. Seconds watched = full segments before the first hit + time",
        "into the hit segment until the event starts (baselines rank all footage, so the miss rule never applies).",
        "Category: Kitware RGB activities of the type in the split; IR segments left out (Gate 0.3); IR@k = share",
        f"of instances found in the top k. CIs: {m['ci_level']:.0%} percentile bootstrap over queries "
        f"({m['bootstrap_n']} resamples, seed {m['bootstrap_seed']}).", "",
    ]
    for split in ("tuning", "dev"):
        lines += [f"## {split}", ""]
        summ = pd.DataFrame({sys: summaries[(sys, split)] for sys in SYSTEMS if (sys, split) in summaries})
        lines += ["### Summary", "", markdown_table(summ.map(fmt).rename_axis("metric")), ""]
        d = df[df.split == split]
        spec = d[d.kind == "specific"]
        for col in ("qtype", "modality", "specificity"):
            t = spec.pivot_table(index=col, columns="system", values="seconds", aggfunc="median")
            n = spec[spec.system == "sequential"].groupby(col).size().rename("n")
            lines += [f"### Specific: median seconds watched by {col}", "",
                      markdown_table(t.join(n).map(fmt)), ""]
        cols = ["query_id", "qtype", "specificity", "modality", "system", "seconds"] + [f"R@{k}" for k in m["recall_k"]]
        lines += ["### Specific queries", "", markdown_table(spec[cols].map(fmt).set_index("query_id")), ""]
        cat = d[d.kind == "category"]
        ccols = ["query_id", "system", "instances"] + [f"{p}@{k}" for k in m["category_k"] for p in ("AP", "IR")]
        lines += ["### Category queries", "", markdown_table(cat[ccols].map(fmt).set_index("query_id")), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "baselines_2.3.md")
    parser.add_argument("--no-log", action="store_true", help="don't append to experiments.csv (report only)")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = [r for split in SPLITS_RUN for r in run_split(conn, split, cfg)]
    df = pd.DataFrame(rows)
    summaries = {}
    for (system, split), g in df.groupby(["system", "split"]):
        summaries[(system, split)] = summarise(g.to_dict("records"), cfg)
    text = report(df, summaries, cfg)
    for (system, split), metrics in sorted(summaries.items()) if not args.no_log else []:
        note = {"sequential": "mean over all camera orders", "random": f"mean over "
                f"{cfg['baselines']['random_repeats']} seeded shuffles", "witness": "perfect witness time, specific only"}
        log_experiment(split, metrics, f"step 2.3 baseline {system} ({note[system]}); "
                                      f"{cfg['baselines']['segment_s']} s segments; results/reports/baselines_2.3.md")
    args.report.write_text(text)
    print(text)
    print(f"\nwrote {args.report.relative_to(REPO_ROOT)}; "
          + ("not logged (--no-log)" if args.no_log else f"logged {len(summaries)} rows to results/experiments.csv"))


if __name__ == "__main__":
    main()
