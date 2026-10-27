"""Steps 2.3 + 3.2: run the sequential, random, witness +-10 min and motion-ranked baselines on tuning and dev.

Never the test split: footage, events and queries come through the guarded accessors (split_video_ids,
split_event_ids, load_queries), which refuse test. Metrics: rewind/eval/metrics.py; baselines:
rewind/eval/baselines.py; settings: config sections `metrics` and `baselines`.

Ground truth:
  - specific query: the frozen event span on its camera;
  - category query: every Kitware (RGB) activity of the type in the split, one instance per activity (pieces cut at
    clip boundaries joined); boundary-flagged activities are don't-care; IR segments are left out of the ranking
    (IR labels are incomplete: no precision/mAP on IR, Gate 0.3).
Motion-ranked (3.2) needs the step 3.1 per-second activity of every clip of the split (artifacts/motion, written by
scripts/index_pass.py; a clip without its done marker stops the run).
Writes one experiments.csv row per logged (baseline, split) and a report (summary, breakdowns by qtype / modality /
specificity, per-query tables). A short run (about a minute).

Run from the repo root:
  uv run python -m scripts.run_baselines
  uv run python -m scripts.run_baselines --no-log   re-render the report only (all inputs are seeded: same numbers)
  uv run python -m scripts.run_baselines --log-systems motion --report results/reports/baselines_3.2.md   (step 3.2)
Logging refuses a working tree with uncommitted code (the row's git hash would not identify the code that ran);
commit first, or pass --allow-dirty for a deliberate exception.
"""

import argparse
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from rewind.eval.baselines import camera_orders, motion_ranked, segment_activity, segments, sequential, shuffled, witness
from rewind.eval.metrics import (Span, average_precision_at_k, bootstrap_ci, drop_dont_care, instance_recall_at_k,
                                 recall_at_k, seconds_watched)
from rewind.eval.queries import load_queries
from rewind.eval.splits import split_event_ids, split_video_ids
from rewind.experiment_log import git_hash, log_experiment
from rewind.ingest.manifest import load_videos
from rewind.motion.store import load_activity
from rewind.paths import display_path
from scripts.select_footage import markdown_table

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_DIR = REPO_ROOT / "eval" / "queries"
SPLITS_RUN = ("tuning", "dev")  # never test before step 8.2
SYSTEMS = ("sequential", "random", "witness", "motion")
NOTES = {"sequential": "mean over all camera orders", "witness": "perfect witness time, specific only",
         "motion": "segments by mean MOG2 activity (step 3.1), one ranking over all cameras"}


def category_truth(conn, split: str, activity_type: str) -> tuple[list[Span], list[Span]]:
    """(instances, don't-care spans) of one activity type in a split: Kitware activities, pieces joined by group."""
    return category_truth_ids(conn, split, activity_type, set(split_event_ids(conn, split)),
                              set(split_event_ids(conn, split, include_boundary=True)))


def category_truth_ids(conn, split: str, activity_type: str, keep: set, every: set) -> tuple[list[Span], list[Span]]:
    """category_truth given the split's event ids (keep: not boundary-flagged; every: all), read by the caller
    through the guarded accessors (step 8.2 reads test itself). split: name for messages."""
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


def motion_scores(videos, segment_s: float, art: Path) -> dict[Span, float]:
    """Mean activity of every segment of the split's clips (each clip's segments from its own per-second scores)."""
    scores = {}
    for v in videos:
        activity = load_activity(art, v)
        for seg in segments([v], segment_s):
            scores[seg] = segment_activity(v, seg, activity)
    return scores


def run_split(conn, split: str, cfg: dict, art: Path) -> list[dict]:
    ids = set(split_video_ids(conn, split))
    return run_videos(conn, split, [v for v in load_videos(conn) if v.video_id in ids], load_queries(QUERIES_DIR, split),
                      lambda t: category_truth(conn, split, t), cfg, art)


def run_videos(conn, split: str, videos: list, queries: list[dict], truth, cfg: dict, art: Path) -> list[dict]:
    """run_split given the split's videos, frozen queries and truth(activity_type) -> (instances, don't-care), read
    by the caller through the guarded accessors (step 8.2 reads test itself). split: name stored in the rows."""
    m, b = cfg["metrics"], cfg["baselines"]
    segs = segments(videos, b["segment_s"])
    scores = motion_scores(videos, b["segment_s"], art)
    if set(scores) != set(segs):
        raise ValueError(f"{split}: motion segments differ from the baseline segments")
    rgb_cams = {v.camera_id for v in videos if v.modality == "rgb"}
    rgb_segs = [s for s in segs if s.camera_id in rgb_cams]
    footage_s = sum(v.n_frames / v.fps for v in videos)
    orders, rgb_orders = camera_orders(segs), camera_orders(rgb_segs)
    motion_all = motion_ranked(scores)
    motion_rgb = [s for s in motion_all if s.camera_id in rgb_cams]
    rows = []
    for q in queries:
        base = {k: q[k] for k in ("query_id", "kind", "qtype", "specificity", "modality")} | {"split": split}
        reps = [shuffled(segs if q["kind"] == "specific" else rgb_segs, f"{b['random_seed']}-{q['query_id']}-{r}")
                for r in range(b["random_repeats"])]
        if q["kind"] == "specific":
            gt = Span(q["camera_id"], datetime.fromisoformat(q["start_time"]), datetime.fromisoformat(q["end_time"]))
            t = gt.start + (gt.end - gt.start) / 2
            lists = {"sequential": [sequential(segs, o) for o in orders], "random": reps,
                     "witness": [witness(segs, t, b["witness_window_s"], o) for o in orders], "motion": [motion_all]}
            for system, ls in lists.items():
                rows.append(base | {"system": system} | specific_metrics(ls, gt, footage_s, m["recall_k"]))
        else:
            inst, dont_care = truth(q["activity_type"])
            lists = {"sequential": [sequential(rgb_segs, o) for o in rgb_orders], "random": reps, "motion": [motion_rgb]}
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
        "# Baselines (steps 2.3 and 3.2)", "",
        "Generated by `uv run python -m scripts.run_baselines`. Tuning and dev only (test is read once, at 8.2).",
        f"Units: {b['segment_s']} s segments of the split's own footage. Sequential and witness are averaged over every",
        f"camera order; random over {b['random_repeats']} seeded shuffles per query. Witness time = the event's true",
        f"middle, window ±{b['witness_window_s'] // 60} min (a perfect witness: the baseline's best case); specific",
        "queries only. Motion: segments by mean MOG2 activity (step 3.1), one ranking over all cameras (step 3.2). Hit = overlap on the same camera. Seconds watched = full segments before the first hit + time",
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
    parser.add_argument("--allow-dirty", action="store_true", help="log even with uncommitted code changes")
    parser.add_argument("--log-systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS),
                        help="which baselines get experiments.csv rows (3.2: motion only; the others were logged at 2.3)")
    parser.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()
    if not args.no_log and not args.allow_dirty and git_hash(REPO_ROOT).endswith("-dirty"):
        raise SystemExit("Uncommitted code changes: commit before a logged run (or --no-log / --allow-dirty)")
    cfg = yaml.safe_load(args.config.read_text())
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = [r for split in SPLITS_RUN for r in run_split(conn, split, cfg, args.art)]
    df = pd.DataFrame(rows)
    summaries = {}
    for (system, split), g in df.groupby(["system", "split"]):
        summaries[(system, split)] = summarise(g.to_dict("records"), cfg)
    text = report(df, summaries, cfg)
    args.report.write_text(text)  # before logging: a failure here must not leave logged rows without their report
    logged = [(k, v) for k, v in sorted(summaries.items()) if k[0] in args.log_systems] if not args.no_log else []
    for (system, split), metrics in logged:
        note = NOTES.get(system, f"mean over {cfg['baselines']['random_repeats']} seeded shuffles")
        step = "3.2" if system == "motion" else "2.3"
        log_experiment(split, metrics, f"step {step} baseline {system} ({note}); "
                                      f"{cfg['baselines']['segment_s']} s segments; {display_path(args.report)}")
    print(text)
    print(f"\nwrote {display_path(args.report)}; "
          + ("not logged (--no-log)" if args.no_log else f"logged {len(logged)} rows to results/experiments.csv"))


if __name__ == "__main__":
    main()
