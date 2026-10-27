"""EXPLORATORY (Ashutosh, 27 Oct; Finding F5): Rewind searched only inside a witness time window. DEV split only.

Why: on test the perfect-witness baseline (segments within ±baselines.witness_window_s of the true event middle
first) beat Rewind on the mean, because knowing the time narrows the search. Rewind + a witness window was not
evaluated on test, and is not now: this runs on dev only and decides nothing.

Systems (specific queries only: a category has no single time):
  - rewind_global: the frozen config, as at 8.2 (all dev footage);
  - witness: the step 2.3 baseline (same code, scripts/run_baselines);
  - rewind_window: the frozen config with the engine search restricted to [t - w, t + w] on every camera's dev
    footage (t = true event middle, the same perfect witness as the baseline; w = baselines.witness_window_s). If no
    clip of that list hits, the investigator falls back to the witness baseline's review: cost = the whole list's
    duration + that query's witness-baseline seconds (so a miss is never cheaper than the list).
Same metric code (rewind/eval/metrics.py). Paired comparisons (wins / ties / losses, median paired difference with a
bootstrap CI) are descriptive: n = 12.

Run from the repo root (~1 min):  uv run python -m scripts.witness_window_dev
"""

import argparse
import statistics
from datetime import datetime
from pathlib import Path

import yaml

from rewind.embed.clip_model import Clip
from rewind.eval.metrics import Span, first_hit, seconds_watched
from rewind.eval.queries import load_queries
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect
from rewind.paths import REPO_ROOT, display_path
from rewind.search.core import SearchConfig
from rewind.search.engine import Engine, EngineConfig, active_branches, branch_model, camera_ranges, wall_seconds
from rewind.search.fusion import check_weights, fuse
from scripts.appearance_search import split_videos
from scripts.fusion_tune import FusedRanker
from scripts.run_baselines import QUERIES_DIR, run_split
from scripts.select_footage import rows_table
from scripts.test_run import fmt_ci

SPLIT = "dev"


def clip_ranges(ranges: dict, t0: float, t1: float) -> dict:
    """camera -> ranges intersected with [t0, t1] (cameras with nothing left are dropped)."""
    out = {}
    for cam, rs in ranges.items():
        cut = [(max(a, t0), min(b, t1)) for a, b in rs if a < t1 and t0 < b]
        if cut:
            out[cam] = cut
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "witness_window_dev.md")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    stamp = run_stamp(args.config)
    w = cfg["baselines"]["witness_window_s"]
    scfg = SearchConfig.from_config(cfg)
    branches = active_branches(cfg)
    weights = check_weights(cfg["search"]["weights"], branches)
    if any(v > 0 for b, v in weights.items() if b != "appearance"):
        raise ValueError("this exploratory run uses the frozen appearance-only config; a non-appearance weight is set")
    conn = connect(args.db)
    videos = split_videos(conn, SPLIT)
    queries = [q for q in load_queries(QUERIES_DIR, SPLIT) if q["kind"] == "specific"]
    engine = Engine(EngineConfig.from_config(cfg), {b: branch_model(cfg, b) for b in branches}, branches)
    clip = Clip(cfg["appearance"]["model"], cfg["appearance"]["pretrained"])
    ranker = FusedRanker(engine, videos, weights, {}, cfg["ingest"]["max_name_duration_diff_s"], SPLIT)
    witness = {r["query_id"]: r["seconds"] for r in run_split(conn, SPLIT, cfg, args.art) if r["system"] == "witness"}

    rows = []
    for q in queries:
        gt = Span(q["camera_id"], datetime.fromisoformat(q["start_time"]), datetime.fromisoformat(q["end_time"]))
        mid = wall_seconds(gt.start + (gt.end - gt.start) / 2)
        z = clip.encode_text([q["query"]])[0]
        glob = [c.span(ranker.by_id[c.video_id]) for c in fuse(ranker.hits(z, False, q["query"]), weights,
                                                               ranker.durations, scfg)]
        win_ranges = clip_ranges(ranker.ranges, mid - w, mid + w)
        hits = {b: engine.search(b, z, ranges=win_ranges) for b, wt in weights.items() if wt > 0}
        win = [c.span(ranker.by_id[c.video_id]) for c in fuse(hits, weights, ranker.durations, scfg)]
        rk = first_hit(win, gt)
        win_s = seconds_watched(win, gt, 0.0) if rk is not None else sum(c.seconds for c in win) + witness[q["query_id"]]
        rows.append({"query_id": q["query_id"], "qtype": q["qtype"], "modality": q["modality"],
                     "global": seconds_watched(glob, gt, ranker.footage_s), "global_rank": first_hit(glob, gt),
                     "witness": witness[q["query_id"]], "window": win_s, "window_rank": rk, "window_clips": len(win),
                     "window_list_s": sum(c.seconds for c in win),
                     "window_footage_s": sum(b - a for rs in win_ranges.values() for a, b in rs)})

    med = statistics.median
    summ = {k: {"median": med(r[k] for r in rows), "mean": statistics.mean(r[k] for r in rows)}
            for k in ("global", "witness", "window")}
    hit_w = sum(r["window_rank"] is not None for r in rows)
    if not args.no_log:
        log_experiment(SPLIT, {"window_median_seconds": summ["window"]["median"],
                               "window_mean_seconds": summ["window"]["mean"], "window_found": hit_w,
                               "global_median_seconds": summ["global"]["median"],
                               "witness_median_seconds": summ["witness"]["median"], "n_specific": len(rows)},
                       notes=f"EXPLORATORY (F5): Rewind inside the witness window ±{w} s, fallback to the witness "
                             f"baseline on a miss; dev only, decides nothing", stamp=stamp)

    def paired(a: str, b: str) -> list:
        d = [r[b] - r[a] for r in rows]       # positive = a needs fewer seconds
        return [f"{a} vs {b}", sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d),
                fmt_ci(d, med, cfg, "{:+,.0f}")]

    fmt_rank = lambda x: "miss" if x is None else str(x + 1)
    lines = [
        "# EXPLORATORY — Rewind inside a witness time window (dev only; decides nothing)", "",
        f"Generated by `scripts/witness_window_dev.py` at git `{stamp['git_hash']}`, config "
        f"`{stamp['config_hash'][:12]}` (the frozen config). Dev: {len(videos)} clips, "
        f"{sum(v.n_frames / v.fps for v in videos) / 3600:.2f} h; {len(rows)} specific queries. Window ±{w} s around the "
        "true event middle (the same perfect witness as the baseline), on every camera. A miss inside the window falls "
        "back to the witness baseline's review (cost = window list + witness seconds). Not evaluated on test (Finding "
        "F5); n = 12, so everything here is descriptive.", "",
        rows_table(["system", "median s [CI]", "mean s", "found"],
                   [["Rewind, all footage (frozen, as 8.2)", fmt_ci([r["global"] for r in rows], med, cfg, "{:,.0f}"),
                     f"{summ['global']['mean']:,.0f}", f"{sum(r['global_rank'] is not None for r in rows)}/{len(rows)}"],
                    ["witness ±10 min baseline", fmt_ci([r["witness"] for r in rows], med, cfg, "{:,.0f}"),
                     f"{summ['witness']['mean']:,.0f}", "(all)"],
                    ["**Rewind inside the window (+ witness fallback)**",
                     fmt_ci([r["window"] for r in rows], med, cfg, "{:,.0f}"), f"{summ['window']['mean']:,.0f}",
                     f"{hit_w}/{len(rows)} in the window list"]]), "",
        "Paired, seconds (positive = first system faster); median paired difference [95% CI]:", "",
        rows_table(["comparison", "faster", "ties", "slower", "median paired difference s [CI]"],
                   [paired("window", "witness"), paired("window", "global"), paired("global", "witness")]), "",
        "## Per query", "",
        rows_table(["query", "qtype", "modality", "Rewind global s / rank", "witness s", "window footage s",
                    "window clips", "Rewind window s / rank"],
                   [[r["query_id"], r["qtype"], r["modality"], f"{r['global']:,.0f} / {fmt_rank(r['global_rank'])}",
                     f"{r['witness']:,.0f}", f"{r['window_footage_s']:,.0f}", r["window_clips"],
                     f"{r['window']:,.0f} / {fmt_rank(r['window_rank'])}"] for r in rows]), ""]
    args.report.write_text("\n".join(lines))
    print("\n".join(lines))
    print("report:", display_path(args.report))


if __name__ == "__main__":
    main()
