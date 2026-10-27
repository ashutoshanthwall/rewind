"""Phase 8 preparation, task 1 (Ashutosh's decisions after Phase 7, PROGRESS.md): consistency check on DEV.

Does the Phase 7 pipeline (C++ engine -> fusion -> clips, the scripts/fusion_tune.py path) with action weight 0
reproduce the Gate 4.4 dev result of pad/motion/raw (NumPy brute force, scripts/appearance_search.py)? Gate 4.4
numbers to reproduce (results/reports/ablation_4.4.md, scripts.gate44_tiebreak): median 394 s watched, 9/12
specific queries found, R@10 0.333, category mAP@50 0.110.

Three runs on dev at the config's search settings: (a) engine HNSW (ef_search of the config), (b) engine exact
brute force (isolates the HNSW approximation), (c) the step 4.4 NumPy Index (pad/motion/raw). Per query: seconds /
hit rank (specific), AP@50 (category). Then the tuning vs dev gap: the same weight-0 engine run on tuning, with
footage hours, miss cost, qtype / modality mix and per-query seconds of both splits.

Run from the repo root (~1 min): uv run python -m scripts.consistency_8
"""

import argparse
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.embed.clip_model import Clip
from rewind.eval.queries import load_queries
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect
from rewind.paths import REPO_ROOT, display_path
from rewind.search.core import SearchConfig
from rewind.search.engine import Engine, EngineConfig, active_branches, branch_model
from scripts.appearance_search import Index, query_rows, split_videos
from scripts.collect_appearance import index_dir
from scripts.fusion_tune import FusedRanker
from scripts.run_baselines import QUERIES_DIR, summarise
from scripts.select_footage import rows_table

GATE44 = {"median_seconds": 394, "hits": 9, "R@10": 0.333, "mAP@50": 0.110}   # ablation_4.4.md, pad/motion/raw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "consistency_8.md")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    stamp = run_stamp(args.config)
    a = cfg["appearance"]
    if (a["frame_mode"], a["crop_source"], a["mean_center"]) != ("pad", "motion", False):
        raise ValueError("the Gate 4.4 reference is pad/motion/raw; the config's appearance variant differs")
    scfg = SearchConfig.from_config(cfg)
    branches = active_branches(cfg)
    conn = connect(args.db)
    engine = Engine(EngineConfig.from_config(cfg), {b: branch_model(cfg, b) for b in branches}, branches)
    clip = Clip(a["model"], a["pretrained"])
    weights = {b: (1.0 if b == "appearance" else 0.0) for b in branches}
    join = cfg["ingest"]["max_name_duration_diff_s"]
    d = index_dir(args.art, a["model"], a["pretrained"])
    vecs, meta = np.load(d / "vectors.npy"), pd.read_parquet(d / "meta.parquet")

    rows, videos = {}, {}
    for split in ("dev", "tuning"):
        videos[split] = split_videos(conn, split)
        rows[(split, "hnsw")] = query_rows(conn, split, FusedRanker(engine, videos[split], weights, {}, join, split),
                                           clip, cfg, scfg)
    rows[("dev", "exact")] = query_rows(conn, "dev", FusedRanker(engine, videos["dev"], weights, {}, join, "dev",
                                                                 exact=True), clip, cfg, scfg)
    np_index = Index(vecs, meta, videos["dev"], "pad", "motion", a["max_crops_per_frame"], False)
    rows[("dev", "numpy")] = query_rows(conn, "dev", np_index, clip, cfg, scfg)
    summ = {k: summarise(r, cfg) for k, r in rows.items()}
    hits = {k: sum(r["hit"] for r in v if r["kind"] == "specific") for k, v in rows.items()}
    if not args.no_log:
        for (split, how), s in summ.items():
            log_experiment(split, s | {"hits": hits[(split, how)]},
                           notes=f"phase 8 consistency check: action weight 0, {how} "
                                 f"({'engine' if how != 'numpy' else 'step 4.4 NumPy Index'}), ef "
                                 f"{cfg['search']['ef_search']}", stamp=stamp)

    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    runs = [("dev", "hnsw"), ("dev", "exact"), ("dev", "numpy")]
    by_q = {k: {r["query_id"]: r for r in rows[k]} for k in runs}
    diff = {k: [q for q in by_q[("dev", "numpy")] if _cell(by_q[("dev", "numpy")][q]) != _cell(by_q[k][q])]
            for k in runs}
    match = {}
    for k in runs:
        s0 = summ[k]
        match[k] = (round(s0["median_seconds"]) == GATE44["median_seconds"] and hits[k] == GATE44["hits"]
                    and round(s0["R@10"], 3) == GATE44["R@10"] and round(s0["mAP@50"], 3) == GATE44["mAP@50"])

    lines = [
        "# Phase 8 preparation — consistency check (dev) and the tuning vs dev gap", "",
        f"Generated by `scripts/consistency_8.py` at git `{stamp['git_hash']}`, config `{stamp['config_hash'][:12]}`. "
        f"Action weight 0 (appearance only), {a['frame_mode']}/{a['crop_source']}/raw, top {scfg.k} per branch, "
        f"ef_search {cfg['search']['ef_search']}.", "",
        "## Dev: engine + fusion vs Gate 4.4", "",
        rows_table(["run", "median s", "mean s", "hits / specific", "R@1", "R@5", "R@10", "mAP@10", "mAP@50"],
                   [["Gate 4.4 report (NumPy)", GATE44["median_seconds"], "", f"{GATE44['hits']}/12", "", "",
                     GATE44["R@10"], "", GATE44["mAP@50"]]] +
                   [[f"{how} ({'engine' if how != 'numpy' else 'NumPy Index, re-run'})", fmt(summ[(s, how)]["median_seconds"]),
                     fmt(summ[(s, how)]["mean_seconds"]), f"{hits[(s, how)]}/{summ[(s, how)]['n_specific']}",
                     *(fmt(summ[(s, how)][f"R@{k}"]) for k in (1, 5, 10)), fmt(summ[(s, how)]["mAP@10"]),
                     fmt(summ[(s, how)]["mAP@50"])] for s, how in runs]), "",
        *(f"- **{how}** reproduces the Gate 4.4 summary: **{'YES' if match[(s, how)] else 'NO'}**; queries that "
          f"differ from the NumPy re-run: {', '.join(diff[(s, how)]) or 'none'}" for s, how in runs), "",
        "A query that differs only between HNSW and exact (same hit rank, different seconds) is the HNSW "
        "approximation: the engine's approximate top k swaps a few vectors, which changes the length of clips "
        "ranked before the hit.", "",
        "## Dev per query (specific: seconds / hit rank; category: AP@50)", "",
        rows_table(["query", "kind", "qtype", "modality", *(how for _, how in runs)],
                   [[q, r["kind"], r["qtype"], r["modality"], *(_cell(by_q[k][q]) for k in runs)]
                    for q, r in by_q[runs[0]].items()]), "",
        "## Tuning vs dev gap (engine, action weight 0)", ""]
    gap, diag = [], []
    for split in ("tuning", "dev"):
        idx = np_index if split == "dev" else Index(vecs, meta, videos[split], "pad", "motion",
                                                     a["max_crops_per_frame"], False)
        for r in rows[(split, "hnsw")]:
            if r["kind"] != "specific":
                continue
            q = next(x for x in load_queries(QUERIES_DIR, split) if x["query_id"] == r["query_id"])
            n_in, rank = gt_vector_rank(idx, clip.encode_text([q["query"]])[0], q)
            diag.append([r["query_id"], r["qtype"], r["modality"], q["camera_id"],
                         f"{(datetime.fromisoformat(q['end_time']) - datetime.fromisoformat(q['start_time'])).total_seconds():.0f}",
                         n_in, rank if rank is not None else "no vector", f"{len(idx.x):,}", _cell(r)])
        spec = [r for r in rows[(split, "hnsw")] if r["kind"] == "specific"]
        foot = sum(v.n_frames / v.fps for v in videos[split])
        ir_h = sum(v.n_frames / v.fps for v in videos[split] if v.modality == "ir") / 3600
        secs = sorted(r["seconds"] for r in spec)
        gap.append([split, f"{foot / 3600:.2f}", f"{ir_h:.2f}", f"{foot:,.0f}", len(spec),
                    sum(not r["hit"] for r in spec), dict(Counter(r["qtype"] for r in spec)),
                    dict(Counter(r["modality"] for r in spec)), f"{statistics.median(secs):,.0f}",
                    ", ".join(f"{x:,.0f}" for x in secs)])
    lines += [rows_table(["split", "footage h", "IR h", "miss cost floor s (= footage s)", "specific n", "misses",
                          "qtypes", "modality", "median s", "sorted seconds per query"], gap), "",
              "## Where the true event ranks among all vectors (specific queries; NumPy exact, same variant)", "",
              "Best rank of a vector on the ground-truth camera inside the ground-truth span, among all vectors of "
              f"the split. A rank above k = {scfg.k} means no vector of the event enters the hit list, so the "
              "query is a miss whatever the grouping does.", "",
              rows_table(["query", "qtype", "modality", "camera", "event s", "vectors in event",
                          "best event-vector rank", "split vectors", "seconds / hit rank (engine)"], diag), ""]
    args.report.write_text("\n".join(lines))
    print("\n".join(lines))
    print("report:", display_path(args.report))
    print("MATCH:", {f"{s}/{h}": v for (s, h), v in match.items()}, "DIFF:", {f"{s}/{h}": v for (s, h), v in diff.items()})


def gt_vector_rank(index: Index, z: np.ndarray, q: dict) -> tuple[int, int | None]:
    """(vectors inside the query's ground-truth span on its camera, best 1-based rank of one of them among all
    vectors of the index by cosine with z; None if the span has no vector)."""
    t0, t1 = datetime.fromisoformat(q["start_time"]), datetime.fromisoformat(q["end_time"])
    start = index.meta["video_id"].map(lambda i: index.by_id[i].start_time)
    wall = start + pd.to_timedelta(index.meta["t_s"], unit="s")
    m = ((index.meta["camera_id"] == q["camera_id"]) & (wall >= t0) & (wall < t1)).to_numpy()
    if not m.any():
        return 0, None
    s = index.x @ z.astype(np.float32)
    return int(m.sum()), int((s > s[m].max()).sum()) + 1


def _cell(r: dict) -> str:
    if r["kind"] == "specific":
        return f"{r['seconds']:.0f} / {r['hit_rank']}"
    return f"{r['AP@50']:.3f}"


if __name__ == "__main__":
    main()
