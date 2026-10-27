"""Step 8.1: dev ablations and the two pre-registered comparisons (PROGRESS.md "Step 8.1 — operational plan",
committed in dde89b0 before any 8.1 result). DEV split only.

Every variant runs the Phase 7 path on the 17 dev queries: C++ engine (HNSW at the config's ef_search, top
search.k_per_index per branch, restricted to the dev footage) -> z-score fusion -> clips -> metrics
(scripts/appearance_search.query_rows, the code that scored the baselines and Gates 4.4 / 5.1 / 7.4).
Appearance variants other than the main indexes (per-camera-per-type centering, other crop sources) get their own
appearance HNSW indexes, built by scripts/build_indexes (same code, same HNSW settings) under
artifacts/indexes_8.1/<variant>/; the action indexes are the main ones.

Comparisons (each one change from the committed config pad/motion/raw, action weight 2; paired rule
rewind/eval/gate.py as written):
  A  action weight 0 (simple) vs 2 (complex), both raw
  B  raw (simple) vs per-camera-per-type centering (complex), both at action weight 2
Final = A's weight x B's centering. Descriptive only: the 4th cell (weight 0 x per-type), crop_source none / motion /
detection_then_motion at the final weight and centering, HNSW vs exact brute force at the final config (metrics and
per-query search latency). Breakdowns per variant by kind x qtype and by modality, bootstrap CIs; n < 5 cells are
marked "descriptive only".

Writes results/reports/ablation_8.1.md, results/reports/ablation_8.1_rows.csv (per query, per variant) and one
experiments row per variant. Changes no config: the freeze is a separate, reviewed edit.

Run from the repo root (~10 min, index builds included): uv run python -m scripts.ablation_8
"""

import argparse
import copy
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from rewind.embed.clip_model import Clip
from rewind.embed.languagebind import LanguageBind
from rewind.eval.gate import paired_decision
from rewind.eval.metrics import bootstrap_ci
from rewind.eval.queries import load_queries
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT, display_path
from rewind.search.core import SearchConfig
from rewind.search.engine import (Engine, EngineConfig, active_branches, branch_model, engine_module,
                                  index_path)
from rewind.search.fusion import check_weights
from scripts.appearance_search import query_rows, split_videos
from scripts.build_indexes import appearance_rows, build_one, with_wall_seconds
from scripts.fusion_tune import FusedRanker
from scripts.run_baselines import QUERIES_DIR, summarise
from scripts.select_footage import rows_table

SPLIT = "dev"
COMMITTED = {"frame_mode": "pad", "crop_source": "motion", "mean_center": False, "action_weight": 2}
SMALL_N = 5          # cells with fewer queries are descriptive only (pre-registered)


def variant_cfg(cfg: dict, crop_source: str, center: str) -> dict:
    """Config with the appearance variant changed. center: raw | camera_type."""
    c = copy.deepcopy(cfg)
    c["appearance"]["crop_source"] = crop_source
    c["appearance"]["mean_center"] = center != "raw"
    c["appearance"]["center_groups"] = "camera" if center == "raw" else center
    return c


def appearance_engine(cfg: dict, vcfg: dict, art: Path, root: Path, main: Engine, stamp: dict) -> Engine:
    """The main engine if vcfg's appearance variant is the config's; otherwise an appearance-only engine over the
    variant's indexes under root (built first if any camera is missing)."""
    if branch_model(vcfg, "appearance") == branch_model(cfg, "appearance"):
        return main
    a = vcfg["appearance"]
    center = "raw" if not a["mean_center"] else a["center_groups"]
    ecfg = EngineConfig.from_config(vcfg)
    ecfg = EngineConfig(**{**ecfg.__dict__, "index_dir": root / f"{a['frame_mode']}_{a['crop_source']}_{center}"})
    model = branch_model(vcfg, "appearance")
    cams = sorted(main.cameras["appearance"])
    if any(not (index_path(ecfg.index_dir, "appearance", c) / "graph.hnsw").exists() for c in cams):
        t0 = time.perf_counter()
        x, meta = appearance_rows(vcfg, art)
        meta = with_wall_seconds(meta, load_videos(connect(REPO_ROOT / "data" / "rewind.sqlite")))
        mod = engine_module()
        for cam in cams:
            rows = np.flatnonzero(meta["camera_id"].to_numpy() == cam)
            build_one(mod, ecfg, "appearance", cam, model, x[rows], meta.iloc[rows], stamp)
        print(f"built {model}: {len(x):,} rows in {time.perf_counter() - t0:.0f} s -> {display_path(ecfg.index_dir)}",
              flush=True)
    return Engine(ecfg, {"appearance": model}, ("appearance",))


def cells(rows: list[dict], cfg: dict, ap_key: str) -> list[list]:
    """Breakdown rows: overall, specific by qtype, specific by modality, category by qtype."""
    m = cfg["metrics"]
    boot = lambda v, f: bootstrap_ci(v, f, m["bootstrap_n"], m["bootstrap_seed"], m["ci_level"]) if len(v) >= 2 else None
    ci = lambda x, f: "n<2" if x is None else f"[{f.format(x[0])}, {f.format(x[1])}]"
    spec = [r for r in rows if r["kind"] == "specific"]
    cat = [r for r in rows if r["kind"] == "category"]
    groups = [("specific", "all", spec)]
    groups += [("specific", f"qtype {q}", [r for r in spec if r["qtype"] == q]) for q in sorted({r["qtype"] for r in spec})]
    groups += [("specific", f"modality {q}", [r for r in spec if r["modality"] == q])
               for q in sorted({r["modality"] for r in spec})]
    groups += [("category", "all", cat)]
    groups += [("category", f"qtype {q}", [r for r in cat if r["qtype"] == q]) for q in sorted({r["qtype"] for r in cat})]
    out = []
    for kind, name, g in groups:
        if not g:
            continue
        note = "descriptive only" if len(g) < SMALL_N else ""
        if kind == "specific":
            secs, r10 = [r["seconds"] for r in g], [r["R@10"] for r in g]
            out.append([kind, name, len(g), sum(r["hit"] for r in g),
                        f"{statistics.median(secs):,.0f} {ci(boot(secs, statistics.median), '{:,.0f}')}",
                        f"{statistics.mean(r10):.3f} {ci(boot(r10, statistics.mean), '{:.3f}')}", "", note])
        else:
            ap = [r[ap_key] for r in g]
            out.append([kind, name, len(g), "", "", "",
                        f"{statistics.mean(ap):.3f} {ci(boot(ap, statistics.mean), '{:.3f}')}", note])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--variant-indexes", type=Path, default=REPO_ROOT / "artifacts" / "indexes_8.1")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "ablation_8.1.md")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    stamp = run_stamp(args.config)
    a = cfg["appearance"]
    now = {"frame_mode": a["frame_mode"], "crop_source": a["crop_source"], "mean_center": a["mean_center"],
           "action_weight": cfg["search"]["weights"]["action"]}
    if now != COMMITTED:
        raise ValueError(f"step 8.1 starts from the committed pre-8.1 config {COMMITTED}; the config has {now}")
    branches = active_branches(cfg)
    if branches != ("appearance", "action"):
        raise ValueError(f"step 8.1 compares action weights; active branches are {branches}")
    m = cfg["metrics"]
    ap_key = f"AP@{max(m['category_k'])}"
    scfg = SearchConfig.from_config(cfg)
    conn = connect(args.db)
    videos = split_videos(conn, SPLIT)
    queries = load_queries(QUERIES_DIR, SPLIT)
    join = cfg["ingest"]["max_name_duration_diff_s"]
    main_engine = Engine(EngineConfig.from_config(cfg), {b: branch_model(cfg, b) for b in branches}, branches)
    lb = LanguageBind(REPO_ROOT / cfg["action"]["languagebind"]["weights_dir"])
    texts = [q["query"] for q in queries]
    action_text = dict(zip(texts, lb.encode_text(texts)))
    del lb
    torch.cuda.empty_cache()
    clip = Clip(a["model"], a["pretrained"])

    rows, notes = {}, {}

    def run(name: str, crop_source: str, center: str, weight: float, exact: bool = False) -> list[dict]:
        vcfg = variant_cfg(cfg, crop_source, center)
        eng = appearance_engine(cfg, vcfg, args.art, args.variant_indexes, main_engine, stamp)
        ranker = FusedRanker(main_engine, videos, check_weights({"appearance": 1.0, "action": weight}, branches),
                             action_text, join, SPLIT, exact=exact, engines={"appearance": eng})
        rows[name] = query_rows(conn, SPLIT, ranker, clip, cfg, scfg)
        notes[name] = (f"step 8.1 dev: {a['frame_mode']}/{crop_source}/{center}, action weight {weight}, "
                       f"{'exact brute force' if exact else 'HNSW ef ' + str(cfg['search']['ef_search'])}")
        s = summarise(rows[name], cfg)
        print(f"{name}: median s {s['median_seconds']:.0f}, hits {sum(r['hit'] for r in rows[name] if r['kind'] == 'specific')}"
              f"/12, R@10 {s['R@10']:.3f}, mAP@50 {s['mAP@50']:.3f}", flush=True)
        return rows[name]

    gate = lambda s, c: paired_decision(rows[s], rows[c], ap_key, m["bootstrap_n"], m["bootstrap_seed"], m["ci_level"])
    run("w0 raw", "motion", "raw", 0)
    run("w2 raw", "motion", "raw", 2)
    run("w2 camera_type", "motion", "camera_type", 2)
    run("w0 camera_type", "motion", "camera_type", 0)
    dec_a = gate("w0 raw", "w2 raw")
    dec_b = gate("w2 raw", "w2 camera_type")
    w_final = 2 if dec_a["adopt"] else 0
    c_final = "camera_type" if dec_b["adopt"] else "raw"
    final = f"w{w_final} {c_final}"
    for cs in ("none", "detection_then_motion"):
        run(f"{final} crops {cs}", cs, c_final, w_final)
    run(f"{final} exact", "motion", c_final, w_final, exact=True)

    # search latency, HNSW vs exact, final config: engine search of every active branch per query (text encoding
    # excluded), one warm-up pass, then the median of 3 passes per query
    vcfg = variant_cfg(cfg, "motion", c_final)
    eng = appearance_engine(cfg, vcfg, args.art, args.variant_indexes, main_engine, stamp)
    lat = {}
    for exact in (False, True):
        r = FusedRanker(main_engine, videos, check_weights({"appearance": 1.0, "action": w_final}, branches),
                        action_text, join, SPLIT, exact=exact, engines={"appearance": eng})
        zs = {q["query"]: clip.encode_text([q["query"]])[0] for q in queries}
        per_q = {q["query_id"]: [] for q in queries}
        for rep in range(4):
            for q in queries:
                t = time.perf_counter()
                r.hits(zs[q["query"]], q["kind"] == "category", q["query"])
                if rep:
                    per_q[q["query_id"]].append((time.perf_counter() - t) * 1000)
        ms = [statistics.median(v) for v in per_q.values()]
        lat["exact" if exact else "hnsw"] = (statistics.median(ms), float(np.percentile(ms, 95)))

    summ = {k: summarise(v, cfg) for k, v in rows.items()}
    if not args.no_log:
        for k in rows:
            log_experiment(SPLIT, summ[k] | {"hits": sum(r["hit"] for r in rows[k] if r["kind"] == "specific")},
                           notes=notes[k], stamp=stamp)
    pd.DataFrame([r | {"variant": k} for k, v in rows.items() for r in v]).to_csv(
        args.report.with_name("ablation_8.1_rows.csv"), index=False)

    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    fam = lambda d, key, f: (lambda x: "—" if x is None else
                             f"{f.format(x['mean_diff'])} " + ("n<2" if x["ci"] is None else
                                                              f"[{f.format(x['ci'][0])}, {f.format(x['ci'][1])}]")
                             + f" (n={x['n']})")(d["families"].get(key))
    lines = [
        "# Step 8.1 — dev ablations and the pre-registered comparisons", "",
        f"Generated by `scripts/ablation_8.py` at git `{stamp['git_hash']}`, config `{stamp['config_hash'][:12]}`. "
        f"Dev: {len(videos)} clips, {sum(v.n_frames / v.fps for v in videos) / 3600:.2f} h; {len(queries)} queries. "
        f"C++ engine (HNSW ef {cfg['search']['ef_search']} unless 'exact'), top {scfg.k} per branch -> z-score "
        f"fusion -> clips (merge {scfg.merge_gap_s} s, cap {scfg.max_clip_s} s, mean top-3, ±{scfg.context_s} s). "
        f"Frame mode {a['frame_mode']}. Plan pre-registered in PROGRESS.md (commit dde89b0) before any 8.1 result.", "",
        "## Variants", "",
        rows_table(["variant", "median s [95% CI]", "mean s", "hits", "R@1", "R@5", "R@10", "mAP@10", "mAP@50"],
                   [[k + (" **(final)**" if k == final else ""),
                     f"{summ[k]['median_seconds']:,.0f} [{summ[k]['median_seconds_ci_lo']:,.0f}, "
                     f"{summ[k]['median_seconds_ci_hi']:,.0f}]", f"{summ[k]['mean_seconds']:,.0f}",
                     f"{sum(r['hit'] for r in rows[k] if r['kind'] == 'specific')}/{summ[k]['n_specific']}",
                     *(fmt(summ[k][f'R@{x}']) for x in m["recall_k"]), fmt(summ[k]["mAP@10"]), fmt(summ[k]["mAP@50"])]
                    for k in rows]), "",
        f"## Pre-registered comparisons (paired rule `rewind/eval/gate.py`; specific: seconds, category: {ap_key})", "",
        rows_table(["comparison", "simple", "complex", "W / T / L", "specific mean diff s [CI]",
                    f"category mean diff {ap_key} [CI]", "mean better", "adopt complex"],
                   [[n, s, c, f"{d['wins']} / {d['ties']} / {d['losses']}", fam(d, "specific_seconds", "{:+,.0f}"),
                     fam(d, "category_" + ap_key, "{:+.3f}"), d["mean_better"], d["adopt"]]
                    for n, s, c, d in (("A: action weight", "w0 raw", "w2 raw", dec_a),
                                       ("B: centering", "w2 raw", "w2 camera_type", dec_b))]), "",
        f"**Final: action weight {w_final}, centering {c_final}** (variant `{final}`)."
        + ("" if final in ("w2 raw", "w0 raw", "w2 camera_type") else
           " This cell was not one of the two compared pairs (A at raw, B at weight 2): flagged for Ashutosh."), "",
        "## HNSW vs exact brute force, final config (descriptive)", "",
        f"Metrics: rows `{final}` and `{final} exact` above. Engine search time per query (all active branches, "
        "dev footage ranges, text encoding excluded; median of 3 passes after a warm-up):", "",
        rows_table(["search", "median ms", "p95 ms"], [[k, f"{v[0]:.1f}", f"{v[1]:.1f}"] for k, v in lat.items()]), "",
        "## Breakdowns per variant (bootstrap 95% CI over queries; n < 5: descriptive only)", ""]
    for k in rows:
        lines += [f"### {k}", "", rows_table(["kind", "group", "n", "hits", "median s [CI]", "R@10 [CI]",
                                              f"m{ap_key} [CI]", ""], cells(rows[k], cfg, ap_key)), ""]
    lines += ["## Per query (specific: seconds / hit rank; category: AP@50)", "",
              rows_table(["query", "kind", "qtype", "modality", *rows],
                         [[r["query_id"], r["kind"], r["qtype"], r["modality"],
                           *(_cell(next(x for x in rows[k] if x["query_id"] == r["query_id"])) for k in rows)]
                          for r in rows["w0 raw"]]), ""]
    args.report.write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    print("report:", display_path(args.report))
    print(f"DECISION: A adopt w2 = {dec_a['adopt']}, B adopt camera_type = {dec_b['adopt']} -> {final}")


def _cell(r: dict) -> str:
    if r["kind"] == "specific":
        return f"{r['seconds']:.0f} / {r['hit_rank']}"
    return f"{r['AP@50']:.3f}"


if __name__ == "__main__":
    main()
