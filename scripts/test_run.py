"""Step 8.2: THE single test run (CLAUDE.md 1.10). Rewind and the four baselines on the TEST split, once.

Before anything is read from test:
  1. the working tree is clean (the experiments rows must name the code that ran);
  2. configs/config.yaml has the SHA-256 recorded at the freeze in configs/FROZEN_CONFIG.sha256 (step 8.1);
  3. every frozen query file matches eval/queries/HASHES (rewind.eval.queries.check_frozen);
  4. the run has never started: results/reports/test_run.started must not exist. It is written before the first
     test read and committed with the results. A run that crashed after writing it can only be repeated with
     --after-crash "<reason>"; the reason is printed in the report.
This is the only script allowed to read test (tests/test_frozen_queries.py TEST_ACCESS_ALLOWED).

Pre-declared secondary result (Ashutosh, PROGRESS.md, committed in 383bfca before any test read): the same config
with search.weights.action = SECONDARY_ACTION_WEIGHT, overridden in memory only (the config file and its frozen hash
are unchanged), run once in this same run, reported in its own section and labelled secondary. It never replaces
the primary (headline) result.

Rewind: the frozen config through the Phase 7 path (C++ engine, HNSW at ef_search, top k per branch over the test
footage -> z-score fusion of the branches with weight > 0 -> clips), scored by scripts/appearance_search
.query_rows_from. Baselines: scripts/run_baselines.run_videos (sequential, random, witness ±10 min, motion). The same
metric code scores every system. Event-rule precision/recall: not run (event branch dropped at Gate 6.2).

Outputs (all written before the experiments rows are logged):
  results/reports/test_results.md       summary with bootstrap CIs; headline metric vs each baseline (paired);
                                        breakdowns by qtype, specificity, and RGB vs IR (IR: recall-type metrics only,
                                        per label source); per query
  results/reports/test_rows.csv         per query x system
  results/reports/test_clips.json       per query: its frozen fields and Rewind's top clips (8.3 reads these: no
                                        new search, no query file)
  results/reports/test_event_ranks.csv  per specific query and active branch: vectors inside the true event and the
                                        best rank of one of them among all test vectors (exact search), for 8.3
  results/figures/test_headline.png, results/figures/test_recall.png

Rehearsal (before the run, any number of times): --rehearse --out <dir> runs the same code on the DEV split, writes
every output under <dir> (report, CSVs, JSON, figures), logs nothing and writes no marker. It exists so a bug cannot
crash the one test run half way.

Run from the repo root, once:  uv run python -m scripts.test_run
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from rewind.embed.clip_model import Clip
from rewind.embed.languagebind import LanguageBind
from rewind.eval.metrics import bootstrap_ci
from rewind.eval.queries import check_frozen, load_queries
from rewind.eval.splits import split_event_ids, split_video_ids
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT, display_path
from rewind.search.core import SearchConfig
from rewind.search.engine import Engine, EngineConfig, active_branches, branch_model, wall_seconds
from rewind.search.fusion import check_weights, fuse
from scripts.appearance_search import query_rows_from
from scripts.fusion_tune import FusedRanker
from scripts.run_baselines import QUERIES_DIR, SYSTEMS, category_truth_ids, run_videos, summarise
from scripts.select_footage import rows_table

REPORTS = REPO_ROOT / "results" / "reports"
FIGURES = REPO_ROOT / "results" / "figures"
STARTED = REPORTS / "test_run.started"
FROZEN = REPO_ROOT / "configs" / "FROZEN_CONFIG.sha256"
TOP_CLIPS = 10         # clips per query saved for the failure analysis (8.3)
SECONDARY = "rewind_secondary_w2"
SECONDARY_ACTION_WEIGHT = 2   # pre-declared in PROGRESS.md (383bfca): an analysis setting fixed before the run, not a
                              # tunable of the system; it must not go into config.yaml (that would change the frozen hash)
QUERY_FIELDS = ("query_id", "kind", "qtype", "specificity", "modality", "label_source", "query", "activity_type",
                "event_id", "camera_id", "clip", "start_time", "end_time", "start_s", "end_s")   # saved for 8.3


class RecordingRanker(FusedRanker):
    """FusedRanker that also keeps the top fused clips of each query (with branches and best hits) for 8.3."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.saved: dict[str, list] = {}

    def ranked(self, z, cfg, rgb_only, query=None):
        clips = fuse(self.hits(z, rgb_only, query["query"]), self.weights, self.durations, cfg)
        self.saved[query["query_id"]] = {
            "query": {k: query[k] for k in QUERY_FIELDS},
            "clips": [clip_record(i + 1, c, self.by_id[c.video_id]) for i, c in enumerate(clips[:TOP_CLIPS])]}
        return [c.span(self.by_id[c.video_id]) for c in clips]


def clip_record(rank: int, c, v) -> dict:
    s = c.span(v)
    best = {b: {k: _plain(h[k]) for k in ("t_s", "score", "z", "type", "crop_source", "class", "x1", "y1", "x2", "y2")
                if k in h} for b, h in c.best.items()}
    return {"rank": rank, "camera_id": v.camera_id, "video_id": v.video_id, "clip": v.clip,
            "start": s.start.isoformat(), "end": s.end.isoformat(), "start_s": c.start_s, "end_s": c.end_s,
            "first_hit_s": c.first_hit_s, "last_hit_s": c.last_hit_s, "score": c.score, "branches": c.branches,
            "best": best}


def _plain(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return None if isinstance(x, float) and np.isnan(x) else x


def preflight(stamp: dict, after_crash: str | None, rehearse: bool) -> None:
    if stamp["git_hash"].endswith("-dirty") and not rehearse:
        raise SystemExit("uncommitted changes: the test run must run on committed code")
    frozen = FROZEN.read_text().split()[0]
    if stamp["config_hash"] != frozen:
        raise SystemExit(f"config hash {stamp['config_hash'][:12]} != frozen {frozen[:12]} ({display_path(FROZEN)})")
    check_frozen(QUERIES_DIR)
    if STARTED.exists() and not after_crash and not rehearse:
        raise SystemExit(f"the test run already started once ({display_path(STARTED)}: "
                         f"{STARTED.read_text().strip()}). It is not repeated.")


def event_ranks(engine: Engine, ranker: FusedRanker, queries: list, z_app: dict, action_text: dict) -> list[dict]:
    """Per specific query and branch with weight > 0: vectors on the true camera inside the true span, and the best
    1-based rank of one of them among all test vectors of the branch (exact search, before grouping)."""
    out = []
    n_rows = {b: sum(engine.collection.get(n).size for n in engine.meta if n.startswith(b + "/"))
              for b in ranker.weights}
    for q in queries:
        if q["kind"] != "specific":
            continue
        t0 = wall_seconds(datetime.fromisoformat(q["start_time"]))
        t1 = wall_seconds(datetime.fromisoformat(q["end_time"]))
        for b, w in ranker.weights.items():
            if w == 0:
                continue
            z = z_app[q["query"]] if b == "appearance" else action_text[q["query"]]
            h = engine.search(b, z, k=n_rows[b], ranges=ranker.ranges, exact=True)
            m = ((h["camera_id"] == q["camera_id"]) & (h["wall_s"] >= t0) & (h["wall_s"] < t1)).to_numpy()
            out.append({"query_id": q["query_id"], "branch": b, "vectors_in_event": int(m.sum()),
                        "best_rank": int(np.flatnonzero(m)[0]) + 1 if m.any() else -1, "test_vectors": len(h)})
    return out


def fmt_ci(v: list, stat, cfg: dict, f: str) -> str:
    m = cfg["metrics"]
    if len(v) < 2:
        return f"{f.format(stat(v))} (n<2)"
    lo, hi = bootstrap_ci(v, stat, m["bootstrap_n"], m["bootstrap_seed"], m["ci_level"])
    return f"{f.format(stat(v))} [{f.format(lo)}, {f.format(hi)}]"


def report(df: pd.DataFrame, summ: dict, ranks: pd.DataFrame, cfg: dict, stamp: dict, meta: dict) -> str:
    m = cfg["metrics"]
    ks, ck = m["recall_k"], m["category_k"]
    ap = f"AP@{max(ck)}"
    systems = ["rewind", *SYSTEMS]
    spec, cat = df[df.kind == "specific"], df[df.kind == "category"]
    med, mean = statistics.median, statistics.mean
    small = lambda n: "descriptive only" if n < m["small_n"] else ""
    lines = [
        f"# Step 8.2 — the single test run{'' if meta['split'] == 'test' else ' — REHEARSAL ON ' + meta['split'].upper()}", "",
        f"Generated by `scripts/test_run.py` at git `{stamp['git_hash']}`, config `{stamp['config_hash']}` (= the frozen "
        f"hash in `configs/FROZEN_CONFIG.sha256`, checked before any test read). Query files checked against "
        f"`eval/queries/HASHES`. Started {meta['started']}." + (f" **Repeated after a crash: {meta['after_crash']}**"
                                                                 if meta["after_crash"] else ""), "",
        f"{meta['split'].capitalize()} footage: {meta['clips']} clips, {meta['hours']:.2f} h over {meta['cameras']} cameras; {meta['n_queries']} "
        f"queries ({len(spec[spec.system == 'rewind'])} specific, {len(cat[cat.system == 'rewind'])} category). Rewind: "
        f"weights {meta['weights']}, appearance {meta['appearance']}, HNSW ef {cfg['search']['ef_search']}, top "
        f"{cfg['search']['k_per_index']} per branch. Baselines: {cfg['baselines']['segment_s']} s segments, as at 2.3 "
        f"/ 3.2. CIs: {m['ci_level']:.0%} percentile bootstrap over queries ({m['bootstrap_n']} resamples, seed "
        f"{m['bootstrap_seed']}). Event-rule precision/recall: not applicable (event branch dropped at Gate 6.2).", "",
        "## Summary", "",
        rows_table(["system", "specific n", "median s watched [CI]", "mean s", "worst s",
                    *(f"R@{k} [CI]" for k in ks), *(f"category mAP@{k} [CI]" for k in ck)],
                   [[s, summ[s]["n_specific"], f"{summ[s]['median_seconds']:,.0f} [{summ[s]['median_seconds_ci_lo']:,.0f}, "
                     f"{summ[s]['median_seconds_ci_hi']:,.0f}]", f"{summ[s]['mean_seconds']:,.0f}",
                     f"{summ[s]['worst_seconds']:,.0f}",
                     *(f"{summ[s][f'R@{k}']:.3f} [{summ[s][f'R@{k}_ci_lo']:.3f}, {summ[s][f'R@{k}_ci_hi']:.3f}]" for k in ks),
                     *((f"{summ[s][f'mAP@{k}']:.3f} [{summ[s][f'mAP@{k}_ci_lo']:.3f}, {summ[s][f'mAP@{k}_ci_hi']:.3f}]"
                        if f"mAP@{k}" in summ[s] else "—") for k in ck)] for s in systems]), "",
        f"Rewind found {meta['hits']} of {summ['rewind']['n_specific']} specific events in its clip list (a miss costs "
        f"the list + a full review of the {meta['hours']:.2f} h). Baselines rank all footage, so they never miss.", "",
        "## Headline: seconds watched before the first correct clip, Rewind vs each baseline (paired, specific)", ""]
    lines += [paired_table(spec, "rewind", SYSTEMS, cfg), "",
              "## Specific queries by qtype (median seconds watched; Rewind R@10)", ""]
    for col in ("qtype", "specificity"):
        rows = []
        for g, x in spec.groupby(col):
            r = x[x.system == "rewind"]
            rows.append([g, len(r), int(r.hit.sum()), fmt_ci(r["R@10"].tolist(), mean, cfg, "{:.3f}"),
                         *(fmt_ci(x[x.system == s]["seconds"].tolist(), med, cfg, "{:,.0f}") for s in systems),
                         small(len(r))])
        lines += [rows_table([col, "n", "Rewind hits", "Rewind R@10 [CI]", *(f"{s} median s [CI]" for s in systems), ""],
                             rows), ""]
    lines += ["## Category queries by qtype (mAP, RGB footage and Kitware labels only)", ""]
    rows = []
    for g, x in cat.groupby("qtype"):
        r = x[x.system == "rewind"]
        rows.append([g, len(r), *(fmt_ci(x[x.system == s][ap].tolist(), mean, cfg, "{:.3f}")
                                  for s in systems if s != "witness"), small(len(r))])
    lines += [rows_table(["qtype", "n", *(f"{s} m{ap} [CI]" for s in systems if s != "witness"), ""], rows), "",
              "## RGB vs IR (specific). IR: recall-type metrics only; both per label source", "",
              "IR events carry third-party labels (incomplete, Gate 0.3), so IR is reported with Recall@k and hits "
              "only; RGB also with seconds watched.", ""]
    rows = []
    for (mod, src), x in spec.groupby(["modality", "label_source"]):
        r = x[x.system == "rewind"]
        rows.append([mod, src, len(r), int(r.hit.sum()), *(fmt_ci(r[f"R@{k}"].tolist(), mean, cfg, "{:.3f}") for k in ks),
                     *(f"{x[x.system == s]['R@10'].mean():.3f}" for s in SYSTEMS),
                     fmt_ci(r["seconds"].tolist(), med, cfg, "{:,.0f}") if mod == "rgb" else "— (IR: recall only)",
                     small(len(r))])
    lines += [rows_table(["modality", "label source", "n", "Rewind hits", *(f"Rewind R@{k} [CI]" for k in ks),
                          *(f"{s} R@10" for s in SYSTEMS), "Rewind median s [CI]", ""], rows), "",
              "## Per query (specific: seconds watched / Rewind hit rank; category: " + ap + ")", ""]
    piv = df.pivot_table(index="query_id", columns="system", values="cell", aggfunc="first")
    info = df[df.system == "rewind"].set_index("query_id")
    rk = ranks.pivot_table(index="query_id", columns="branch", values="best_rank", aggfunc="first") if len(ranks) else None
    rows = []
    for q in info.index:
        r = info.loc[q]
        best = "" if rk is None or q not in rk.index else ", ".join(f"{b} {int(v)}" for b, v in rk.loc[q].items())
        rows.append([q, r.kind, r.qtype, r.specificity, r.modality, r.label_source, r.camera_id or "",
                     *(piv.loc[q].get(s, "—") if isinstance(piv.loc[q].get(s), str) else "—"
                       for s in (*systems, SECONDARY)), best])
    lines += [rows_table(["query", "kind", "qtype", "specificity", "modality", "label source", "camera", *systems,
                          f"{SECONDARY} (secondary)", "best event-vector rank (exact)"], rows), ""]
    sec = summ[SECONDARY]
    hits2 = int(spec[spec.system == SECONDARY].hit.sum())
    lines += [f"## Pre-declared SECONDARY result — action weight {SECONDARY_ACTION_WEIGHT} (not the headline)", "",
              "Declared in PROGRESS.md (commit 383bfca) before any test read, run once in this same run: the frozen "
              f"config with only `search.weights.action` = {SECONDARY_ACTION_WEIGHT} (in memory; the config file and its "
              "hash are unchanged). **It cannot replace the primary result above, whatever the outcome.**", "",
              rows_table(["system", "specific n", "hits", "median s watched [CI]", "mean s",
                          *(f"R@{k} [CI]" for k in ks), *(f"category mAP@{k} [CI]" for k in ck)],
                         [[s, summ[s]["n_specific"], int(spec[spec.system == s].hit.sum()),
                           f"{summ[s]['median_seconds']:,.0f} [{summ[s]['median_seconds_ci_lo']:,.0f}, "
                           f"{summ[s]['median_seconds_ci_hi']:,.0f}]", f"{summ[s]['mean_seconds']:,.0f}",
                           *(f"{summ[s][f'R@{k}']:.3f} [{summ[s][f'R@{k}_ci_lo']:.3f}, {summ[s][f'R@{k}_ci_hi']:.3f}]"
                             for k in ks),
                           *(f"{summ[s][f'mAP@{k}']:.3f} [{summ[s][f'mAP@{k}_ci_lo']:.3f}, {summ[s][f'mAP@{k}_ci_hi']:.3f}]"
                             for k in ck)] for s in ("rewind", SECONDARY)]), "",
              f"Secondary found {hits2} of {sec['n_specific']} specific events.", "",
              "Secondary vs each baseline (paired, specific; seconds watched):", "",
              paired_table(spec, SECONDARY, SYSTEMS, cfg), "",
              "Secondary vs primary (paired, specific; informational, no decision):", "",
              paired_table(spec, SECONDARY, ["rewind"], cfg), ""]
    return "\n".join(lines)


def paired_table(spec: pd.DataFrame, system: str, others: list, cfg: dict) -> str:
    """Seconds watched of `system` vs each of `others`, paired by query (positive saving = `system` faster)."""
    med = statistics.median
    a = spec[spec.system == system].set_index("query_id")["seconds"]
    rows = []
    for s in others:
        b = spec[spec.system == s].set_index("query_id")["seconds"].reindex(a.index)
        d = (b - a).tolist()
        rows.append([s, f"{med(a):,.0f}", f"{med(b):,.0f}", sum(x > 0 for x in d), sum(x == 0 for x in d),
                     sum(x < 0 for x in d), fmt_ci(d, med, cfg, "{:+,.0f}"), f"{med(b) / med(a):.1f}x"])
    return rows_table(["vs", f"{system} median s", "other median s", f"{system} faster", "ties", f"{system} slower",
                       "median paired saving s [CI]", "ratio of medians"], rows)


def hex_bgr(h: str) -> tuple:
    return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))


SURFACE, INK, MUTE, GRID = hex_bgr("#fcfcfb"), hex_bgr("#0b0b0b"), hex_bgr("#52514e"), hex_bgr("#e4e3df")
SERIES = [hex_bgr(h) for h in ("#2a78d6", "#eb6834", "#1baf7a")]     # reference categorical slots 1-3


def put(img, s, xy, c=INK, sc=0.5):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, c, 1, cv2.LINE_AA)


def plot_headline(summ: dict, path: Path, title: str) -> None:
    """Median seconds watched per system with its CI, log x axis; one row per system, values labelled."""
    systems = ["rewind", *SYSTEMS]
    W, ml, mr, mt, row = 1000, 130, 150, 70, 56
    H = mt + row * len(systems) + 60
    img = np.full((H, W, 3), SURFACE, np.uint8)
    vals = [summ[s][k] for s in systems for k in ("median_seconds_ci_lo", "median_seconds_ci_hi", "median_seconds")]
    lx0, lx1 = np.floor(np.log10(max(1.0, min(vals)))), np.ceil(np.log10(max(vals)))
    X = lambda v: int(ml + (np.log10(max(v, 10 ** lx0)) - lx0) / (lx1 - lx0) * (W - ml - mr))
    put(img, title, (ml, 30), INK, 0.6)
    put(img, "median seconds watched before the first correct clip (log scale; bar = 95% CI)", (ml, 52), MUTE, 0.45)
    for e in range(int(lx0), int(lx1) + 1):
        x = X(10 ** e)
        cv2.line(img, (x, mt), (x, H - 50), GRID, 1)
        put(img, f"{10 ** e:,}", (x - 12, H - 30), MUTE, 0.45)
    for i, s in enumerate(systems):
        y = mt + row * i + row // 2
        c = SERIES[0] if s == "rewind" else MUTE
        put(img, s, (16, y + 5), INK, 0.55)
        cv2.line(img, (X(summ[s]["median_seconds_ci_lo"]), y), (X(summ[s]["median_seconds_ci_hi"]), y), c, 2, cv2.LINE_AA)
        p = (X(summ[s]["median_seconds"]), y)
        cv2.circle(img, p, 7, SURFACE, -1, cv2.LINE_AA)
        cv2.circle(img, p, 5, c, -1, cv2.LINE_AA)
        put(img, f"{summ[s]['median_seconds']:,.0f} s", (X(summ[s]["median_seconds_ci_hi"]) + 10, y + 5), INK, 0.45)
    cv2.imwrite(str(path), img)


def plot_recall(summ: dict, ks: list, path: Path, title: str) -> None:
    """Recall@k of specific queries per system: grouped horizontal bars, one colour per k, legend + value labels."""
    systems = ["rewind", *SYSTEMS]
    W, ml, mr, mt, bar, gap = 1000, 130, 90, 90, 14, 2
    grp = len(ks) * (bar + gap) + 22
    H = mt + grp * len(systems) + 50
    img = np.full((H, W, 3), SURFACE, np.uint8)
    X = lambda v: int(ml + v * (W - ml - mr))
    put(img, title, (ml, 30), INK, 0.6)
    for j, k in enumerate(ks):
        lx = ml + j * 110
        cv2.rectangle(img, (lx, 50), (lx + 14, 64), SERIES[j], -1)
        put(img, f"R@{k}", (lx + 20, 63), INK, 0.45)
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        cv2.line(img, (X(t), mt - 8), (X(t), H - 40), GRID, 1)
        put(img, f"{t:.2f}", (X(t) - 14, H - 20), MUTE, 0.45)
    for i, s in enumerate(systems):
        y0 = mt + grp * i
        put(img, s, (16, y0 + grp // 2), INK, 0.55)
        for j, k in enumerate(ks):
            v = summ[s][f"R@{k}"]
            y = y0 + j * (bar + gap)
            if v > 0:
                cv2.rectangle(img, (X(0), y), (max(X(v), X(0) + 2), y + bar), SERIES[j], -1)
            put(img, f"{v:.2f}", (max(X(v), X(0)) + 6, y + bar - 2), MUTE, 0.4)
    cv2.imwrite(str(path), img)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--after-crash", help="repeat a run that crashed after test_run.started was written: the reason")
    p.add_argument("--rehearse", action="store_true", help="run on DEV into --out; no marker, no logging")
    p.add_argument("--out", type=Path, help="--rehearse: output directory")
    args = p.parse_args()
    if args.rehearse and not args.out:
        raise SystemExit("--rehearse needs --out")
    split = "dev" if args.rehearse else "test"
    reports, figures = (args.out, args.out) if args.rehearse else (REPORTS, FIGURES)
    reports.mkdir(parents=True, exist_ok=True)
    stamp = run_stamp(args.config)
    preflight(stamp, args.after_crash, args.rehearse)
    cfg = yaml.safe_load(args.config.read_text())
    started = datetime.now().isoformat(timespec="seconds")
    if not args.rehearse:
        STARTED.write_text(f"{started} git {stamp['git_hash']} config {stamp['config_hash']}"
                           + (f" after crash: {args.after_crash}" if args.after_crash else "") + "\n")

    # --- the only test reads (dev when rehearsing) ---
    test = split == "test"
    conn = connect(args.db)
    ids = set(split_video_ids(conn, split, allow_test=test))
    keep = set(split_event_ids(conn, split, allow_test=test))
    every = set(split_event_ids(conn, split, allow_test=test, include_boundary=True))
    queries = load_queries(QUERIES_DIR, split, allow_test=test)
    videos = [v for v in load_videos(conn) if v.video_id in ids]
    truth = lambda t: category_truth_ids(conn, split, t, keep, every)

    branches = active_branches(cfg)
    weights = check_weights(cfg["search"]["weights"], branches)
    a = cfg["appearance"]
    scfg = SearchConfig.from_config(cfg)
    engine = Engine(EngineConfig.from_config(cfg), {b: branch_model(cfg, b) for b in branches}, branches)
    sec_weights = check_weights({**cfg["search"]["weights"], "action": SECONDARY_ACTION_WEIGHT}, branches)
    action_text = {}
    if weights.get("action", 0) > 0 or sec_weights.get("action", 0) > 0:
        lb = LanguageBind(REPO_ROOT / cfg["action"]["languagebind"]["weights_dir"])
        texts = [q["query"] for q in queries]
        action_text = dict(zip(texts, lb.encode_text(texts)))
        del lb
        torch.cuda.empty_cache()
    clip = Clip(a["model"], a["pretrained"])
    join = cfg["ingest"]["max_name_duration_diff_s"]
    ranker = RecordingRanker(engine, videos, weights, action_text, join, split)
    rewind_rows = query_rows_from(queries, truth, ranker, clip, cfg, scfg)
    sec_ranker = FusedRanker(engine, videos, sec_weights, action_text, join, split)
    sec_rows = query_rows_from(queries, truth, sec_ranker, clip, cfg, scfg)
    z_app = {q["query"]: clip.encode_text([q["query"]])[0] for q in queries}
    # event ranks for every branch searched by either system (the secondary searches both; ranges are the same)
    ranks = pd.DataFrame(event_ranks(engine, sec_ranker, queries, z_app, action_text))
    base_rows = run_videos(conn, split, videos, queries, truth, cfg, args.art)

    # --- assemble, report, then log ---
    qinfo = {q["query_id"]: q for q in queries}
    ap = f"AP@{max(cfg['metrics']['category_k'])}"
    rows = [r | {"system": "rewind"} for r in rewind_rows] + [r | {"system": SECONDARY} for r in sec_rows] + base_rows
    for r in rows:
        q = qinfo[r["query_id"]]
        r |= {"label_source": q["label_source"], "camera_id": q["camera_id"]}
        if r["kind"] == "specific":
            r["cell"] = f"{r['seconds']:,.0f}" + (f" / {r['hit_rank']}" if r["system"] in ("rewind", SECONDARY) else "")
        else:
            r["cell"] = f"{r[ap]:.3f}"
    df = pd.DataFrame(rows)
    summ = {s: summarise(g.to_dict("records"), cfg) for s, g in df.groupby("system")}
    spec_rw = df[(df.system == "rewind") & (df.kind == "specific")]
    meta = {"split": split, "started": started, "after_crash": args.after_crash, "clips": len(videos),
            "hours": sum(v.n_frames / v.fps for v in videos) / 3600, "cameras": len({v.camera_id for v in videos}),
            "n_queries": len(queries), "weights": {b: w for b, w in weights.items()},
            "appearance": f"{a['frame_mode']}/{a['crop_source']}/"
                          f"{a['center_groups'] + ' centered' if a['mean_center'] else 'raw'}",
            "hits": int(spec_rw.hit.sum())}
    df.drop(columns=["cell"]).to_csv(reports / "test_rows.csv", index=False)
    (reports / "test_clips.json").write_text(json.dumps(ranker.saved, indent=1))
    ranks.to_csv(reports / "test_event_ranks.csv", index=False)
    (reports / "test_results.md").write_text(report(df, summ, ranks, cfg, stamp, meta))
    name = split.capitalize() + " split"
    plot_headline(summ, figures / "test_headline.png", f"{name}, {summ['rewind']['n_specific']} specific queries")
    plot_recall(summ, cfg["metrics"]["recall_k"], figures / "test_recall.png",
                f"{name}: Recall@k of {summ['rewind']['n_specific']} specific queries")
    print((reports / "test_results.md").read_text())
    if args.rehearse:
        print("REHEARSAL on dev: nothing logged, no marker written")
        return
    for s, x in summ.items():
        if s in ("rewind", SECONDARY):
            hits = int(df[(df.system == s) & (df.kind == "specific")].hit.sum())
            w = meta["weights"] if s == "rewind" else dict(sec_weights)
            role = "PRIMARY rewind" if s == "rewind" else f"PRE-DECLARED SECONDARY {s} (cannot replace primary)"
            log_experiment(split, x | {"hits": hits}, stamp=stamp,
                           notes=f"step 8.2 THE test run: {role} (weights {w}, {meta['appearance']}, "
                                 f"ef {cfg['search']['ef_search']})")
        else:
            log_experiment(split, x, notes=f"step 8.2 THE test run: {s} baseline", stamp=stamp)


if __name__ == "__main__":
    main()
