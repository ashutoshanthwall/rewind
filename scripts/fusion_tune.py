"""Step 7.4: tune the action-branch fusion weight on the TUNING split with the pre-registered rule.

Rule (PROGRESS.md "Pre-agreed decision rules", 26 Oct, written before any fusion result): choose the weight that
maximises mean Recall@10 on tuning specific-event queries; ties broken by lower median seconds watched; grid
{0, 0.25, 0.5, 1, 2} for each non-appearance branch, appearance fixed at 1.0. (Only action is a non-appearance branch:
events were dropped at Gate 6.2.) If a tie remains after both criteria, the rule does not decide: the script reports
NEEDS_USER and changes nothing.

Search per query: the C++ engine (rewind/search/engine.py, HNSW at the config's ef_search / filter settings),
restricted to the tuning footage with per-camera time ranges (camera_ranges of the tuning videos; every returned row is
checked to belong to a tuning video), top search.k_per_index per branch -> fusion (rewind/search/fusion.py) -> clips.
Text: CLIP (appearance) and LanguageBind (action) encoders of the config. Specific queries search all cameras,
category queries the RGB cameras (as steps 4.4 / 5.1). Metrics: scripts/appearance_search.query_rows (the code that
scored the baselines and Gates 4.4 / 5.1). Cross-check: weight 0 (appearance only) is also run with NumPy brute force
(scripts/appearance_search.Index), to see what HNSW changes.

Run from the repo root:  uv run python -m scripts.fusion_tune
"""

import argparse
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from rewind.embed.clip_model import Clip
from rewind.embed.languagebind import LanguageBind
from rewind.eval.queries import load_queries
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect
from rewind.paths import REPO_ROOT, display_path
from rewind.search.core import SearchConfig
from rewind.search.engine import Engine, EngineConfig, active_branches, branch_model, camera_ranges
from rewind.search.fusion import check_weights, fuse
from scripts.appearance_search import Index, query_rows, split_videos
from scripts.collect_appearance import index_dir
from scripts.run_baselines import QUERIES_DIR, summarise
from scripts.select_footage import rows_table

SPLIT = "tuning"
GRID = [0, 0.25, 0.5, 1, 2]


class FusedRanker:
    """ranked(z, cfg, rgb_only, query) for query_rows: engine search per active branch over the split -> fusion."""

    def __init__(self, engine: Engine, videos: list, weights: dict, action_text: dict, join_gap_s: float):
        self.engine, self.weights, self.action_text = engine, weights, action_text
        self.by_id = {v.video_id: v for v in videos}
        self.durations = {v.video_id: v.n_frames / v.fps for v in videos}
        self.footage_s = sum(self.durations.values())
        self.ranges = camera_ranges(videos, join_gap_s)
        self.rgb_ranges = camera_ranges([v for v in videos if v.modality == "rgb"], join_gap_s)

    def hits(self, z: np.ndarray, rgb_only: bool, text: str) -> dict:
        ranges = self.rgb_ranges if rgb_only else self.ranges
        out = {}
        for b, w in self.weights.items():
            if w == 0:
                continue
            q = z if b == "appearance" else self.action_text[text]
            h = self.engine.search(b, q, ranges=ranges)
            bad = set(h["video_id"]) - set(self.by_id)
            if bad:
                raise RuntimeError(f"{b}: search returned rows of videos outside the {SPLIT} split: {sorted(bad)[:5]}")
            out[b] = h
        return out

    def ranked(self, z: np.ndarray, cfg: SearchConfig, rgb_only: bool, query: dict | None = None) -> list:
        clips = fuse(self.hits(z, rgb_only, query["query"]), self.weights, self.durations, cfg)
        return [c.span(self.by_id[c.video_id]) for c in clips]


def decide(summ: dict) -> tuple[str, list]:
    """Pre-registered rule -> (decision, ranking). decision: the weight as text, or NEEDS_USER on a remaining tie."""
    key = lambda w: (-summ[w]["R@10"], summ[w]["median_seconds"])
    ranking = sorted(summ, key=key)
    best = [w for w in ranking if key(w) == key(ranking[0])]
    return (str(best[0]) if len(best) == 1 else "NEEDS_USER"), ranking


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "fusion_7.4.md")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    stamp = run_stamp(args.config)
    branches = active_branches(cfg)
    if branches != ("appearance", "action"):
        raise ValueError(f"step 7.4 tunes the action weight; active branches are {branches}")
    a = cfg["appearance"]
    scfg = SearchConfig.from_config(cfg)
    conn = connect(args.db)
    videos = split_videos(conn, SPLIT)
    queries = load_queries(QUERIES_DIR, SPLIT)
    engine = Engine(EngineConfig.from_config(cfg), {b: branch_model(cfg, b) for b in branches}, branches)
    lb = LanguageBind(REPO_ROOT / cfg["action"]["languagebind"]["weights_dir"])
    texts = [q["query"] for q in queries]
    action_text = dict(zip(texts, lb.encode_text(texts)))
    del lb
    torch.cuda.empty_cache()
    clip = Clip(a["model"], a["pretrained"])

    rows, summ = {}, {}
    for w in GRID:
        weights = check_weights({"appearance": 1.0, "action": w}, branches)
        ranker = FusedRanker(engine, videos, weights, action_text, cfg["ingest"]["max_name_duration_diff_s"])
        rows[w] = query_rows(conn, SPLIT, ranker, clip, cfg, scfg)
        summ[w] = summarise(rows[w], cfg)
        print(f"action weight {w}: specific R@10 {summ[w]['R@10']:.3f}, median s {summ[w]['median_seconds']:.0f}, "
              f"category mAP@10 {summ[w]['mAP@10']:.3f}", flush=True)
    # cross-check: appearance only with NumPy brute force (steps 4.4 code path) on the same tuning footage
    d = index_dir(args.art, a["model"], a["pretrained"])
    np_index = Index(np.load(d / "vectors.npy"), pd.read_parquet(d / "meta.parquet"), videos, a["frame_mode"],
                     a["crop_source"], a["max_crops_per_frame"], a["mean_center"])
    np_rows = query_rows(conn, SPLIT, np_index, clip, cfg, scfg)
    np_summ = summarise(np_rows, cfg)

    decision, ranking = decide(summ)
    if not args.no_log:
        for w in GRID:
            log_experiment(SPLIT, summ[w], notes=f"step 7.4 fusion sweep: appearance 1.0, action {w}; C++ engine HNSW "
                                                 f"ef {cfg['search']['ef_search']}", stamp=stamp)
        log_experiment(SPLIT, np_summ, notes="step 7.4 cross-check: appearance only, NumPy brute force", stamp=stamp)

    fmt = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (
        f"{x:.3f}" if isinstance(x, float) else str(x))
    ks = cfg["metrics"]["recall_k"]
    spec = [q for q in queries if q["kind"] == "specific"]
    lines = [
        "# Step 7.4 — fusion weight for the action branch (tuning split)", "",
        f"Generated by `scripts/fusion_tune.py` at git `{stamp['git_hash']}`, config `{stamp['config_hash'][:12]}`. "
        f"Tuning: {len(videos)} clips, {sum(v.n_frames / v.fps for v in videos) / 3600:.2f} h; {len(queries)} queries "
        f"({len(spec)} specific, {len(queries) - len(spec)} category). C++ engine (HNSW, ef_search "
        f"{cfg['search']['ef_search']}), top {scfg.k} per branch -> z-scores -> weighted fusion -> clips (merge "
        f"{scfg.merge_gap_s} s, cap {scfg.max_clip_s} s, mean top-3, ±{scfg.context_s} s).", "",
        "**Pre-registered rule** (PROGRESS.md, 26 Oct): maximise mean Recall@10 on tuning specific-event queries; ties "
        "-> lower median seconds watched; grid {0, 0.25, 0.5, 1, 2}; appearance fixed at 1.0.", "",
        rows_table(["action weight", *(f"R@{k}" for k in ks), "median s", "mean s", "category mAP@10", "mAP@50"],
                   [[w, *(fmt(summ[w][f"R@{k}"]) for k in ks), fmt(summ[w]["median_seconds"]),
                     fmt(summ[w]["mean_seconds"]), fmt(summ[w]["mAP@10"]), fmt(summ[w]["mAP@50"])] for w in GRID]), "",
        f"Cross-check, appearance only with NumPy brute force (step 4.4 code): R@10 {fmt(np_summ['R@10'])}, median s "
        f"{fmt(np_summ['median_seconds'])}, category mAP@10 {fmt(np_summ['mAP@10'])} (engine, weight 0: R@10 "
        f"{fmt(summ[0]['R@10'])}, median s {fmt(summ[0]['median_seconds'])}, mAP@10 {fmt(summ[0]['mAP@10'])}).", "",
        f"**Decision by the rule: {'action weight = ' + decision if decision != 'NEEDS_USER' else 'NEEDS_USER (tie remains)'}**"
        f" (ranking by the rule: {', '.join(str(w) for w in ranking)}). n = {len(spec)} specific queries: R@10 moves in "
        f"steps of {1 / len(spec):.3f}; this choice is noisy.", "",
        "## Per query (specific: seconds watched / hit rank; category: AP@10)", "",
        rows_table(["query", "kind", "qtype", "modality", *(f"w={w}" for w in GRID), "NumPy w=0"],
                   [[r["query_id"], r["kind"], r["qtype"], r["modality"],
                     *(_cell(next(x for x in rows[w] if x["query_id"] == r["query_id"])) for w in GRID),
                     _cell(next(x for x in np_rows if x["query_id"] == r["query_id"]))] for r in rows[GRID[0]]]), ""]
    args.report.write_text("\n".join(lines))
    print("\n".join(lines))
    print("report:", display_path(args.report))
    print("DECISION:", decision)


def _cell(r: dict) -> str:
    if r["kind"] == "specific":
        return f"{r['seconds']:.0f} / {r['hit_rank']}"
    return f"{r['AP@10']:.3f}"


if __name__ == "__main__":
    main()
