"""Appearance search with NumPy brute force (steps 4.2 tag choice and 4.4 ablations). Tuning and dev only.

text -> CLIP text embedding -> cosine with every vector of the variant -> top search.k_per_index -> clips
(rewind/search/core.py) -> metrics (rewind/eval/metrics.py, through scripts/run_baselines helpers, so Rewind and
the baselines are scored by the same code). A variant = frame_mode (tiles2 | pad) x crop_source (none | motion |
detection_then_motion) x mean-centering (per camera, over the variant's own vectors; step 4.3).
Category queries search RGB vectors only (category ground truth is Kitware RGB, IR labels are incomplete: Gate 0.3);
specific queries search all cameras of the split.

Mode --tag-check (step 4.2, PROGRESS open-issue item 5, tuning split): for each candidate (model, tag), templated
prompts from the activity names ("a " + name with spaces, e.g. "a person opens vehicle door") for every Kitware type
with >= queries.min_category_instances tuning activities -> AP@k per type; the 9 tuning queries as a secondary check.
Main variant of the config (frame_mode, crop_source, mean_center). Decision rule (fixed before the first run): the
tag with the highest mean AP@max(metrics.category_k) over types, if it also wins on more types than it loses against
each other tag (paired per type); otherwise a near-tie (AUTOPILOT.md 3: ask the user).

Run from the repo root:
  uv run python -m scripts.appearance_search --tag-check
"""

import argparse
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from rewind.embed.clip_model import Clip
from rewind.embed.store import load_appearance, variant_mask
from rewind.eval.metrics import Span, bootstrap_ci
from rewind.eval.queries import load_queries
from rewind.eval.splits import split_video_ids
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT, display_path
from rewind.search.core import SearchConfig, group_hits, mean_center, top_k
from scripts.run_baselines import QUERIES_DIR, category_metrics, category_truth, specific_metrics
from scripts.select_footage import rows_table

SPLITS_RUN = ("tuning", "dev")


class Index:
    """The vectors of one variant over one split's videos, ready for brute-force search."""

    def __init__(self, vecs: np.ndarray, meta: pd.DataFrame, videos: list, frame_mode: str, crop_source: str,
                 max_crops: int, center: bool):
        keep = variant_mask(meta, frame_mode, crop_source, max_crops)
        self.meta = meta[keep].reset_index(drop=True)
        x = vecs[keep]
        self.x = mean_center(x, self.meta["camera_id"].to_numpy()) if center else x.astype(np.float32)
        self.by_id = {v.video_id: v for v in videos}
        self.durations = {v.video_id: v.n_frames / v.fps for v in videos}
        self.rgb = (self.meta["modality"] == "rgb").to_numpy()
        self.footage_s = sum(self.durations.values())

    def ranked(self, q: np.ndarray, cfg: SearchConfig, rgb_only: bool) -> list[Span]:
        rows = np.flatnonzero(self.rgb) if rgb_only else np.arange(len(self.x))
        idx, sc = top_k(self.x[rows], q, cfg.k)
        idx = rows[idx]
        clips = group_hits(idx, sc, self.meta["video_id"].to_numpy(), self.meta["t_s"].to_numpy(), self.durations, cfg)
        return [c.span(self.by_id[c.video_id]) for c in clips]


def split_videos(conn, split: str) -> list:
    if split not in SPLITS_RUN:
        raise PermissionError(f"appearance_search runs on {SPLITS_RUN} only (test is read once, at step 8.2)")
    ids = set(split_video_ids(conn, split))
    return [v for v in load_videos(conn) if v.video_id in ids]


def query_rows(conn, split: str, index: Index, clip: Clip, cfg: dict, scfg: SearchConfig) -> list[dict]:
    """The frozen queries of a split, scored like the baselines."""
    m = cfg["metrics"]
    out = []
    for q in load_queries(QUERIES_DIR, split):
        base = {k: q[k] for k in ("query_id", "kind", "qtype", "specificity", "modality", "camera_id")}
        z = clip.encode_text([q["query"]])[0]
        if q["kind"] == "specific":
            gt = Span(q["camera_id"], datetime.fromisoformat(q["start_time"]), datetime.fromisoformat(q["end_time"]))
            out.append(base | specific_metrics([index.ranked(z, scfg, rgb_only=False)], gt, index.footage_s,
                                               m["recall_k"]))
        else:
            inst, dont_care = category_truth(conn, split, q["activity_type"])
            out.append(base | {"instances": len(inst)} | category_metrics([index.ranked(z, scfg, rgb_only=True)],
                                                                         inst, dont_care, m["category_k"]))
    return out


def template_types(conn, split: str, min_instances: int) -> dict[str, tuple]:
    """Kitware activity types with >= min_instances activities in the split -> (instances, don't-care)."""
    types = [r[0] for r in conn.execute("SELECT DISTINCT activity_type FROM events WHERE label_source != 'contrib' "
                                        "ORDER BY activity_type")]
    out = {}
    for t in types:
        try:
            inst, dc = category_truth(conn, split, t)
        except ValueError:          # no instance of this type in the split
            continue
        if len(inst) >= min_instances:
            out[t] = (inst, dc)
    return out


def prompt(activity_type: str) -> str:
    return "a " + activity_type.replace("_", " ")


def tag_check(args, cfg, stamp) -> None:
    a, m = cfg["appearance"], cfg["metrics"]
    scfg = SearchConfig.from_config(cfg)
    kmax = max(m["category_k"])
    conn = connect(args.db)
    videos = split_videos(conn, "tuning")
    types = template_types(conn, "tuning", cfg["queries"]["min_category_instances"])
    per_type, summary, qrows = {}, {}, {}
    for spec in args.models:
        model, tag = spec.split(":")
        vecs, meta = load_appearance(args.art, model, tag, videos)
        clip = Clip(model, tag)
        for center in (True, False):
            index = Index(vecs, meta, videos, a["frame_mode"], a["crop_source"], a["max_crops_per_frame"], center)
            z = clip.encode_text([prompt(t) for t in types])
            ap = {t: category_metrics([index.ranked(z[i], scfg, rgb_only=True)], *types[t], m["category_k"])
                  for i, t in enumerate(types)}
            per_type[(spec, center)] = ap
            rows = query_rows(conn, "tuning", index, clip, cfg, scfg)
            qrows[(spec, center)] = rows
            spec_rows = [r for r in rows if r["kind"] == "specific"]
            summary[(spec, center)] = {
                f"type_mAP@{k}": statistics.mean(v[f"AP@{k}"] for v in ap.values()) for k in m["category_k"]} | {
                "queries_median_seconds": statistics.median(r["seconds"] for r in spec_rows),
                **{f"queries_R@{k}": statistics.mean(r[f"R@{k}"] for r in spec_rows) for k in m["recall_k"]},
                "queries_cat_mAP@10": statistics.mean(r["AP@10"] for r in rows if r["kind"] == "category"),
                "vectors": len(index.x)}
        del clip
        torch.cuda.empty_cache()

    specs = args.models
    key = f"AP@{kmax}"
    main = {s: summary[(s, a["mean_center"])][f"type_mAP@{kmax}"] for s in specs}
    best = max(specs, key=lambda s: main[s])
    paired = {}
    for other in specs:
        if other == best:
            continue
        d = [per_type[(best, a["mean_center"])][t][key] - per_type[(other, a["mean_center"])][t][key] for t in types]
        lo, hi = bootstrap_ci(d, statistics.mean, m["bootstrap_n"], m["bootstrap_seed"], m["ci_level"])
        paired[other] = {"wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d),
                         "losses": sum(x < 0 for x in d), "mean_diff": statistics.mean(d), "ci": (lo, hi)}
    clear = all(p["wins"] > p["losses"] for p in paired.values())

    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    lines = [
        "# Step 4.2 — pretrained tag check (tuning)", "",
        f"Generated by `scripts/appearance_search.py --tag-check` at git `{stamp['git_hash']}`, config "
        f"`{stamp['config_hash'][:12]}`. {len(videos)} tuning clips. Variant: frame_mode {a['frame_mode']}, crop_source "
        f"{a['crop_source']}, max {a['max_crops_per_frame']} crops; decision on mean_center={a['mean_center']} "
        f"(the other setting shown for information). Search: top {scfg.k} vectors -> clips (merge "
        f"{scfg.merge_gap_s} s, cap {scfg.max_clip_s} s, mean top-3, ±{scfg.context_s} s).",
        f"Templated prompts ('a ' + activity name with spaces) for the {len(types)} Kitware types with >= "
        f"{cfg['queries']['min_category_instances']} tuning activities; RGB vectors only.", "",
        "## Summary", "",
        rows_table(["model:tag", "centered", "vectors", *(f"type mAP@{k}" for k in m["category_k"]),
                    "9 queries: median s (6 specific)", *(f"R@{k}" for k in m["recall_k"]), "category mAP@10 (3)"],
                   [[s, c, summary[(s, c)]["vectors"],
                     *(fmt(summary[(s, c)][f"type_mAP@{k}"]) for k in m["category_k"]),
                     fmt(summary[(s, c)]["queries_median_seconds"]),
                     *(fmt(summary[(s, c)][f"queries_R@{k}"]) for k in m["recall_k"]),
                     fmt(summary[(s, c)]["queries_cat_mAP@10"])] for s in specs for c in (True, False)]), "",
        f"## Decision (rule in the script docstring; {key}, mean_center={a['mean_center']})", "",
        f"Best mean: **{best}** ({main[best]:.3f}). Paired per type against each other tag:", "",
        rows_table(["vs", "wins", "ties", "losses", "mean diff", f"{m['ci_level']:.0%} CI"],
                   [[o, p["wins"], p["ties"], p["losses"], f"{p['mean_diff']:+.3f}",
                     f"[{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]"] for o, p in paired.items()]), "",
        f"**{'Clear: adopt ' + best if clear else 'Near-tie: the rule does not decide (NEEDS_USER)'}**", "",
        f"## {key} per type (mean_center={a['mean_center']})", "",
        rows_table(["type", "instances", *specs],
                   [[t, len(types[t][0]), *(fmt(per_type[(s, a['mean_center'])][t][key]) for s in specs)]
                    for t in types]), "",
        "## Tuning queries per query", "",
        rows_table(["query", "kind", "qtype", *specs],
                   [[r["query_id"], r["kind"], r["qtype"],
                     *(fmt(next(x for x in qrows[(s, a['mean_center'])] if x["query_id"] == r["query_id"])
                           ["seconds" if r["kind"] == "specific" else "AP@10"]) for s in specs)]
                    for r in qrows[(specs[0], a["mean_center"])]]), "",
        "(specific: seconds watched; category: AP@10)", ""]
    args.report.write_text("\n".join(lines))
    print("\n".join(lines[:30]))
    print("report:", display_path(args.report))
    if not args.no_log:
        for s in specs:
            for c in (True, False):
                log_experiment("tuning", {k: float(v) for k, v in summary[(s, c)].items()},
                               notes=f"step 4.2 tag check {s} mean_center={c}; templated activity prompts", stamp=stamp)
    print("DECISION:", best if clear else "NEAR_TIE")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--tag-check", action="store_true")
    p.add_argument("--models", nargs="+", default=["ViT-B-16:laion2b_s34b_b88k", "ViT-B-16:datacomp_xl_s13b_b90k",
                                                   "ViT-B-16-quickgelu:openai"])
    p.add_argument("--report", type=Path)
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    if args.tag_check:
        args.report = args.report or REPO_ROOT / "results" / "reports" / "pretrained_4.2.md"
        tag_check(args, cfg, stamp)
    else:
        raise SystemExit("choose a mode (--tag-check)")


if __name__ == "__main__":
    main()
