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

Mode --ablation (step 4.4, GATE 4.4, dev split): all 12 variants (frame_mode x crop_source x mean_center) on the 17
dev queries over the collected all-footage index (scripts/collect_appearance.py; centering with all-footage camera
means), one experiments row each. Decisions with the paired rule (rewind/eval/gate.py), in this fixed order, fixed
before any dev result existed:
  1. crop_source, at the config's frame_mode and mean_center: start from none; motion replaces it if the rule
     adopts motion over none; then detection_then_motion replaces the current choice if adopted over it;
  2. mean_center: on (complex) vs off, at the chosen crop_source and the config's frame_mode;
  3. frame_mode: tiles2 (complex, 2 vectors per frame) vs pad, at the choices of 1 and 2.
Report: results/reports/ablation_4.4.md (variant table, decisions with wins/ties/losses and CIs, breakdowns by qtype,
camera (G330 separately, open-issue item 7) and modality, 5 example queries (the first 5 of queries_dev.csv) with
their top-3 clips).

Run from the repo root:
  uv run python -m scripts.appearance_search --tag-check
  uv run python -m scripts.appearance_search --ablation
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
from rewind.eval.metrics import Span, bootstrap_ci, first_hit
from rewind.eval.queries import load_queries
from rewind.eval.splits import split_event_ids, split_video_ids
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT, display_path
from rewind.search.core import SearchConfig, group_hits, mean_center, top_k
from rewind.eval.gate import paired_decision
from scripts.collect_appearance import index_dir, load_means
from scripts.run_baselines import QUERIES_DIR, category_metrics, category_truth, specific_metrics, summarise
from scripts.select_footage import rows_table

SPLITS_RUN = ("tuning", "dev")


class Index:
    """The vectors of one variant over one split's videos, ready for brute-force search."""

    def __init__(self, vecs: np.ndarray, meta: pd.DataFrame, videos: list, frame_mode: str, crop_source: str,
                 max_crops: int, center: bool, means: dict | None = None):
        """means: per-camera means over ALL footage (step 4.3, scripts/collect_appearance); None = the means of
        these vectors (the 4.2 tag check, before all footage is embedded)."""
        ids = {v.video_id for v in videos}
        keep = variant_mask(meta, frame_mode, crop_source, max_crops) & meta["video_id"].isin(ids).to_numpy()
        self.meta = meta[keep].reset_index(drop=True)
        x = vecs[keep]
        self.x = mean_center(x, self.meta["camera_id"].to_numpy(), means) if center else x.astype(np.float32)
        self.by_id = {v.video_id: v for v in videos}
        self.durations = {v.video_id: v.n_frames / v.fps for v in videos}
        self.rgb = (self.meta["modality"] == "rgb").to_numpy()
        self.footage_s = sum(self.durations.values())

    def search(self, q: np.ndarray, cfg: SearchConfig, rgb_only: bool) -> list:
        rows = np.flatnonzero(self.rgb) if rgb_only else np.arange(len(self.x))
        idx, sc = top_k(self.x[rows], q, cfg.k)
        return group_hits(rows[idx], sc, self.meta["video_id"].to_numpy(), self.meta["t_s"].to_numpy(),
                          self.durations, cfg)

    def ranked(self, q: np.ndarray, cfg: SearchConfig, rgb_only: bool, query: dict | None = None) -> list[Span]:
        """query: the frozen query row (unused here; rankers that read the text, e.g. step 6.2's, need it)."""
        return [c.span(self.by_id[c.video_id]) for c in self.search(q, cfg, rgb_only)]


def split_videos(conn, split: str) -> list:
    if split not in SPLITS_RUN:
        raise PermissionError(f"appearance_search runs on {SPLITS_RUN} only (test is read once, at step 8.2)")
    ids = set(split_video_ids(conn, split))
    return [v for v in load_videos(conn) if v.video_id in ids]


def query_rows(conn, split: str, index: Index, clip: Clip, cfg: dict, scfg: SearchConfig,
               qtypes: tuple | None = None) -> list[dict]:
    """The frozen queries of a split (only these qtypes, if given), scored like the baselines. index: anything with
    footage_s and ranked(q, cfg, rgb_only, query=<query row>); clip: anything with encode_text()."""
    m = cfg["metrics"]
    out = []
    for q in load_queries(QUERIES_DIR, split):
        if qtypes is not None and q["qtype"] not in qtypes:
            continue
        base = {k: q[k] for k in ("query_id", "kind", "qtype", "specificity", "modality", "camera_id")}
        z = clip.encode_text([q["query"]])[0]
        if q["kind"] == "specific":
            gt = Span(q["camera_id"], datetime.fromisoformat(q["start_time"]), datetime.fromisoformat(q["end_time"]))
            ranked = index.ranked(z, scfg, rgb_only=False, query=q)
            rank = first_hit(ranked, gt)
            out.append(base | specific_metrics([ranked], gt, index.footage_s, m["recall_k"])
                       | {"hit": rank is not None, "hit_rank": -1 if rank is None else rank + 1,
                          "n_clips": len(ranked)})
        else:
            inst, dont_care = category_truth(conn, split, q["activity_type"])
            ranked = index.ranked(z, scfg, rgb_only=True, query=q)
            out.append(base | {"instances": len(inst)} | category_metrics([ranked], inst, dont_care, m["category_k"]))
    return out


def template_types(conn, split: str, min_instances: int) -> dict[str, tuple]:
    """Kitware activity types with >= min_instances activities in the split -> (instances, don't-care)."""
    in_split = set(split_event_ids(conn, split))
    types = sorted({r[1] for r in conn.execute("SELECT event_id, activity_type FROM events "
                                               "WHERE label_source != 'contrib'") if r[0] in in_split})
    out = {}
    for t in types:
        inst, dc = category_truth(conn, split, t)
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
    if args.clips:      # smoke test of the code path only: ground truth stays split-wide, numbers are meaningless
        videos = [v for v in videos if v.clip in args.clips]
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


GRID = [(fm, cs, c) for fm in ("tiles2", "pad") for cs in ("none", "motion", "detection_then_motion")
        for c in (True, False)]


def vname(v: tuple) -> str:
    return f"{v[0]}/{v[1]}/{'centered' if v[2] else 'raw'}"


def ablation(args, cfg, stamp) -> None:
    a, m = cfg["appearance"], cfg["metrics"]
    scfg = SearchConfig.from_config(cfg)
    ap_key = f"AP@{max(m['category_k'])}"
    model, tag = a["model"], a["pretrained"]
    conn = connect(args.db)
    videos = split_videos(conn, args.split)
    if args.clips:
        videos = [v for v in videos if v.clip in args.clips]
    d = index_dir(args.art, model, tag)
    vecs, meta = np.load(d / "vectors.npy"), pd.read_parquet(d / "meta.parquet")
    clip = Clip(model, tag)
    results, n_vec, indexes = {}, {}, {}
    for v in GRID:
        fm, cs, c = v
        index = Index(vecs, meta, videos, fm, cs, a["max_crops_per_frame"], c,
                      load_means(args.art, model, tag, fm, cs) if c else None)
        results[v] = query_rows(conn, args.split, index, clip, cfg, scfg)
        n_vec[v] = len(index.x)
        indexes[v] = index
        print(f"{vname(v)}: {n_vec[v]:,} vectors", flush=True)
    summ = {v: summarise(results[v], cfg) for v in GRID}
    if not args.no_log:
        for v in GRID:
            log_experiment(args.split, summ[v] | {"vectors": n_vec[v]},
                           notes=f"step 4.4 appearance ablation {vname(v)}; {model}/{tag}; NumPy brute force",
                           stamp=stamp)

    gate = lambda s, c_: paired_decision(results[s], results[c_], ap_key, m["bootstrap_n"], m["bootstrap_seed"],
                                         m["ci_level"])
    fm0, c0 = a["frame_mode"], a["mean_center"]
    steps, cs_star = [], "none"
    for nxt in ("motion", "detection_then_motion"):
        dec = gate((fm0, cs_star, c0), (fm0, nxt, c0))
        steps.append((f"crop_source: {nxt} vs {cs_star}", (fm0, cs_star, c0), (fm0, nxt, c0), dec))
        if dec["adopt"]:
            cs_star = nxt
    dec = gate((fm0, cs_star, False), (fm0, cs_star, True))
    steps.append(("mean_center: on vs off", (fm0, cs_star, False), (fm0, cs_star, True), dec))
    c_star = dec["adopt"]
    dec = gate(("pad", cs_star, c_star), ("tiles2", cs_star, c_star))
    steps.append(("frame_mode: tiles2 vs pad", ("pad", cs_star, c_star), ("tiles2", cs_star, c_star), dec))
    fm_star = "tiles2" if dec["adopt"] else "pad"
    final = (fm_star, cs_star, c_star)

    fmt = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (
        f"{x:.3f}" if isinstance(x, float) else str(x))
    ks = m["recall_k"]
    lines = [
        "# Step 4.4 — appearance search on dev: ablations and GATE 4.4", "",
        f"Generated by `scripts/appearance_search.py --ablation` at git `{stamp['git_hash']}`, config "
        f"`{stamp['config_hash'][:12]}`. {model}/{tag}; NumPy brute force over the dev footage ({len(videos)} clips, "
        f"{sum(v.n_frames / v.fps for v in videos) / 3600:.2f} h). Search: top {scfg.k} vectors -> clips (merge "
        f"{scfg.merge_gap_s} s, cap {scfg.max_clip_s} s, mean top-3, ±{scfg.context_s} s). Centering: per-camera "
        "means over all footage (step 4.3). Metrics as in the baselines report (same code); category queries "
        "search RGB vectors only.", "",
        "## Variants", "",
        rows_table(["variant", "vectors", "specific median s", "mean s", *(f"R@{k}" for k in ks),
                    "category mAP@10", "mAP@50"],
                   [[vname(v) + (" **(final)**" if v == final else ""), f"{n_vec[v]:,}",
                     fmt(summ[v]["median_seconds"]), fmt(summ[v]["mean_seconds"]),
                     *(fmt(summ[v][f"R@{k}"]) for k in ks), fmt(summ[v]["mAP@10"]), fmt(summ[v]["mAP@50"])]
                    for v in GRID]), "",
        "Dev baselines for reference (`results/reports/baselines_3.2.md`): median seconds watched sequential 5,932, "
        "random 6,413, witness ±10 min 3,237, motion 1,513.", "",
        f"## GATE 4.4 decisions (paired rule, `rewind/eval/gate.py`; specific: seconds watched, category: {ap_key})", "",
        rows_table(["step", "simple", "complex", "wins / ties / losses", "specific mean diff s [CI]",
                    f"category mean diff {ap_key} [CI]", "mean better", "adopt complex"],
                   [[name, vname(s), vname(c_), f"{dd['wins']} / {dd['ties']} / {dd['losses']}",
                     _fam(dd, "specific_seconds", "{:+.0f}"), _fam(dd, "category_" + ap_key, "{:+.3f}"),
                     dd["mean_better"], dd["adopt"]] for name, s, c_, dd in steps]), "",
        f"**Final variant: {vname(final)}** -> config appearance.frame_mode={fm_star}, crop_source={cs_star}, "
        f"mean_center={c_star}.", "",
    ]
    for v, label in ((final, "final"), ((fm0, a["crop_source"], c0), "config before the gate")):
        rows = pd.DataFrame(results[v])
        spec = rows[rows.kind == "specific"]
        lines += [f"## Breakdowns, {label} variant ({vname(v)}); specific queries, descriptive where n < 5", ""]
        for col in ("qtype", "camera_id", "modality"):
            g = spec.groupby(col)
            lines += [rows_table([col, "n", "median s", "R@10", "hits"],
                                 [[k, len(x), fmt(float(x.seconds.median())), fmt(float(x["R@10"].mean())),
                                   int(x.hit.sum())] for k, x in g]), ""]
    rows = results[final]
    lines += ["## Per query (final variant)", "",
              rows_table(["query", "kind", "qtype", "camera", "seconds / AP@10", "hit rank", "R@10"],
                         [[r["query_id"], r["kind"], r["qtype"], r.get("camera_id") or "",
                           fmt(r["seconds"] if r["kind"] == "specific" else r["AP@10"]),
                           r.get("hit_rank", ""), fmt(r.get("R@10"))] for r in rows]), "",
              "## 5 example queries (first 5 of queries_dev.csv), final variant, top-3 clips", ""]
    proxies = {r["video_id"]: r["path"] for r in conn.execute("SELECT video_id, path FROM proxies")}
    index = indexes[final]
    for q in load_queries(QUERIES_DIR, args.split)[:5]:
        z = clip.encode_text([q["query"]])[0]
        clips = index.search(z, scfg, rgb_only=q["kind"] == "category")[:3]
        lines += [f"**{q['query_id']}** ({q['kind']}, {q['qtype']}): \"{q['query']}\"" +
                  (f" — truth: {q['camera_id']} {q['start_time'][11:19]}-{q['end_time'][11:19]}"
                   if q["kind"] == "specific" else f" — truth: every {q['activity_type']}"), ""]
        trs = []
        for i, c_ in enumerate(clips, 1):
            vid = index.by_id[c_.video_id]
            best = max(c_.hits, key=lambda h: h[1])
            r = index.meta.iloc[best[2]]
            what = f"{r['view']}" if r["type"] == "frame" else f"{r['crop_source']} crop {r['class'] or ''}".strip()
            span = c_.span(vid)
            trs.append([i, vid.camera_id, f"{span.start:%H:%M:%S}-{span.end:%H:%M:%S}",
                        f"{c_.start_s:.0f}-{c_.end_s:.0f} s", proxies[c_.video_id], f"{best[0]:.0f} s", what,
                        f"({r['x1']:.0f}, {r['y1']:.0f}, {r['x2']:.0f}, {r['y2']:.0f})", f"{c_.score:.3f}"])
        lines += [rows_table(["#", "camera", "wall clock", "in file", "proxy", "best hit at", "matched", "box",
                              "score"], trs), ""]
    args.report.write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    print("report:", display_path(args.report))
    print("FINAL:", vname(final))


def _fam(dec: dict, key: str, f: str) -> str:
    x = dec["families"].get(key)
    if x is None:
        return "—"
    ci = "n<2" if x["ci"] is None else f"[{f.format(x['ci'][0])}, {f.format(x['ci'][1])}]"
    return f"{f.format(x['mean_diff'])} {ci} (n={x['n']})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--tag-check", action="store_true")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--split", choices=SPLITS_RUN, default="dev", help="--ablation split (GATE 4.4 uses dev)")
    p.add_argument("--models", nargs="+", default=["ViT-B-16:laion2b_s34b_b88k", "ViT-B-16:datacomp_xl_s13b_b90k",
                                                   "ViT-B-16-quickgelu:openai"])
    p.add_argument("--report", type=Path)
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--clips", nargs="+", help="smoke test on these clips only (implies --no-log; pass --report)")
    args = p.parse_args()
    if args.clips:
        args.no_log = True
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    if args.tag_check:
        args.report = args.report or REPO_ROOT / "results" / "reports" / "pretrained_4.2.md"
        tag_check(args, cfg, stamp)
    elif args.ablation:
        args.report = args.report or REPO_ROOT / "results" / "reports" / "ablation_4.4.md"
        ablation(args, cfg, stamp)
    else:
        raise SystemExit("choose a mode (--tag-check or --ablation)")


if __name__ == "__main__":
    main()
