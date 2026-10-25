"""Step 4.2 design check (tuning only): how many annotated actors do the <= 4 embedded crops of a frame cover, per
crop ranking (appearance.crop_rank: score_area vs moving_first) and crop_source variant?

Frames: every 1 fps candidate of every tuning clip (no decoding: index_pass boxes only). Ground truth: annotated actor
boxes on those frames (scripts/detector_smoke.load_ground_truth: RGB person / vehicle, Kitware), clipped and
de-duplicated as in step 3.3. A GT box is covered if a selected crop matches it one-to-one at IoU >=
detection_check.match_iou. IR (third-party boxes, loose) uses the step 3.3 centre match and is descriptive only.
"all detections" (no cap) is the upper bound for detection crops.

Decision rule (fixed before the first run): moving_first replaces score_area only if the detection_then_motion
recall, mean of the RGB person and RGB vehicle cells, is higher; else score_area (the simpler rule, as worded in
CLAUDE.md step 4.2).

Run from the repo root (~2 min): uv run python -m scripts.check_crop_ranking
"""

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.embed.crops import CropConfig, select_crops, variant
from rewind.embed.dedup import candidate_step
from rewind.eval.splits import split_video_ids
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.motion import store
from rewind.paths import REPO_ROOT, display_path
from scripts.detector_smoke import centre_match, dedup_tagged, greedy_match, load_ground_truth
from scripts.select_footage import rows_table

RANKS = ("score_area", "moving_first")
SETS = ("detection", "motion", "detection_then_motion", "all_detections")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "crops_4.2.md")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    base = CropConfig.from_config(cfg, rank="score_area")   # rank replaced per run below
    chk = cfg["detection_check"]

    conn = connect(args.db)
    ids = set(split_video_ids(conn, "tuning"))
    videos = [v for v in load_videos(conn) if v.video_id in ids]
    size_of = {r["video_id"]: (r["width"], r["height"]) for r in conn.execute("SELECT video_id, width, height FROM videos")}
    step = {candidate_step(v.fps, cfg["sampling"]["appearance_fps"], cfg["motion"]["fps"]) for v in videos}
    if len(step) != 1:
        raise ValueError(f"tuning clips have different frame steps {step}")
    gt = load_ground_truth(conn, videos, step.pop())

    # (rank, set, cell) -> [n_gt, found]; cells: rgb person / rgb vehicle per camera and pooled, ir
    tally = defaultdict(lambda: [0, 0])
    n_frames, n_crops = 0, defaultdict(int)
    for v in videos:
        store.load_marker(args.art, v)
        w, h = size_of[v.video_id]
        mot = pd.read_parquet(store.boxes_path(args.art, v))
        mot = {f: g[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, g in mot.groupby("frame")}
        det = pd.read_parquet(store.detections_path(args.art, v))
        det = {f: g for f, g in det.groupby("frame")}
        for frame, kinds in gt[v.video_id].items():
            d = det.get(frame)
            classes = d["class"].to_numpy(str) if d is not None else np.zeros(0, str)
            boxes = d[["x1", "y1", "x2", "y2"]].to_numpy(float) if d is not None else np.zeros((0, 4))
            scores = d["score"].to_numpy(float) if d is not None else np.zeros(0)
            n_frames += 1
            for rank in RANKS:
                crops = select_crops(classes, boxes, scores, mot.get(frame, np.zeros((0, 4))), (w, h),
                                     replace(base, rank=rank))
                chosen = {"detection": [c.box for c in crops if c.source == "detection"],
                          "motion": [c.box for c in variant(crops, "motion", base.max_crops)],
                          "detection_then_motion": [c.box for c in variant(crops, "detection_then_motion",
                                                                           base.max_crops)]}
                big = select_crops(classes, boxes, scores, np.zeros((0, 4)), (w, h),
                                   replace(base, rank=rank, max_crops=10 ** 6))
                chosen["all_detections"] = [c.box for c in big if c.source == "detection"]
                for name, sel in chosen.items():
                    sel = np.array(sel, float).reshape(-1, 4)
                    n_crops[(rank, name)] += len(sel)
                    for kind, items in kinds.items():
                        g, _ = dedup_tagged(items, w, h, chk["gt_dup_iou"])
                        if not len(g):
                            continue
                        found = centre_match(g, sel) if v.modality == "ir" else greedy_match(g, sel, chk["match_iou"])
                        cells = ["ir"] if v.modality == "ir" else [f"rgb {kind}", f"{v.camera_id} {kind}"]
                        for cell in cells:
                            t = tally[(rank, name, cell)]
                            t[0] += len(g)
                            t[1] += int(found.sum())

    def rec(rank, name, cell):
        n, f = tally[(rank, name, cell)]
        return f / n if n else float("nan")

    cells = ["rgb person", "rgb vehicle", "ir"] + sorted({c for (_, _, c) in tally if c[0] == "G" and c != "ir"})
    rows = [[rank, name, f"{n_crops[(rank, name)] / n_frames:.2f}",
             f"{(rec(rank, name, 'rgb person') + rec(rank, name, 'rgb vehicle')) / 2:.3f}",
             *(f"{rec(rank, name, c):.3f}" for c in cells)] for rank in RANKS for name in SETS]
    score = {r: (rec(r, "detection_then_motion", "rgb person") + rec(r, "detection_then_motion", "rgb vehicle")) / 2
             for r in RANKS}
    choice = "moving_first" if score["moving_first"] > score["score_area"] else "score_area"
    n_gt = {c: tally[("score_area", "detection", c)][0] for c in cells}
    lines = [
        "# Step 4.2 — crop ranking check (tuning)", "",
        f"Generated by `scripts/check_crop_ranking.py` at git `{stamp['git_hash']}`, config "
        f"`{stamp['config_hash'][:12]}`. {len(videos)} tuning clips, {n_frames:,} 1 fps frames with annotated actors. "
        f"Max {base.max_crops} crops, min longer side {base.min_crop_px} px, moving share {base.moving_share}, "
        f"dup cover {base.dup_cover}. Covered = one-to-one IoU >= {chk['match_iou']} (IR: centre match, descriptive).",
        "", "GT boxes per cell: " + ", ".join(f"{c} {n:,}" for c, n in n_gt.items()), "",
        rows_table(["rank", "crop set", "crops / frame", "mean(RGB person, vehicle)", *cells], rows), "",
        f"**Decision rule** (docstring): moving_first only if its detection_then_motion mean is higher. "
        f"score_area {score['score_area']:.3f}, moving_first {score['moving_first']:.3f} -> **{choice}**.", ""]
    args.report.write_text("\n".join(lines))
    print("\n".join(lines))
    print("report:", display_path(args.report))
    if not args.no_log:
        log_experiment("tuning", {f"{r}_dtm_rgb_mean": score[r] for r in RANKS}
                       | {f"{r}_{n}_{c.replace(' ', '_')}": rec(r, n, c) for r in RANKS for n in SETS
                          for c in ("rgb person", "rgb vehicle")},
                       notes="step 4.2 crop ranking check (score_area vs moving_first), GT actor coverage by <=4 crops",
                       stamp=stamp)


if __name__ == "__main__":
    main()
