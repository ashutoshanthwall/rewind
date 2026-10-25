"""Check the outputs of scripts/index_pass.py (read-only; safe to run while the pass is still going).

For every clip with a done marker:
  - all output files exist, no leftover *.tmp, the marker's settings hash equals the current config's;
  - samples = ceil(n_frames / step) (marker), per-second rows = ceil(n_frames / fps), activity finite and in [0, 1];
  - motion boxes and detections: frames on the sample grid, t_s = frame / fps, boxes inside the frame (detections
    may overhang by `--overhang` x the frame size: the detector predicts boxes past the edge), x2 > x1, y2 > y1;
  - detections: classes within detection.classes, score >= conf_thresh, counts per class.
Also flags clips with zero activity or zero detections (not an error: an empty scene is possible) and summarises
per camera. Exits non-zero if any check fails. No labels or splits are read.

Run from the repo root:
  uv run python -m scripts.check_index_pass [--report results/reports/index_pass_check.md]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT, display_path
from rewind.video import keep_step
from scripts.index_pass import settings_hash
from scripts.select_footage import markdown_table


def check_clip(v, size, marker: dict, art: Path, cfg: dict, overhang: float) -> tuple[list[str], dict]:
    """overhang: fraction of the frame width/height a detection box may extend past the edge."""
    problems, w, h = [], *size
    dcfg = cfg["detection"]
    step = keep_step(v.fps, cfg["motion"]["fps"])
    n_samples = math.ceil(v.n_frames / step)
    if marker["settings"] != settings_hash(cfg):
        problems.append(f"settings {marker['settings']} != current {settings_hash(cfg)}")
    if marker["n_samples"] != n_samples:
        problems.append(f"{marker['n_samples']} samples, expected {n_samples}")
    sec = pd.read_parquet(art / "motion" / v.camera_id / f"{v.clip}.seconds.parquet")
    if len(sec) != math.ceil(v.n_frames / v.fps) or sec["second"].tolist() != list(range(len(sec))):
        problems.append(f"{len(sec)} per-second rows, expected {math.ceil(v.n_frames / v.fps)}")
    act = sec["activity"].to_numpy()
    if not np.isfinite(act).all() or act.min() < 0 or act.max() > 1:
        problems.append("activity not finite or outside [0, 1]")

    def frame_checks(df: pd.DataFrame, what: str, slack_frac: float) -> None:
        sx, sy = slack_frac * w, slack_frac * h
        if len(df) == 0:
            return
        if (df["frame"] % step != 0).any() or df["frame"].min() < 0 or df["frame"].max() >= v.n_frames:
            problems.append(f"{what}: frames off the sample grid or outside the clip")
        if not np.allclose(df["t_s"], df["frame"] / v.fps):
            problems.append(f"{what}: t_s != frame / fps")
        if ((df["x2"] <= df["x1"]) | (df["y2"] <= df["y1"])).any():
            problems.append(f"{what}: empty boxes")
        if (df["x1"] < -sx).any() or (df["y1"] < -sy).any() or (df["x2"] > w + sx).any() or (df["y2"] > h + sy).any():
            problems.append(f"{what}: boxes outside the {w}x{h} frame by more than {slack_frac:.0%} of its size")

    boxes = pd.read_parquet(art / "motion" / v.camera_id / f"{v.clip}.boxes.parquet")
    frame_checks(boxes, "motion boxes", 0)
    if len(boxes) != marker["n_motion_boxes"]:
        problems.append("motion box count differs from the marker")
    stats = {"camera": v.camera_id, "modality": v.modality, "activity_mean": float(act.mean()),
             "activity_zero_s": int((act == 0).sum()), "seconds": len(act), "motion_boxes": len(boxes)}
    if dcfg["enabled"] is True:
        det_path = art / "detections" / v.camera_id / f"{v.clip}.parquet"
        if not det_path.exists():
            problems.append("detections file missing")
            return problems, stats
        det = pd.read_parquet(det_path)
        frame_checks(det, "detections", overhang)
        if len(det) != marker["n_detections"]:
            problems.append("detection count differs from the marker")
        if len(det) and (~det["class"].isin(dcfg["classes"])).any():
            problems.append(f"detection classes outside config: {sorted(set(det['class']) - set(dcfg['classes']))}")
        if len(det) and (det["score"] < dcfg["conf_thresh"] - 1e-6).any():
            problems.append("detections below conf_thresh")
        stats["detections"] = len(det)
        stats["frames_with_det"] = det["frame"].nunique() if len(det) else 0
        stats.update({f"n_{c}": int((det["class"] == c).sum()) if len(det) else 0 for c in dcfg["classes"]})
    return problems, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--overhang", type=float, default=0.03,
                   help="fraction of the frame size a detection box may extend past the edge (seen: 16.7 px of 1080)")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    conn = connect(args.db)
    videos = load_videos(conn)
    size_of = {r["video_id"]: (r["width"], r["height"]) for r in conn.execute("SELECT video_id, width, height FROM videos")}

    leftovers = sorted(str(x.relative_to(args.art)) for x in args.art.rglob("*.tmp"))
    rows, problems, n_done = [], defaultdict(list), 0
    for v in videos:
        marker_path = args.art / "motion" / v.camera_id / f"{v.clip}.done.json"
        if not marker_path.exists():
            continue
        n_done += 1
        probs, stats = check_clip(v, size_of[v.video_id], json.loads(marker_path.read_text()), args.art, cfg,
                                  args.overhang)
        problems[v.clip] += probs
        rows.append({"clip": v.clip, **stats})
    df = pd.DataFrame(rows)
    lines = [f"# index_pass output check", "", f"Clips done: {n_done} of {len(videos)}. Leftover .tmp files: "
             f"{len(leftovers)} (only expected for the clip being written right now)." + (f" {leftovers}" if leftovers else ""), ""]
    bad = {c: pr for c, pr in problems.items() if pr}
    lines += [f"**Problems: {sum(map(len, bad.values()))} in {len(bad)} clips.**", ""]
    lines += [f"- {c}: {'; '.join(pr)}" for c, pr in bad.items()]
    if len(df):
        agg = {"clip": "count", "seconds": "sum", "activity_mean": "mean", "motion_boxes": "sum"}
        if "detections" in df:
            agg.update({c: "sum" for c in df.columns if c.startswith("n_")} | {"detections": "sum"})
        per_cam = df.groupby(["camera", "modality"]).agg(agg).rename(columns={"clip": "clips"})
        lines += ["", "Per camera (sums over done clips):", "", markdown_table(per_cam.round(4))]
        quiet = df[(df["activity_mean"] == 0) | (df.get("detections", pd.Series(1, index=df.index)) == 0)]
        lines += ["", f"Clips with zero activity or zero detections ({len(quiet)}; not an error by itself):", ""]
        lines += [f"- {r.clip}: activity mean {r.activity_mean:.5f}, zero-activity seconds {r.activity_zero_s}/"
                  f"{r.seconds}, motion boxes {r.motion_boxes}, detections {getattr(r, 'detections', '–')}"
                  for r in quiet.itertuples()]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.report:
        args.report.write_text(text)
        print(f"report: {display_path(args.report)}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
