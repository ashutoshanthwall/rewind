"""Step 4.1: sweep the dedup pixel threshold on the TUNING split; report kept frames and pipeline recall per value.

For every tuning clip the 1 fps candidates are decoded once at full resolution (the same path step 4.2 uses), turned
into analysis frames (rewind/embed/dedup.py) with the clip's motion boxes and detections from scripts/index_pass.py,
and every threshold in sampling.dedup_sweep is applied to the same candidates.

Pipeline recall (fixed before the first run): share of tuning events (Kitware RGB + third-party IR, each piece of a
cut activity counted in its own clip) with at least one kept frame inside the event's frame span widened by
metrics.event_tolerance_s on both sides (a hit clip carries search.context_s of context, so a frame that close to
the event still leads to it). The strict variant (no widening) is reported next to it. Decision rule (AUTOPILOT.md
3): the threshold with the fewest kept frames whose pipeline recall is >= sampling.dedup_min_recall.

Never reads the test split (split_video_ids / split_event_ids refuse it). One experiments.csv row per threshold.

Run from the repo root (~10 min):
  uv run python -m scripts.dedup_sweep
  uv run python -m scripts.dedup_sweep --clips <clip> ... --report <scratch.md>   (trial on a few clips; no row)
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.embed.dedup import Candidate, DedupConfig, analysis_gray, boxes_mask, candidate_step, keep_flags
from rewind.eval.splits import split_event_ids, split_video_ids
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.annotations import VEHICLE_START_STOP
from rewind.ingest.manifest import connect, load_videos
from rewind.motion import store
from rewind.paths import REPO_ROOT, display_path
from rewind.video import sampled_frames
from scripts.select_footage import rows_table

FULL_RES = 10 ** 6   # sampled_frames max_width that never scales


def load_candidates(v, size: tuple[int, int], cfg: dict, art: Path) -> list[Candidate]:
    """1 fps candidates of one clip with motion masks and detections (index_pass outputs must be complete)."""
    store.load_marker(art, v)
    step = candidate_step(v.fps, cfg["sampling"]["appearance_fps"], cfg["motion"]["fps"])
    motion = pd.read_parquet(store.boxes_path(art, v))
    motion = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in motion[motion.frame % step == 0].groupby("frame")}
    dets = pd.read_parquet(store.detections_path(art, v)) if cfg["detection"]["enabled"] is True else None
    if dets is not None:
        dets = {f: g for f, g in dets[dets.frame % step == 0].groupby("frame")}
    width = cfg["motion"]["downscale_width"]
    cands = []
    for frame, bgr in sampled_frames(REPO_ROOT / v.path, v.n_frames, step, FULL_RES):
        gray = analysis_gray(bgr, width)
        mask = boxes_mask(motion.get(frame, np.zeros((0, 4))), size, (gray.shape[1], gray.shape[0]))
        d = dets.get(frame) if dets is not None else None
        cands.append(Candidate(
            frame, frame / v.fps, gray, mask,
            d["class"].to_numpy(dtype=str) if d is not None else np.zeros(0, dtype=str),
            d[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) if d is not None else np.zeros((0, 4)),
            d["score"].to_numpy(dtype=float) if d is not None else np.zeros(0)))
    return cands


def recalled(kept_frames: np.ndarray, start: int, end: int, pad: int) -> bool:
    """Some kept frame in [start - pad, end + pad] (inclusive frame span)."""
    i = np.searchsorted(kept_frames, start - pad)
    return bool(i < len(kept_frames) and kept_frames[i] <= end + pad)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "dedup_4.1.md")
    p.add_argument("--clips", nargs="+", help="only these tuning clips (trial: implies --no-log; pass a scratch --report)")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()
    stamp = run_stamp(args.config)
    log = not args.no_log and not args.clips
    if log and stamp["git_hash"].endswith("-dirty") and not args.allow_dirty:
        raise SystemExit("uncommitted changes: commit first (or --allow-dirty), so the rows name the code that ran")
    cfg = yaml.safe_load(args.config.read_text())
    s = cfg["sampling"]
    grid = s["dedup_sweep"]
    base = DedupConfig.from_config(cfg, pixel_thresh=0.0)
    use_det = cfg["detection"]["enabled"] is True
    fps_tol = cfg["metrics"]["event_tolerance_s"]

    conn = connect(args.db)
    ids = set(split_video_ids(conn, "tuning"))
    videos = [v for v in load_videos(conn) if v.video_id in ids]
    if args.clips:
        unknown = set(args.clips) - {v.clip for v in videos}
        if unknown:
            raise ValueError(f"not tuning clips: {sorted(unknown)}")
        videos = [v for v in videos if v.clip in args.clips]
    size_of = {r["video_id"]: (r["width"], r["height"]) for r in conn.execute("SELECT video_id, width, height FROM videos")}
    ev_ids = set(split_event_ids(conn, "tuning"))
    events = defaultdict(list)
    for r in conn.execute("SELECT event_id, video_id, activity_type, start_frame, end_frame FROM events"):
        if r["event_id"] in ev_ids:
            events[r["video_id"]].append(r)

    # per threshold: per camera counters
    kept = defaultdict(lambda: defaultdict(int))
    cands_n = defaultdict(int)
    hits = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))   # thr -> group -> [n, hit widened, hit strict]
    t0 = time.perf_counter()
    for n, v in enumerate(videos, 1):
        cands = load_candidates(v, size_of[v.video_id], cfg, args.art)
        cands_n[v.camera_id] += len(cands)
        pad = round(fps_tol * v.fps)
        for thr in grid:
            c = DedupConfig(**{**base.__dict__, "pixel_thresh": float(thr)})
            flags = keep_flags(cands, c, use_det)
            frames = np.array([x.frame for x, f in zip(cands, flags) if f])
            kept[thr][v.camera_id] += len(frames)
            for e in events[v.video_id]:
                vehicle = e["activity_type"] in VEHICLE_START_STOP
                for group in ("all", v.camera_id, v.modality, "vehicle_start_stop" if vehicle else "other"):
                    h = hits[thr][group]
                    h[0] += 1
                    h[1] += recalled(frames, e["start_frame"], e["end_frame"], pad)
                    h[2] += recalled(frames, e["start_frame"], e["end_frame"], 0)
        print(f"[{n}/{len(videos)}] {v.clip}: {len(cands)} candidates, {time.perf_counter() - t0:.0f} s", flush=True)

    total_c = sum(cands_n.values())
    rows = []
    for thr in grid:
        k = sum(kept[thr].values())
        rows.append({"thresh": thr, "kept": k, "kept_share": k / total_c,
                     "recall": hits[thr]["all"][1] / hits[thr]["all"][0],
                     "recall_strict": hits[thr]["all"][2] / hits[thr]["all"][0]})
    ok = [r for r in rows if r["recall"] >= s["dedup_min_recall"]]
    choice = min(ok, key=lambda r: (r["kept"], -r["thresh"])) if ok else None
    for r in rows:
        print(f"thresh {r['thresh']}: kept {r['kept']} ({r['kept_share']:.3f}), recall {r['recall']:.4f}, "
              f"strict {r['recall_strict']:.4f}")
    print("choice:", choice)

    cams = sorted(cands_n)
    groups = ["all", *cams, "rgb", "ir", "vehicle_start_stop", "other"]
    lines = [
        "# Step 4.1 — dedup threshold sweep (tuning split)", "",
        f"Generated by `scripts/dedup_sweep.py` at git `{stamp['git_hash']}`, config `{stamp['config_hash'][:12]}`. "
        f"{len(videos)} tuning clips, {total_c:,} candidates at {s['appearance_fps']} fps; "
        f"{hits[grid[0]]['all'][0]:,} tuning events. Rules: `rewind/embed/dedup.py`; recall definition: script docstring.",
        "",
        f"Fixed (not swept): pixel delta {s['dedup_pixel_delta']} grey levels at {cfg['motion']['downscale_width']} px wide; "
        f"detections {'on' if use_det else 'off'} (IoU >= {s['dedup_det_iou']}, score floor {s['dedup_det_min_score']}); "
        f"max gap {s['max_gap_s']} s; event widening {fps_tol} s.",
        "",
        "## Summary", "",
        rows_table(["threshold (share of frame)", "kept frames", "kept share", "pipeline recall (±tol)",
                        "strict recall"],
                       [[r["thresh"], f"{r['kept']:,}", f"{r['kept_share']:.3f}", f"{r['recall']:.4f}",
                         f"{r['recall_strict']:.4f}"] for r in rows]),
        "",
        f"**Decision rule** (AUTOPILOT.md 3): fewest kept frames with pipeline recall >= {s['dedup_min_recall']}. "
        + (f"**Choice: {choice['thresh']}** ({choice['kept']:,} kept = {choice['kept_share']:.1%} of candidates, "
           f"recall {choice['recall']:.4f})." if choice else "**No threshold reaches the target.**"),
        "",
        "## Kept share per camera", "",
        rows_table(["threshold", *cams],
                       [[t, *(f"{kept[t][c] / cands_n[c]:.3f}" for c in cams)] for t in grid]),
        "",
        "## Pipeline recall (±tol) per group", "",
        rows_table(["threshold", *(f"{g} (n={hits[grid[0]][g][0]})" for g in groups)],
                       [[t, *(f"{hits[t][g][1] / hits[t][g][0]:.4f}" if hits[t][g][0] else "-" for g in groups)]
                        for t in grid]),
        "",
        "## Strict recall (no widening) per group", "",
        rows_table(["threshold", *groups],
                       [[t, *(f"{hits[t][g][2] / hits[t][g][0]:.4f}" if hits[t][g][0] else "-" for g in groups)]
                        for t in grid]),
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print("report:", display_path(args.report))
    if log:
        for r in rows:
            log_experiment("tuning", {"dedup_thresh": float(r["thresh"]), "kept_frames": r["kept"],
                                      "kept_share": r["kept_share"], "pipeline_recall": r["recall"],
                                      "pipeline_recall_strict": r["recall_strict"]},
                           notes=f"step 4.1 dedup sweep; settings {json.dumps({k: s[k] for k in sorted(s) if k.startswith('dedup_') and k != 'dedup_sweep'})}",
                           stamp=stamp)


if __name__ == "__main__":
    main()
