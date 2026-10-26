"""Step 6.1: within-clip ByteTrack over the vehicle detections of every clip (rewind/track/tracks.py).

Every clip of the manifest (all splits: indexing reads detections, never labels). Per clip: the detection grid
(frames 0, step, ... with step = video fps / detection.fps; must equal the index pass's n_samples), the clip's
detections (artifacts/detections, scripts/index_pass.py) -> one fresh tracker -> rows on matched samples.
Outputs (written under temporary names, then renamed; the marker last):
  artifacts/tracks/<camera>/<clip>.parquet     frame, t_s, track (within-clip index, never an identity), class,
                                               score, x1, y1, x2, y2 (the matched detection's box, original pixels)
  artifacts/tracks/<camera>/<clip>.done.json   settings hash + counts
Re-running skips done clips; a marker with other settings stops the run (outputs of two settings never mix).

Run from the repo root (background, CLAUDE.md rule 16):
  nohup uv run python -m scripts.track_pass > results/logs/track_pass.log 2>&1 & echo $! > results/logs/track_pass.pid
  uv run python -m scripts.track_pass --clips <clip> [--out-root <dir>]     (test on single clips)
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.motion import store as motion_store
from rewind.paths import REPO_ROOT, display_path
from rewind.track import store
from rewind.track.tracks import TrackerConfig, track_clip
from rewind.video import keep_step


def settings_hash(cfg: dict, det_settings: str) -> str:
    """Tracker settings + the settings hash of the detections they were computed from."""
    blob = json.dumps({"tracker": cfg["events"]["tracker"], "bytetrack": cfg["events"]["bytetrack"],
                       "vehicle_classes": cfg["events"]["vehicle_classes"], "merge_iou": cfg["detection"]["nms_iou"],
                       "detections": det_settings}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def process_clip(v, cfg: dict, tcfg: TrackerConfig, art: Path, out_root: Path, settings: str) -> dict:
    t0 = time.perf_counter()
    det_marker = motion_store.load_marker(art, v)
    frames = np.arange(0, v.n_frames, keep_step(v.fps, cfg["detection"]["fps"]))
    if len(frames) != det_marker["n_samples"]:
        raise ValueError(f"{v.clip}: sample grid {len(frames)} != index pass n_samples {det_marker['n_samples']}")
    dets = pd.read_parquet(motion_store.detections_path(art, v))
    if len(dets) != det_marker["n_detections"]:
        raise ValueError(f"{v.clip}: {len(dets)} detections != marker {det_marker['n_detections']}")
    tracks = track_clip(dets, frames, v.fps, tcfg)
    path = store.tracks_path(out_root, v)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tracks.to_parquet(tmp, index=False)
    tmp.rename(path)
    n_tracks = int(tracks["track"].nunique()) if len(tracks) else 0
    marker = {"clip": v.clip, "video_id": v.video_id, "settings": settings, "n_rows": len(tracks),
              "n_tracks": n_tracks, "n_vehicle_detections": int(dets["class"].isin(tcfg.classes).sum()),
              "seconds": round(time.perf_counter() - t0, 2), "video_s": v.n_frames / v.fps}
    final = store.marker_path(out_root, v)
    tmp = final.with_name(final.name + ".tmp")
    tmp.write_text(json.dumps(marker))
    tmp.rename(final)
    return marker


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--out-root", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--clips", nargs="+", help="only these clips (testing); implies no experiments.csv row")
    args = p.parse_args()
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["detection"]["enabled"] is not True:
        raise ValueError("detection.enabled is not true: no detections to track")
    tcfg = TrackerConfig.from_config(cfg)
    art = REPO_ROOT / "artifacts"

    videos = load_videos(connect(args.db))
    if args.clips:
        missing = set(args.clips) - {v.clip for v in videos}
        if missing:
            raise ValueError(f"not in the manifest: {sorted(missing)}")
        videos = [v for v in videos if v.clip in args.clips]
    det_settings = {motion_store.load_marker(art, v)["settings"] for v in videos}
    if len(det_settings) != 1:
        raise ValueError(f"detections were made with {len(det_settings)} different settings: {det_settings}")
    settings = settings_hash(cfg, det_settings.pop())

    todo = []
    for v in videos:
        marker = store.marker_path(args.out_root, v)
        if marker.exists():
            old = json.loads(marker.read_text())
            if old["settings"] != settings:
                raise ValueError(f"{marker} has settings {old['settings']}, current {settings}: remove the old "
                                 "tracks or restore the config")
            continue
        todo.append(v)
    print(f"track_pass: {len(videos) - len(todo)} clips done, {len(todo)} to do; settings {settings}", flush=True)
    t_start, totals = time.perf_counter(), {"rows": 0, "tracks": 0, "vehicle_detections": 0}
    for n, v in enumerate(todo, 1):
        m = process_clip(v, cfg, tcfg, art, args.out_root, settings)
        totals["rows"] += m["n_rows"]
        totals["tracks"] += m["n_tracks"]
        totals["vehicle_detections"] += m["n_vehicle_detections"]
        print(f"[{n}/{len(todo)}] {v.clip}: {m['n_vehicle_detections']} vehicle detections -> {m['n_tracks']} tracks, "
              f"{m['n_rows']} rows, {m['seconds']:.1f} s", flush=True)
    elapsed = time.perf_counter() - t_start
    print(f"track_pass: finished {len(todo)} clips in {elapsed / 60:.1f} min", flush=True)
    if todo and not args.clips and args.out_root == art:
        log_experiment("none", {"clips": len(todo), "minutes": elapsed / 60, **totals},
                       f"step 6.1 track_pass (ByteTrack, within clip), settings {settings}", stamp=stamp)
    print(f"outputs under {display_path(args.out_root / 'tracks')}")


if __name__ == "__main__":
    main()
