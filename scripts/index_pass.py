"""Phase 3 full pass (steps 3.1 + 3.4 in one decode): per clip, 5 fps samples -> MOG2 motion + object detection.

Every clip of the manifest (all splits: indexing reads footage, never labels) is decoded once
(rewind/video.sampled_frames) at max(motion.downscale_width, detection.imgsz). Each sample goes to
  - MOG2 (rewind/motion/mog2.py) at motion.downscale_width -> per-second activity + motion boxes;
  - the detector of config `detection` (rewind/detect/detectors.py), if detection.enabled, at the input size of the
    clip's modality (detection.imgsz.rgb / .ir).
A decode thread runs ahead of MOG2 + detection (queue of 2 batches), so CPU decoding overlaps GPU detection.

Outputs per clip (paths: rewind/motion/store.py; boxes in ORIGINAL video pixels; t_s = frame / fps = seconds in the
clip = proxy time):
  artifacts/motion/<camera>/<clip>.seconds.parquet     second, activity
  artifacts/motion/<camera>/<clip>.boxes.parquet       frame, t_s, x1, y1, x2, y2
  artifacts/detections/<camera>/<clip>.parquet         frame, t_s, class, score, x1, y1, x2, y2   (if enabled)
  artifacts/motion/<camera>/<clip>.done.json           marker: written last; settings hash + counts
Files are written under temporary names and renamed; a clip counts as done only when its marker exists. Re-running
skips done clips. A marker with different settings (config sections motion/detection or weights hash) stops the
run: outputs from two settings are never mixed (delete the artifacts by hand, or restore the config). Any error
stops the run (no clip is skipped silently); fix it and re-run to resume.

Run from the repo root (long job: background, CLAUDE.md rule 16):
  nohup uv run python -m scripts.index_pass > results/logs/index_pass.log 2>&1 & echo $! > results/logs/index_pass.pid
  uv run python -m scripts.index_pass --clips <clip> [--out-root <dir>]     (test on single clips)
"""

import argparse
import hashlib
import json
import queue
import threading
import time
from pathlib import Path

import pandas as pd
import yaml

from rewind.detect.detectors import COCO_NAMES, make_detector
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos, sha256_file
from rewind.motion import store
from rewind.motion.mog2 import MotionConfig, MotionScorer, per_second, to_size, warmed
from rewind.paths import REPO_ROOT, display_path
from rewind.video import keep_step, sampled_frames, scaled_size

_END = object()


def settings_hash(cfg: dict) -> str:
    """Hash of everything that changes the outputs: config sections motion + detection (incl. weights hash)."""
    blob = json.dumps({"motion": cfg["motion"], "detection": cfg["detection"]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def prefetch(iterable, maxsize: int):
    """Iterate `iterable` in a background thread; exceptions are re-raised in the consumer."""
    q: queue.Queue = queue.Queue(maxsize=maxsize)

    def work():
        try:
            for item in iterable:
                q.put(item)
            q.put(_END)
        except BaseException as e:  # handed to the consumer, which re-raises it
            q.put(e)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is _END:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def load_detectors(dcfg: dict, modalities: set[str]) -> dict:
    """{modality: detector} at detection.imgsz[modality]; {} if detection is disabled. Checks the weights hash."""
    if dcfg["enabled"] is not True:
        if dcfg["enabled"] is False:
            return {}
        raise ValueError(f"config detection.enabled must be true or false, got {dcfg['enabled']!r}")
    for key in ("backend", "model", "weights", "weights_sha256", "imgsz", "conf_thresh", "nms_iou", "classes"):
        if dcfg.get(key) in (None, "TBD"):
            raise ValueError(f"config detection.{key} is missing or TBD")
    missing = modalities - set(dcfg["imgsz"])
    if missing:
        raise ValueError(f"config detection.imgsz has no input size for {sorted(missing)}")
    weights = REPO_ROOT / dcfg["weights"]
    if sha256_file(weights) != dcfg["weights_sha256"]:
        raise ValueError(f"{weights} does not match detection.weights_sha256 (wrong or changed weights file)")
    return {m: make_detector(dcfg["backend"], str(weights), dcfg["imgsz"][m], dcfg["conf_thresh"], dcfg["nms_iou"],
                             dcfg["classes"], dcfg["model"]) for m in sorted(modalities)}


def process_clip(v, size: tuple[int, int], cfg: dict, mcfg: MotionConfig, detector, out_root: Path) -> dict:
    """Run one clip and write its outputs + done marker. Returns the marker dict."""
    dcfg = cfg["detection"]
    w, h = size
    step = keep_step(v.fps, mcfg.fps)
    det_width = dcfg["imgsz"][v.modality] if detector else 0
    decode_width = max(mcfg.downscale_width, det_width)
    motion_size = scaled_size(w, h, mcfg.downscale_width)
    det_size = scaled_size(w, h, det_width) if detector else None
    decoded = scaled_size(w, h, decode_width)
    det_scale = w / det_size[0] if detector else None

    scorer = MotionScorer(mcfg, (w, h))
    motion, det_rows, batch = [], [], []
    t0 = time.perf_counter()

    def flush():
        frames = [to_size(f, det_size) for _, f in batch]
        for (i, _), d in zip(batch, detector(frames)):
            for box, score, cls in zip(d.boxes * det_scale, d.scores, d.classes):
                det_rows.append((i, i / v.fps, COCO_NAMES[cls], float(score), *map(float, box)))
        batch.clear()

    samples = prefetch(sampled_frames(REPO_ROOT / v.path, v.n_frames, step, decode_width), 2 * dcfg["batch_size"])
    for i, frame in warmed(samples, scorer, motion_size):
        motion.append(scorer.score(i, to_size(frame, motion_size)))
        if detector:
            batch.append((i, frame))
            if len(batch) == dcfg["batch_size"]:
                flush()
    if detector and batch:
        flush()
    elapsed = time.perf_counter() - t0

    act = per_second(motion, v.fps, v.n_frames)
    boxes_file = store.boxes_path(out_root, v)
    outputs = {
        store.seconds_path(out_root, v): pd.DataFrame({"second": range(len(act)), "activity": act}),
        boxes_file: pd.DataFrame(
            [(m.frame, m.frame / v.fps, *b) for m in motion for b in m.boxes],
            columns=["frame", "t_s", "x1", "y1", "x2", "y2"]),
    }
    if detector:
        outputs[store.detections_path(out_root, v)] = pd.DataFrame(
            det_rows, columns=["frame", "t_s", "class", "score", "x1", "y1", "x2", "y2"]).astype(
            {"score": "float32", "x1": "float32", "y1": "float32", "x2": "float32", "y2": "float32"})
    for path, df in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.rename(path)
    marker = {"clip": v.clip, "video_id": v.video_id, "settings": settings_hash(cfg), "n_samples": len(motion),
              "n_motion_boxes": len(outputs[boxes_file]),
              "n_detections": len(det_rows) if detector else None, "decode_width": decoded[0],
              "seconds": round(elapsed, 2), "video_s": v.n_frames / v.fps}
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
    stamp = run_stamp(args.config)  # code + settings at the start: the row must name what ran, not what's there at the end
    cfg = yaml.safe_load(args.config.read_text())
    mcfg = MotionConfig.from_config(cfg["motion"])
    if cfg["detection"]["enabled"] is True and cfg["detection"]["fps"] != mcfg.fps:
        raise ValueError("motion.fps and detection.fps must be equal: both run on the same samples")
    settings = settings_hash(cfg)

    conn = connect(args.db)
    videos = load_videos(conn)
    if args.clips:
        missing = set(args.clips) - {v.clip for v in videos}
        if missing:
            raise ValueError(f"not in the manifest: {sorted(missing)}")
        videos = [v for v in videos if v.clip in args.clips]
    size_of = {r["video_id"]: (r["width"], r["height"]) for r in conn.execute("SELECT video_id, width, height FROM videos")}

    todo, done_s = [], 0.0
    for v in videos:
        marker = store.marker_path(args.out_root, v)
        if marker.exists():
            old = json.loads(marker.read_text())
            if old["settings"] != settings:
                raise ValueError(f"{marker} was written with settings {old['settings']}, current {settings}: "
                                 "outputs of two settings would mix. Restore the config or remove the artifacts.")
            done_s += old["video_s"]
            continue
        todo.append(v)
    total_s = sum(v.n_frames / v.fps for v in todo)
    print(f"index_pass: {len(videos) - len(todo)} clips done, {len(todo)} to do ({total_s / 3600:.2f} h of video); "
          f"settings {settings}; detection {'on: ' + cfg['detection']['model'] if cfg['detection']['enabled'] is True else 'off'}",
          flush=True)
    detectors = load_detectors(cfg["detection"], {v.modality for v in todo}) if todo else {}

    t_start, video_done, totals = time.perf_counter(), 0.0, {"samples": 0, "detections": 0, "motion_boxes": 0}
    for n, v in enumerate(todo, 1):
        m = process_clip(v, size_of[v.video_id], cfg, mcfg, detectors.get(v.modality), args.out_root)
        video_done += m["video_s"]
        totals["samples"] += m["n_samples"]
        totals["motion_boxes"] += m["n_motion_boxes"]
        totals["detections"] += m["n_detections"] or 0
        rate = video_done / (time.perf_counter() - t_start)
        eta_h = (total_s - video_done) / rate / 3600
        print(f"[{n}/{len(todo)}] {v.clip}: {m['n_samples']} samples, {m['n_motion_boxes']} motion boxes, "
              f"{m['n_detections']} detections, {m['seconds']:.1f} s ({m['video_s'] / m['seconds']:.0f}x realtime); "
              f"ETA {eta_h:.2f} h", flush=True)
    elapsed = time.perf_counter() - t_start
    print(f"index_pass: finished {len(todo)} clips in {elapsed / 3600:.2f} h", flush=True)
    if todo and not args.clips and args.out_root == REPO_ROOT / "artifacts":
        log_experiment("none", {"clips": len(todo), "video_h": total_s / 3600, "wall_h": elapsed / 3600,
                                "x_realtime": total_s / elapsed, **totals},
                       f"phase 3 index_pass (MOG2 + detection {cfg['detection'].get('model')}), settings {settings}",
                       stamp=stamp)
    print(f"outputs under {display_path(args.out_root)}")


if __name__ == "__main__":
    main()
