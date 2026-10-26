"""Step 6.2: crop and CLIP-embed the best box of every detected vehicle event (SQLite `detected_events`).

Per clip with events: decode the original once (rewind/video.frames_at, only the best-box frames converted), cut
each event's padded square crop exactly like appearance crops (rewind/embed/clip_model.crop_square +
square_region, appearance.crop_pad_frac), embed it with the appearance model (appearance.model / pretrained, fp16,
the same GPU preprocessing), and save the crop as a JPEG thumbnail (side <= events.crop_thumb_px) under
artifacts/event_crops/<camera>/<det_event_id>.jpg; crop_path is written to SQLite. Small crops are kept (an event
is evidence even if its vehicle is far away); their side in pixels is stored.

Outputs per clip, under artifacts/embeddings/events/<model>__<tag>/<camera>/:
  <clip>.npy (n, d) float16 L2-normalised; <clip>.parquet (det_event_id, video_id, camera_id, modality, event_type,
  t_s, crop_side_px), same order; <clip>.done.json marker (written last; settings hash).
Resumable per clip; a marker with other settings stops the run. Indexing: every clip, no labels read.

Run from the repo root (background, CLAUDE.md rule 16):
  setsid nohup uv run python -m scripts.embed_events --pid-file results/logs/embed_events.pid \
      > results/logs/embed_events.log 2>&1 &
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from rewind.embed.clip_model import Clip, crop_square, frame_tensor, square_region
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT, display_path
from rewind.video import frames_at
from scripts.embed_appearance import PREPROCESS_VERSION

META_COLUMNS = ["det_event_id", "video_id", "camera_id", "modality", "event_type", "t_s", "crop_side_px"]


def out_dir(art: Path, model: str, tag: str) -> Path:
    return art / "embeddings" / "events" / f"{model}__{tag}"


def settings_hash(cfg: dict, rules_settings: str) -> str:
    blob = {"rules": rules_settings, "model": cfg["appearance"]["model"], "tag": cfg["appearance"]["pretrained"],
            "pad": cfg["appearance"]["crop_pad_frac"], "thumb": cfg["events"]["crop_thumb_px"],
            "preprocess": PREPROCESS_VERSION}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


def rules_settings_of(conn) -> str:
    s = {r[0] for r in conn.execute("SELECT DISTINCT settings FROM detected_events")}
    if len(s) != 1:
        raise ValueError(f"detected_events has {len(s)} settings: run scripts.build_vehicle_events")
    return s.pop()


def load_event_vectors(art: Path, cfg: dict, conn, videos: list) -> tuple[np.ndarray, pd.DataFrame]:
    """Vectors + metadata of the detected events of these videos; every event in SQLite must have its vector."""
    root = out_dir(art, cfg["appearance"]["model"], cfg["appearance"]["pretrained"])
    settings = settings_hash(cfg, rules_settings_of(conn))
    want = {r[0]: r[1] for r in conn.execute("SELECT video_id, COUNT(*) FROM detected_events GROUP BY video_id")}
    vecs, metas = [], []
    for v in videos:
        if v.video_id not in want:
            continue
        marker = root / v.camera_id / f"{v.clip}.done.json"
        if not marker.exists() or json.loads(marker.read_text())["settings"] != settings:
            raise FileNotFoundError(f"{v.clip}: no event vectors with settings {settings} ({marker})")
        x = np.load(root / v.camera_id / f"{v.clip}.npy")
        m = pd.read_parquet(root / v.camera_id / f"{v.clip}.parquet")
        if len(x) != len(m) or len(m) != want[v.video_id]:
            raise ValueError(f"{v.clip}: {len(x)} vectors / {len(m)} rows / {want[v.video_id]} events in SQLite")
        vecs.append(x)
        metas.append(m)
    if not vecs:
        raise ValueError("no detected events in these videos")
    return np.concatenate(vecs), pd.concat(metas, ignore_index=True)


def thumbnail(square: torch.Tensor, max_px: int) -> np.ndarray:
    """(3, s, s) float RGB 0..255 on the GPU -> BGR uint8 image with side <= max_px (area downscaling)."""
    img = square.clamp(0, 255).round().byte().permute(1, 2, 0).flip(-1).cpu().numpy()
    side = img.shape[0]
    if side > max_px:
        img = cv2.resize(img, (max_px, max_px), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def process_clip(v, events: pd.DataFrame, cfg: dict, clip: Clip, art: Path, root: Path, settings: str, conn) -> dict:
    t0 = time.perf_counter()
    pad = cfg["appearance"]["crop_pad_frac"]
    thumb_px = cfg["events"]["crop_thumb_px"]
    by_frame = {f: g for f, g in events.groupby("box_frame")}
    rows, inputs, paths = [], [], []
    for frame, bgr in frames_at(REPO_ROOT / v.path, v.n_frames, set(by_frame)):
        img = frame_tensor(bgr)
        for e in by_frame[frame].itertuples(index=False):
            x1, y1, side = crop_square((e.x1, e.y1, e.x2, e.y2), pad)
            sq = square_region(img, x1, y1, side)
            path = art / "event_crops" / v.camera_id / f"{e.det_event_id}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), thumbnail(sq, thumb_px)):
                raise OSError(f"could not write {path}")
            rows.append((e.det_event_id, v.video_id, v.camera_id, v.modality, e.event_type, e.t_s, side))
            inputs.append(clip.to_input(sq))
            paths.append((str(path.relative_to(REPO_ROOT)), e.det_event_id))
    if len(rows) != len(events):
        raise ValueError(f"{v.clip}: {len(rows)} crops for {len(events)} events")
    bs = cfg["appearance"]["batch_size"]
    x = np.concatenate([clip.encode_images(torch.stack(inputs[i:i + bs])) for i in range(0, len(inputs), bs)])
    order = np.argsort([r[0] for r in rows], kind="stable")      # rows in det_event_id order
    x = x[order]
    meta = pd.DataFrame([rows[i] for i in order], columns=META_COLUMNS)
    d = root / v.camera_id
    d.mkdir(parents=True, exist_ok=True)
    for name, write in ((f"{v.clip}.npy", lambda p: np.save(p, x)),
                        (f"{v.clip}.parquet", lambda p: meta.to_parquet(p, index=False))):
        tmp = d / (name + ".tmp")
        with open(tmp, "wb") as f:
            write(f)
        tmp.rename(d / name)
    with conn:
        conn.executemany("UPDATE detected_events SET crop_path = ? WHERE det_event_id = ?", paths)
    marker = {"clip": v.clip, "settings": settings, "n_events": len(rows), "seconds": round(time.perf_counter() - t0, 1)}
    tmp = d / f"{v.clip}.done.json.tmp"
    tmp.write_text(json.dumps(marker))
    tmp.rename(d / f"{v.clip}.done.json")
    return marker


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    ap.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    ap.add_argument("--pid-file", type=Path)
    args = ap.parse_args()
    if args.pid_file:
        args.pid_file.write_text(f"{os.getpid()}\n")
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    art = REPO_ROOT / "artifacts"
    conn = connect(args.db)
    settings = settings_hash(cfg, rules_settings_of(conn))
    root = out_dir(art, cfg["appearance"]["model"], cfg["appearance"]["pretrained"])
    events = pd.read_sql_query("SELECT * FROM detected_events ORDER BY det_event_id", conn)
    videos = [v for v in load_videos(conn) if v.video_id in set(events["video_id"])]
    todo = []
    for v in videos:
        marker = root / v.camera_id / f"{v.clip}.done.json"
        if marker.exists():
            if json.loads(marker.read_text())["settings"] != settings:
                raise ValueError(f"{marker}: other settings than {settings}; remove the old outputs")
            continue
        todo.append(v)
    print(f"embed_events: {len(events)} events in {len(videos)} clips; {len(videos) - len(todo)} done, "
          f"{len(todo)} to do; settings {settings}", flush=True)
    if not todo:
        return
    clip = Clip(cfg["appearance"]["model"], cfg["appearance"]["pretrained"])
    t_start, n_ev = time.perf_counter(), 0
    for n, v in enumerate(todo, 1):
        m = process_clip(v, events[events["video_id"] == v.video_id], cfg, clip, art, root, settings, conn)
        n_ev += m["n_events"]
        el = time.perf_counter() - t_start
        print(f"[{n}/{len(todo)}] {v.clip}: {m['n_events']} events, {m['seconds']:.1f} s; elapsed {el / 60:.1f} min, "
              f"ETA {el / n * (len(todo) - n) / 60:.1f} min", flush=True)
    el = time.perf_counter() - t_start
    print(f"done: {n_ev} events in {el / 60:.1f} min", flush=True)
    log_experiment("none", {"clips": len(todo), "events": n_ev, "minutes": el / 60},
                   f"step 6.2 embed_events ({cfg['appearance']['model']}/{cfg['appearance']['pretrained']}); "
                   f"outputs {display_path(root)}", stamp=stamp)


if __name__ == "__main__":
    main()
