"""Step 4.2: appearance embeddings. Per clip: 1 fps candidates -> motion-aware dedup (rewind/embed/dedup.py) -> for
each kept frame, CLIP fp16 embeddings of the frame views (tile0, tile1, pad) and of the object crops
(rewind/embed/crops.py: detection crops and motion crops, stored separately so step 4.4 can compare crop_source
variants from one run). Preprocessing on the GPU, never a centre crop (rewind/embed/clip_model.py).

Outputs per (model, tag) and clip, under artifacts/embeddings/appearance/<model>__<tag>/<camera>/:
  <clip>.npy        (n, d) float16, L2-normalised (before any mean-centering, step 4.3)
  <clip>.parquet    one row per vector, same order: video_id, camera_id, modality, frame, t_s (= frame / fps =
                    proxy time), type (frame | crop), view (tile0 | tile1 | pad | ''), crop_source (detection |
                    motion | ''), crop_rank, dup, class, score, x1, y1, x2, y2 (frame view region or crop box before
                    padding, original pixels)
  <clip>.done.json  marker, written last: settings hash, counts, timing
Resumable: done clips are skipped; a marker with other settings stops the run. Any error stops the run.
Indexing reads footage of every split but no labels (CLAUDE.md 1.10 allows this; nothing here is tuned).

Run from the repo root:
  uv run python -m scripts.embed_appearance --clips <clip> --out-root <scratch>      trial on single clips
  uv run python -m scripts.embed_appearance --check-preprocess <clip>                GPU vs open_clip PIL preprocessing
  uv run python -m scripts.embed_appearance --models ViT-B-16:laion2b_s34b_b88k ... --tuning-only   (4.2 tag check)
  setsid nohup uv run python -m scripts.embed_appearance --pid-file results/logs/embed_appearance.pid \
      > results/logs/embed_appearance.log 2>&1 &                                       (full run, background)
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from rewind.embed.clip_model import FRAME_VIEWS, Clip, crop_square, frame_tensor, frame_view, square_region, view_box
from rewind.embed.crops import CropConfig, select_crops
from rewind.embed.dedup import Candidate, DedupConfig, Deduper, analysis_gray, boxes_mask, candidate_step
from rewind.embed.store import clip_paths
from rewind.eval.splits import split_video_ids
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.motion import store
from rewind.paths import REPO_ROOT, display_path
from rewind.video import sampled_frames
from scripts.index_pass import prefetch, settings_hash as index_settings_hash

FULL_RES = 10 ** 6
PREPROCESS_VERSION = "gpu-square-bicubic-aa-v1"   # bump when rewind/embed/clip_model.py preprocessing changes
META_COLUMNS = ["video_id", "camera_id", "modality", "frame", "t_s", "type", "view", "crop_source", "crop_rank",
                "dup", "class", "score", "x1", "y1", "x2", "y2"]


def settings_hash(cfg: dict, model: str, tag: str) -> str:
    """Everything that changes the vectors: model + tag, sampling/dedup, crop choice, frame views, preprocessing,
    and the index_pass settings that produced the motion boxes and detections. Not mean_center / crop_source
    (applied later, from these vectors)."""
    s = {k: v for k, v in cfg["sampling"].items() if k not in ("dedup_sweep", "dedup_min_recall")}
    a = {k: cfg["appearance"][k] for k in ("max_crops_per_frame", "crop_pad_frac", "min_crop_px", "crop_rank",
                                          "crop_moving_share", "crop_dup_cover")}
    blob = {"model": model, "pretrained": tag, "sampling": s, "appearance": a, "views": FRAME_VIEWS,
            "preprocess": PREPROCESS_VERSION, "index_pass": index_settings_hash(cfg)}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


def frame_inputs(v, size, cfg, art: Path):
    """Yield (meta rows, list of (3, s, s) float squares on the GPU) per KEPT frame of one clip, plus counts."""
    dcfg = DedupConfig.from_config(cfg)
    ccfg = CropConfig.from_config(cfg)
    pad_frac = cfg["appearance"]["crop_pad_frac"]
    use_det = cfg["detection"]["enabled"] is True
    store.load_marker(art, v)
    step = candidate_step(v.fps, dcfg.fps, cfg["motion"]["fps"])
    mot = pd.read_parquet(store.boxes_path(art, v))
    mot = {f: g[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, g in mot[mot.frame % step == 0].groupby("frame")}
    det = pd.read_parquet(store.detections_path(art, v)) if use_det else pd.DataFrame(columns=["frame"])
    det = {f: g for f, g in det[det.frame % step == 0].groupby("frame")}
    w, h = size
    dedup = Deduper(dcfg, use_det)
    counts = {"candidates": 0, "kept": 0}
    for frame, bgr in prefetch(sampled_frames(REPO_ROOT / v.path, v.n_frames, step, FULL_RES), 4):
        counts["candidates"] += 1
        d = det.get(frame)
        classes = d["class"].to_numpy(str) if d is not None else np.zeros(0, str)
        boxes = d[["x1", "y1", "x2", "y2"]].to_numpy(float) if d is not None else np.zeros((0, 4))
        scores = d["score"].to_numpy(float) if d is not None else np.zeros(0)
        mboxes = mot.get(frame, np.zeros((0, 4)))
        gray = analysis_gray(bgr, dcfg.analysis_width)
        cand = Candidate(frame, frame / v.fps, gray, boxes_mask(mboxes, size, (gray.shape[1], gray.shape[0])),
                         classes, boxes, scores)
        if not dedup.offer(cand):
            continue
        counts["kept"] += 1
        img = frame_tensor(bgr)
        base = (v.video_id, v.camera_id, v.modality, frame, frame / v.fps)
        rows, squares = [], []
        for view in FRAME_VIEWS:
            rows.append((*base, "frame", view, "", -1, False, "", float("nan"), *map(float, view_box(view, w, h))))
            squares.append(frame_view(img, view))
        for c in select_crops(classes, boxes, scores, mboxes, size, ccfg):
            x1, y1, side = crop_square(c.box, pad_frac)
            rows.append((*base, "crop", "", c.source, c.rank, c.dup, c.cls, c.score, *map(float, c.box)))
            squares.append(square_region(img, x1, y1, side))
        yield rows, squares
    counts["seconds_video"] = v.n_frames / v.fps
    return counts


def process_clip(v, size, cfg, models: list[tuple[str, str, Clip]], art: Path, out_root: Path, batch_size: int):
    t0 = time.perf_counter()
    rows, vecs, pending = [], {m: [] for m, _, _ in models}, []

    def flush():
        for m, _, clip in models:
            batch = torch.stack([clip.to_input(s) for s in pending])
            vecs[m].append(clip.encode_images(batch))
        pending.clear()

    gen = frame_inputs(v, size, cfg, art)
    counts = None
    while True:
        try:
            r, squares = next(gen)
        except StopIteration as stop:
            counts = stop.value
            break
        rows += r
        pending += squares
        if len(pending) >= batch_size:
            flush()
    if pending:
        flush()
    meta = pd.DataFrame(rows, columns=META_COLUMNS)
    elapsed = time.perf_counter() - t0
    for m, tag, clip in models:
        arr = np.concatenate(vecs[m]) if vecs[m] else np.zeros((0, 0), np.float16)
        if len(arr) != len(meta):
            raise RuntimeError(f"{v.clip}: {len(arr)} vectors for {len(meta)} metadata rows ({m})")
        paths = clip_paths(out_root, m.split("/")[0], tag, v)
        paths["npy"].parent.mkdir(parents=True, exist_ok=True)
        for key, write in (("npy", lambda p: np.save(p, arr)), ("meta", lambda p: meta.to_parquet(p, index=False))):
            tmp = paths[key].with_name(paths[key].name + ".tmp")
            with open(tmp, "wb") as f:
                write(f)
            tmp.rename(paths[key])
        marker = {"clip": v.clip, "video_id": v.video_id, "settings": settings_hash(cfg, m.split("/")[0], tag),
                  **counts, "vectors": len(meta), "seconds": round(elapsed, 2)}
        tmp = paths["done"].with_name(paths["done"].name + ".tmp")
        tmp.write_text(json.dumps(marker))
        tmp.rename(paths["done"])
    return counts | {"vectors": len(meta), "seconds": elapsed}


def check_preprocess(v, size, cfg, models) -> None:
    """Embed the tile0 square of the first 32 kept frames of a clip through our GPU path and through open_clip's
    own PIL transform; report input difference and embedding cosine."""
    from PIL import Image
    for m, tag, clip in models:
        ours, theirs = [], []
        for rows, squares in frame_inputs(v, size, cfg, REPO_ROOT / "artifacts"):
            sq = squares[0]                                   # tile0
            ours.append(clip.to_input(sq))
            pil = Image.fromarray(sq.permute(1, 2, 0).byte().cpu().numpy())
            theirs.append(clip.preprocess(pil).to(clip.device))
            if len(ours) == 32:
                break
        a, b = torch.stack(ours), torch.stack(theirs)
        za, zb = clip.encode_images(a).astype(np.float32), clip.encode_images(b).astype(np.float32)
        cos = (za * zb).sum(1)
        print(f"{m}: input |diff| mean {float((a - b).abs().mean()):.4f} max {float((a - b).abs().max()):.3f} "
              f"(normalised units); embedding cosine ours vs PIL: min {cos.min():.4f} mean {cos.mean():.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts", help="index_pass outputs")
    p.add_argument("--out-root", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--models", nargs="+", help="model:tag pairs (default: appearance.model:appearance.pretrained)")
    p.add_argument("--tuning-only", action="store_true", help="only tuning clips (4.2 pretrained-tag check)")
    p.add_argument("--clips", nargs="+", help="only these clips (trial; no experiments row)")
    p.add_argument("--check-preprocess", metavar="CLIP")
    p.add_argument("--batch-size", type=int, default=None, help="images per CLIP call (default appearance.batch_size)")
    p.add_argument("--pid-file", type=Path, help="write this process's PID here (background runs: the job itself, "
                                                 "not the uv launcher)")
    args = p.parse_args()
    if args.pid_file:
        args.pid_file.write_text(f"{os.getpid()}\n")
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    a = cfg["appearance"]
    pairs = [s.split(":") for s in args.models] if args.models else [[a["model"], a["pretrained"]]]
    for model, tag in pairs:
        if "TBD" in (model, tag):
            raise ValueError("appearance.model / appearance.pretrained is TBD: pass --models or set the config")
    batch_size = args.batch_size or a["batch_size"]

    conn = connect(args.db)
    videos = load_videos(conn)
    size_of = {r["video_id"]: (r["width"], r["height"]) for r in conn.execute("SELECT video_id, width, height FROM videos")}
    if args.tuning_only:
        ids = set(split_video_ids(conn, "tuning"))
        videos = [v for v in videos if v.video_id in ids]
    if args.clips or args.check_preprocess:
        want = set(args.clips or [args.check_preprocess])
        if want - {v.clip for v in videos}:
            raise ValueError(f"not in the manifest (or not tuning): {sorted(want - {v.clip for v in videos})}")
        videos = [v for v in videos if v.clip in want]

    todo, done = [], 0
    for v in videos:
        state = []
        for model, tag in pairs:
            path = clip_paths(args.out_root, model, tag, v)["done"]
            if path.exists():
                old = json.loads(path.read_text())
                if old["settings"] != settings_hash(cfg, model, tag):
                    raise ValueError(f"{path} has settings {old['settings']}, current "
                                     f"{settings_hash(cfg, model, tag)}: outputs of two settings would mix")
            state.append(path.exists())
        if all(state):
            done += 1
        elif any(state):
            raise ValueError(f"{v.clip}: done for some models only; run those models separately")
        else:
            todo.append(v)
    hours = sum(v.n_frames / v.fps for v in todo) / 3600
    print(f"embed_appearance: models {[':'.join(x) for x in pairs]}; {done} clips done, {len(todo)} to do "
          f"({hours:.2f} h of video); batch {batch_size}", flush=True)
    if not todo and not args.check_preprocess:
        return
    models = [(f"{model}/{tag}", tag, Clip(model, tag)) for model, tag in pairs]
    if args.check_preprocess:
        check_preprocess(videos[0], size_of[videos[0].video_id], cfg, models)
        return

    t_start, tot = time.perf_counter(), {"candidates": 0, "kept": 0, "vectors": 0, "seconds_video": 0.0}
    for n, v in enumerate(todo, 1):
        c = process_clip(v, size_of[v.video_id], cfg, models, args.art, args.out_root, batch_size)
        for k in tot:
            tot[k] += c[k]
        el = time.perf_counter() - t_start
        eta = el / tot["seconds_video"] * (hours * 3600 - tot["seconds_video"])
        print(f"[{n}/{len(todo)}] {v.clip}: kept {c['kept']}/{c['candidates']}, {c['vectors']} vectors, "
              f"{c['seconds']:.1f} s; elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min, "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB", flush=True)
    el = time.perf_counter() - t_start
    print(f"done: {tot}; {el / 60:.1f} min", flush=True)
    if not args.clips:
        log_experiment("none", {"clips": len(todo), "candidates": tot["candidates"], "kept_frames": tot["kept"],
                                "vectors_per_model": tot["vectors"], "minutes": el / 60,
                                "x_realtime": tot["seconds_video"] / el},
                       notes=f"step 4.2 embed_appearance {'tuning-only ' if args.tuning_only else ''}"
                             f"models {[':'.join(x) for x in pairs]}; outputs {display_path(args.out_root)}",
                       stamp=stamp)


if __name__ == "__main__":
    main()
