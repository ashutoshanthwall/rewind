"""Steps 5.1 / 5.2: action-branch window embeddings with LanguageBind-Video (rewind/embed/languagebind.py).

Per clip: windows of rewind/embed/windows.py (action.window_s, stride action.stride_s), action.frames_per_window frames
each (decoded on the fly at frames_per_window / window_s fps, every frame preprocessed once and shared by the windows
that use it), embedded action.batch_windows windows per model call.

Outputs under artifacts/embeddings/action/<model key>/<camera>/:
  <clip>.npy        (n_windows, dim) float16, L2-normalised
  <clip>.parquet    video_id, camera_id, modality, window_start, t_s (= window centre, clip seconds = proxy time)
  <clip>.done.json  marker, written last: settings hash, windows, seconds (end to end), model_seconds, peak_vram_mib
Resumable: done clips are skipped; a marker with other settings stops the run. Any error stops the run.
Indexing reads footage of every split but no labels (CLAUDE.md 1.10).

Modes:
  --sanity            pre-registered implementation check on tuning clips (PROGRESS.md, step 5.1): fp16 vs fp32
                      embeddings, and the AUC of vehicle start/stop prompts vs annotated windows, next to
                      mean-pooled CLIP. Prints PASS / FAIL; writes nothing under artifacts/.
  --encode-queries S  text vectors of every frozen query of split S (tuning | dev) -> <model dir>/text_vectors_<S>.npz
  (default)           embed clips: --split S for one split, --clips for single clips, else all footage

Run from the repo root:
  uv run python -m scripts.embed_action --sanity
  uv run python -m scripts.embed_action --split dev && uv run python -m scripts.embed_action --encode-queries dev
  nohup uv run python -m scripts.embed_action --pid-file results/logs/embed_action.pid \\
      > results/logs/embed_action.log 2>&1 &                                        (step 5.2, all footage)
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

from rewind.embed.languagebind import LanguageBind
from rewind.embed.windows import pooled_windows, window_frames, window_starts
from rewind.eval.metrics import auc
from rewind.eval.queries import load_queries
from rewind.eval.splits import split_video_ids
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT, display_path
from rewind.video import keep_step, sampled_frames
from scripts.index_pass import prefetch
from scripts.run_baselines import QUERIES_DIR

FULL_RES = 10 ** 6
PREPROCESS_VERSION = "pad-bicubic-aa-openai-norm-v1"   # bump when rewind/embed/languagebind.py preprocessing changes
WINDOW_COLUMNS = ["video_id", "camera_id", "modality", "window_start", "t_s"]


def model_key(cfg: dict) -> str:
    return Path(cfg["action"]["languagebind"]["weights_dir"]).name


def out_dir(root: Path, cfg: dict) -> Path:
    return root / "embeddings" / "action" / model_key(cfg)


def settings_hash(cfg: dict) -> str:
    act = cfg["action"]
    blob = {"model": model_key(cfg), "revision": act["languagebind"]["revision"],
            "weights_sha256": act["languagebind"]["weights_sha256"], "preprocess": PREPROCESS_VERSION,
            **{k: act[k] for k in ("frames_per_window", "window_s", "stride_s")}}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


def load_model(cfg: dict, dtype=torch.float16) -> LanguageBind:
    lb = cfg["action"]["languagebind"]
    lbm = LanguageBind(REPO_ROOT / lb["weights_dir"], dtype=dtype)
    if lbm.num_frames != cfg["action"]["frames_per_window"]:
        raise ValueError(f"model takes {lbm.num_frames} frames, action.frames_per_window is "
                         f"{cfg['action']['frames_per_window']}")
    return lbm


def clip_windows(lbm: LanguageBind, v, cfg: dict, batch: int, keep_frames: frozenset = frozenset()):
    """Window vectors of one clip -> (vectors fp16, meta, timing, {frame index: BGR frame} for keep_frames)."""
    act = cfg["action"]
    t0 = time.perf_counter()
    starts = window_starts(v.n_frames / v.fps, act["window_s"], act["stride_s"])
    frames = window_frames(starts, act["window_s"], act["frames_per_window"], v.fps)
    step = keep_step(v.fps, act["frames_per_window"] / act["window_s"])
    if (frames % step).any() or frames.max() >= v.n_frames:
        raise ValueError(f"{v.clip}: window frames are not on the {step}-frame decode grid inside the clip")
    need = set(frames.ravel().tolist())
    inputs, bgr = {}, {}
    for idx, frame in prefetch(sampled_frames(REPO_ROOT / v.path, v.n_frames, step, FULL_RES), 4):
        if idx in need:
            inputs[idx] = lbm.frame_input(frame)
            if idx in keep_frames:
                bgr[idx] = frame
    if need - set(inputs):
        raise ValueError(f"{v.clip}: {len(need - set(inputs))} window frames were not decoded")
    model_s, vecs = 0.0, []
    for i in range(0, len(frames), batch):
        x = torch.stack([torch.stack([inputs[f] for f in row]) for row in frames[i:i + batch].tolist()])
        torch.cuda.synchronize()
        tm = time.perf_counter()
        vecs.append(lbm.encode_windows(x))
        torch.cuda.synchronize()
        model_s += time.perf_counter() - tm
    meta = pd.DataFrame({"video_id": v.video_id, "camera_id": v.camera_id, "modality": v.modality,
                         "window_start": starts, "t_s": starts + act["window_s"] / 2})[WINDOW_COLUMNS]
    x = np.concatenate(vecs) if vecs else np.zeros((0, lbm.dim), np.float16)
    return x, meta, {"seconds": time.perf_counter() - t0, "model_seconds": model_s}, bgr


def write_clip(root: Path, cfg: dict, v, x: np.ndarray, meta: pd.DataFrame, timing: dict) -> None:
    d = out_dir(root, cfg) / v.camera_id
    d.mkdir(parents=True, exist_ok=True)
    paths = {"npy": d / f"{v.clip}.npy", "meta": d / f"{v.clip}.parquet", "done": d / f"{v.clip}.done.json"}
    for key, write in (("npy", lambda f: np.save(f, x)), ("meta", lambda f: meta.to_parquet(f, index=False))):
        tmp = paths[key].with_name(paths[key].name + ".tmp")
        with open(tmp, "wb") as f:
            write(f)
        tmp.rename(paths[key])
    marker = {"clip": v.clip, "video_id": v.video_id, "settings": settings_hash(cfg), "windows": len(x),
              "seconds_video": v.n_frames / v.fps, "seconds": round(timing["seconds"], 2),
              "model_seconds": round(timing["model_seconds"], 2),
              "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20)}
    tmp = paths["done"].with_name(paths["done"].name + ".tmp")
    tmp.write_text(json.dumps(marker))
    tmp.rename(paths["done"])


def sanity(cfg: dict, conn, art: Path) -> None:
    """Pre-registered implementation check (PROGRESS.md step 5.1); tuning clips only."""
    from rewind.embed.clip_model import Clip
    from scripts.action_bakeoff import clip_pooled

    act = cfg["action"]
    s = act["sanity"]
    tuning = set(split_video_ids(conn, "tuning"))
    videos = {v.video_id: v for v in load_videos(conn) if v.video_id in tuning and v.modality == "rgb"}
    types = list(s["prompts"])
    ev = pd.read_sql_query(
        f"SELECT video_id, activity_type, start_s, end_s FROM events WHERE label_source != 'contrib' AND "
        f"activity_type IN ({','.join('?' * len(types))})", conn, params=types)
    ev = ev[ev.video_id.isin(videos)]
    counts = ev.groupby("video_id").size().reset_index(name="n").sort_values(["n", "video_id"],
                                                                              ascending=[False, True])
    chosen = [videos[i] for i in counts.video_id.head(s["n_clips"]).tolist()]
    print("sanity clips:", [(v.clip, int(n)) for v, n in zip(chosen, counts.n.head(s["n_clips"]))], flush=True)

    lbm = load_model(cfg)
    print("(1) weights loaded: every learned tensor matched, no unexpected keys (only the constant text position_ids "
          "buffer is absent from the checkpoint; checked = 0..76)", flush=True)
    # (2) uses the first fp16_windows windows of the first clip: keep their decoded frames for the fp32 model
    frames16 = window_frames(window_starts(chosen[0].n_frames / chosen[0].fps, act["window_s"], act["stride_s"])
                             [:s["fp16_windows"]], act["window_s"], act["frames_per_window"], chosen[0].fps)
    xs, metas, bgr0 = [], [], None
    for k, v in enumerate(chosen):
        x, m, timing, bgr = clip_windows(lbm, v, cfg, act["batch_windows"],
                                         frozenset(frames16.ravel().tolist()) if k == 0 else frozenset())
        xs.append(x)
        metas.append(m)
        if k == 0:
            bgr0 = bgr
        print(f"  {v.clip}: {len(x)} windows in {timing['seconds']:.1f} s (model {timing['model_seconds']:.1f} s), "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB", flush=True)
    z_b = lbm.encode_text([s["prompts"][t] for t in types])

    # (2) fp16 vs fp32 on the first fp16_windows windows of the first clip
    x16 = xs[0][:s["fp16_windows"]].astype(np.float32)
    del lbm
    torch.cuda.empty_cache()
    lb32 = load_model(cfg, dtype=torch.float32)
    x32 = lb32.encode_windows(torch.stack([torch.stack([lb32.frame_input(bgr0[f]) for f in row])
                                           for row in frames16.tolist()])).astype(np.float32)
    cos = (x16 * x32).sum(1) / np.linalg.norm(x16, axis=1) / np.linalg.norm(x32, axis=1)
    del lb32
    torch.cuda.empty_cache()
    ok2 = bool(cos.min() >= s["fp16_min_cos"])
    print(f"(2) fp16 vs fp32 cosine over {len(cos)} windows: min {cos.min():.4f}, mean {cos.mean():.4f} "
          f"-> {'ok' if ok2 else 'FAIL'} (>= {s['fp16_min_cos']})", flush=True)

    # (3) AUC: windows overlapping an annotated event of the type vs the rest, (b) next to (a)
    xa, ma = clip_pooled(cfg, chosen, art)
    clip = Clip(cfg["appearance"]["model"], cfg["appearance"]["pretrained"])
    z_a = clip.encode_text([s["prompts"][t] for t in types])
    mb = pd.concat(metas, ignore_index=True)
    if not ma[["video_id", "window_start"]].equals(mb[["video_id", "window_start"]]):
        raise ValueError("pooled CLIP and LanguageBind windows differ")
    xb = np.concatenate(xs).astype(np.float32)
    ok3 = True
    for i, t in enumerate(types):
        lab = np.zeros(len(mb), bool)
        for e in ev[ev.activity_type == t].itertuples():
            lab |= ((mb.video_id == e.video_id) & (mb.window_start < e.end_s)
                    & (mb.window_start + act["window_s"] > e.start_s)).to_numpy()
        auc_b, auc_a = auc(xb @ z_b[i], lab), auc(xa @ z_a[i], lab)
        ok3 &= auc_b > s["min_auc"]
        print(f"(3) {t!r} prompt {s['prompts'][t]!r}: {lab.sum()} / {len(lab)} windows positive; AUC (b) "
              f"{auc_b:.3f}, (a) mean-pooled CLIP {auc_a:.3f}", flush=True)
    print(f"(3) -> {'ok' if ok3 else 'FAIL'} (b) AUC > {s['min_auc']} on every prompt")
    print("SANITY:", "PASS" if ok2 and ok3 else "FAIL")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts", help="appearance index (sanity check)")
    p.add_argument("--out-root", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--split", choices=("tuning", "dev"), help="only this split's clips (footage, no labels)")
    p.add_argument("--clips", nargs="+", help="only these clips (trial; no experiments row)")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--encode-queries", choices=("tuning", "dev"))
    p.add_argument("--batch-windows", type=int, help="windows per model call (default action.batch_windows)")
    p.add_argument("--pid-file", type=Path)
    args = p.parse_args()
    if args.pid_file:
        args.pid_file.write_text(f"{os.getpid()}\n")
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    conn = connect(args.db)
    if args.sanity:
        sanity(cfg, conn, args.art)
        return
    if args.encode_queries:
        qs = load_queries(QUERIES_DIR, args.encode_queries)
        texts = sorted({q["query"] for q in qs})
        z = load_model(cfg).encode_text(texts)
        path = out_dir(args.out_root, cfg) / f"text_vectors_{args.encode_queries}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, texts=np.array(texts), vectors=z, settings=settings_hash(cfg))
        print(f"{len(texts)} query texts -> {display_path(path)}")
        return

    batch = args.batch_windows or cfg["action"]["batch_windows"]
    videos = load_videos(conn)
    if args.split:
        ids = set(split_video_ids(conn, args.split))
        videos = [v for v in videos if v.video_id in ids]
    if args.clips:
        if set(args.clips) - {v.clip for v in videos}:
            raise ValueError(f"not in the manifest: {sorted(set(args.clips) - {v.clip for v in videos})}")
        videos = [v for v in videos if v.clip in args.clips]
    todo, done = [], 0
    for v in videos:
        marker = out_dir(args.out_root, cfg) / v.camera_id / f"{v.clip}.done.json"
        if marker.exists():
            old = json.loads(marker.read_text())["settings"]
            if old != settings_hash(cfg):
                raise ValueError(f"{marker} has settings {old}, current {settings_hash(cfg)}")
            done += 1
        else:
            todo.append(v)
    hours = sum(v.n_frames / v.fps for v in todo) / 3600
    print(f"embed_action: {model_key(cfg)} settings {settings_hash(cfg)}; {done} clips done, {len(todo)} to do "
          f"({hours:.2f} h of video); batch {batch} windows", flush=True)
    if not todo:
        return
    lbm = load_model(cfg)
    t_start, tot = time.perf_counter(), {"windows": 0, "seconds_video": 0.0, "model_seconds": 0.0}
    for n, v in enumerate(todo, 1):
        x, meta, timing, _ = clip_windows(lbm, v, cfg, batch)
        write_clip(args.out_root, cfg, v, x, meta, timing)
        tot["windows"] += len(x)
        tot["seconds_video"] += v.n_frames / v.fps
        tot["model_seconds"] += timing["model_seconds"]
        el = time.perf_counter() - t_start
        eta = el / tot["seconds_video"] * (hours * 3600 - tot["seconds_video"])
        print(f"[{n}/{len(todo)}] {v.clip}: {len(x)} windows, {timing['seconds']:.1f} s (model "
              f"{timing['model_seconds']:.1f} s); elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min, "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB", flush=True)
    el = time.perf_counter() - t_start
    print(f"done: {tot}; {el / 60:.1f} min", flush=True)
    if not args.clips:
        log_experiment("none", {"clips": len(todo), "windows": tot["windows"], "minutes": el / 60,
                                "windows_per_s": tot["windows"] / el, "x_realtime": tot["seconds_video"] / el,
                                "peak_vram_mib": torch.cuda.max_memory_allocated() / 2 ** 20},
                       notes=f"step 5.x embed_action {model_key(cfg)} {args.split or 'all footage'}; settings "
                             f"{settings_hash(cfg)}", stamp=stamp)


if __name__ == "__main__":
    main()
