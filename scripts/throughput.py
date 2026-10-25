"""Step 1.4: decode and CLIP throughput on this machine, projected to the whole selection.

Decode: one hour of RGB video (12 clips, spread over the RGB cameras, seeded) and one hour of IR video, each decoded
sequentially from frame 0 (frame index = decoded count, 1.1 rule; every clip must give exactly n_frames frames) in
three ways: decode only; keep 1 frame per second (sampling.appearance_fps) as full-resolution RGB arrays, as the
appearance branch needs for crops; keep 5 frames per second (motion.fps) as RGB arrays scaled to
motion.downscale_width. Threading: PyAV's default is slice threading; frame threading is compared on one clip
(same frames, checked by hash) and the faster one is used.
CLIP: appearance.model in fp16 (open_clip mixed precision) on 1,000 real frames (1 fps from the RGB hour), with the
model's own preprocessing timed separately on the CPU, then the image encoder timed per batch size (host->GPU copy
included, after warm-up) with peak VRAM. Weights are left untrained (pretrained=None): appearance.pretrained is
still TBD, and the weights do not change the compute.
Writes results/reports/throughput.md and one row in results/experiments.csv (split 'none').

Run from the repo root (about 10 minutes; keep the machine otherwise idle):
  uv run python -m scripts.throughput
"""

import argparse
import hashlib
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import open_clip
import torch
import yaml
from PIL import Image

from rewind.experiment_log import log_experiment
from rewind.ingest.manifest import connect, load_videos

REPO_ROOT = Path(__file__).resolve().parent.parent


def pick_clips(videos: list, modality: str, n: int, seed: int) -> list:
    """n clips of one modality, taken round-robin over cameras from a seeded shuffle of each camera's clips."""
    by_camera = defaultdict(list)
    for v in sorted(videos, key=lambda v: v.clip):
        if v.modality == modality:
            by_camera[v.camera_id].append(v)
    rng = random.Random(seed)
    for clips in by_camera.values():
        rng.shuffle(clips)
    picked, cameras = [], sorted(by_camera)
    while len(picked) < n:
        if not any(by_camera.values()):
            raise ValueError(f"Only {len(picked)} {modality} clips available, asked for {n}")
        for cam in cameras:
            if by_camera[cam] and len(picked) < n:
                picked.append(by_camera[cam].pop())
    return picked


def keep_step(fps: float, rate: float) -> int:
    """Keep every k-th decoded frame to sample `rate` frames per second; k must be a whole number."""
    k = fps / rate
    if k != int(k) or k < 1:
        raise ValueError(f"{rate} fps is not a whole-frame step at {fps} fps")
    return int(k)


def decode(path: Path, n_frames: int, thread_type: str, step: int | None = None, width: int | None = None,
           on_keep=None) -> tuple[int, float]:
    """Decode every frame sequentially; every `step`-th frame (from frame 0) is converted to an RGB array (scaled
    to `width`, aspect kept, if given) and passed to on_keep(index, array). Returns (frames kept, seconds)."""
    kept, t0 = 0, time.perf_counter()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = thread_type
        height = None if width is None else 2 * round(stream.codec_context.height * width / stream.codec_context.width / 2)
        n = 0
        for n, frame in enumerate(container.decode(stream), start=1):
            if step and (n - 1) % step == 0:
                arr = frame.to_ndarray(format="rgb24", width=width, height=height)
                kept += 1
                if on_keep:
                    on_keep(n - 1, arr)
    seconds = time.perf_counter() - t0
    if n != n_frames:
        raise ValueError(f"{path.name}: decoded {n} frames, manifest says {n_frames}")
    return kept, seconds


def frame_hashes(path: Path, n: int, thread_type: str) -> list[str]:
    out = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = thread_type
        for frame in container.decode(stream):
            out.append(hashlib.md5(frame.to_ndarray(format="rgb24").tobytes()).hexdigest())
            if len(out) == n:
                break
    return out


def clip_benchmark(model_name: str, frames: list[np.ndarray], batch_sizes: list[int]) -> tuple[dict, list[dict]]:
    device = torch.device("cuda")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None, device=device,
                                                                 precision="fp16")
    model.eval()
    t0 = time.perf_counter()
    batch = torch.stack([preprocess(Image.fromarray(f)) for f in frames])
    prep_s = time.perf_counter() - t0
    prep = {"images": len(frames), "seconds": prep_s, "per_s": len(frames) / prep_s,
            "input": f"{frames[0].shape[1]}x{frames[0].shape[0]}", "transform": str(preprocess).replace("\n", " ")}
    rows = []
    with torch.inference_mode():
        for bs in batch_sizes:
            for i in range(0, 2 * bs, bs):  # warm-up: kernels and allocator for this batch size
                model.encode_image(batch[i % len(batch):i % len(batch) + bs].to(device, dtype=torch.float16))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            for i in range(0, len(batch), bs):
                emb = model.encode_image(batch[i:i + bs].to(device, dtype=torch.float16))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            free_b, total_b = torch.cuda.mem_get_info(device)
            rows.append({"batch": bs, "seconds": seconds, "per_s": len(batch) / seconds,
                         "peak_alloc_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                         "device_used_mib": (total_b - free_b) / 2**20, "total_mib": total_b / 2**20,
                         "dim": emb.shape[1]})
    return prep, rows


def hms(seconds: float) -> str:
    return f"{int(seconds // 3600)} h {int(seconds % 3600 // 60):02d} min"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "throughput.md")
    parser.add_argument("--clips", type=int, default=12, help="clips per modality (12 x 5 min = 1 h)")
    parser.add_argument("--clip-frames", type=int, default=1000, help="frames for the CLIP benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    app_fps, motion_fps, motion_w = cfg["sampling"]["appearance_fps"], cfg["motion"]["fps"], cfg["motion"]["downscale_width"]
    model_name = cfg["appearance"]["model"]
    if "TBD" in (app_fps, motion_fps, motion_w, model_name):
        raise ValueError("config values used here are TBD")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    videos = load_videos(connect(args.db))
    root = REPO_ROOT

    # Threading: same frames either way? Which is faster? (one RGB clip, decode only)
    probe = pick_clips(videos, "rgb", 1, args.seed)[0]
    same = frame_hashes(root / probe.path, 300, "SLICE") == frame_hashes(root / probe.path, 300, "AUTO")
    if not same:
        raise RuntimeError(f"{probe.clip}: frame threading gives different frames than slice threading")
    threading = {t: decode(root / probe.path, probe.n_frames, t)[1] for t in ("SLICE", "AUTO")}
    thread_type = min(threading, key=threading.get)
    print(f"threading on {probe.clip}: {threading} -> {thread_type} (first 300 frames identical: {same})")

    modes = [("decode only", None, None), (f"keep {app_fps} fps, full resolution", app_fps, None),
             (f"keep {motion_fps} fps, {motion_w} px wide", motion_fps, motion_w)]
    results = {}
    for modality in ("rgb", "ir"):
        clips = pick_clips(videos, modality, args.clips, args.seed)
        video_s = sum(v.n_frames / v.fps for v in clips)
        for label, rate, width in modes:
            seconds = kept = 0
            for v in clips:
                k, s = decode(root / v.path, v.n_frames, thread_type, keep_step(v.fps, rate) if rate else None, width)
                kept, seconds = kept + k, seconds + s
            results[(modality, label)] = {"video_s": video_s, "seconds": seconds, "kept": kept,
                                          "frames": sum(v.n_frames for v in clips), "clips": [v.clip for v in clips]}
            print(f"{modality} {label}: {video_s / 3600:.2f} h of video in {seconds:.1f} s "
                  f"({video_s / seconds:.0f}x realtime, {results[(modality, label)]['frames'] / seconds:.0f} frames/s)")

    # CLIP: 1 fps frames from the RGB clips, until clip_frames.
    frames = []
    for v in pick_clips(videos, "rgb", args.clips, args.seed):
        decode(root / v.path, v.n_frames, thread_type, keep_step(v.fps, app_fps), None,
               on_keep=lambda i, a: frames.append(a) if len(frames) < args.clip_frames else None)
        if len(frames) >= args.clip_frames:
            break
    prep, clip_rows = clip_benchmark(model_name, frames, args.batch_sizes)
    best = max(clip_rows, key=lambda r: r["per_s"])
    for r in clip_rows:
        print(f"CLIP batch {r['batch']:>3}: {r['per_s']:.0f} img/s, peak {r['peak_alloc_mib']:.0f} MiB allocated")
    print(f"preprocess (CPU, 1 process): {prep['per_s']:.0f} img/s")

    # Projection to the whole selection.
    total = {m: sum(v.n_frames / v.fps for v in videos if v.modality == m) for m in ("rgb", "ir")}
    n_1fps = sum(-(-v.n_frames // keep_step(v.fps, app_fps)) for v in videos)
    max_crops = cfg["appearance"]["max_crops_per_frame"]
    proj = {}
    for label, _, _ in modes:
        proj[label] = sum(total[m] / (results[(m, label)]["video_s"] / results[(m, label)]["seconds"]) for m in total)
    clip_frames_s = n_1fps / best["per_s"]
    clip_crops_s = n_1fps * (1 + max_crops) / best["per_s"]
    prep_frames_s = n_1fps / prep["per_s"]

    lines = [
        "# Throughput (step 1.4)", "",
        f"Generated by `uv run python -m scripts.throughput` (seed {args.seed}). Machine: {platform.processor() or 'x86_64'}, "
        f"{torch.cuda.get_device_name(0)}, {torch.cuda.mem_get_info()[1] / 2**30:.1f} GiB VRAM. torch {torch.__version__}, "
        f"open_clip {open_clip.__version__}, PyAV {av.__version__}. Projection basis: the whole selection, "
        f"{total['rgb'] / 3600:.2f} h RGB (1920x1080) + {total['ir'] / 3600:.2f} h IR (352x240) = "
        f"{(total['rgb'] + total['ir']) / 3600:.2f} h.", "",
        "## Decoding (CPU, sequential from frame 0)", "",
        f"Threading compared on {probe.clip}, decode only: slice {threading['SLICE']:.1f} s, frame (AUTO) "
        f"{threading['AUTO']:.1f} s; the first 300 frames are identical (hash) → **{thread_type}** used below. "
        "Every clip decoded to exactly its manifest frame count.", "",
        "| modality | mode | video | wall time | × realtime | decoded frames/s | kept frames |",
        "|---|---|---|---|---|---|---|"]
    for (m, label), r in results.items():
        lines.append(f"| {m} | {label} | {r['video_s'] / 3600:.2f} h | {r['seconds']:.0f} s | "
                     f"{r['video_s'] / r['seconds']:.0f}× | {r['frames'] / r['seconds']:.0f} | {r['kept']} |")
    lines += ["", "Clips: RGB " + ", ".join(results[("rgb", "decode only")]["clips"]) + "; IR "
              + ", ".join(results[("ir", "decode only")]["clips"]) + ".", "",
              "## CLIP image encoder (GPU)", "",
              f"`{model_name}`, open_clip `precision=\"fp16\"` (mixed precision), untrained weights (same compute as any "
              f"tag), {len(frames)} real 1 fps frames, host→GPU copy included, after 2 warm-up batches.", "",
              "| batch | img/s | wall time | peak allocated (torch) | device memory in use |", "|---|---|---|---|---|"]
    for r in clip_rows:
        lines.append(f"| {r['batch']} | {r['per_s']:.0f} | {r['seconds']:.2f} s | {r['peak_alloc_mib']:.0f} MiB | "
                     f"{r['device_used_mib']:.0f} / {r['total_mib']:.0f} MiB |")
    lines += ["", f"Fastest: batch **{best['batch']}**, {best['per_s']:.0f} img/s. Embedding dim {best['dim']}.", "",
              f"Preprocessing ({prep['transform']}) on the CPU in one process: **{prep['per_s']:.0f} img/s** "
              f"for {prep['input']} frames.", "",
              "## Projected times for the whole selection", "",
              "| job | projected wall time |", "|---|---|"]
    for label in proj:
        lines.append(f"| decode, {label} | {hms(proj[label])} |")
    lines += [f"| CLIP encoder, full frames only ({n_1fps:,} frames at {app_fps} fps, before dedup) | {hms(clip_frames_s)} |",
              f"| CLIP encoder, full frame + {max_crops} crops (upper bound, {n_1fps * (1 + max_crops):,} images) | {hms(clip_crops_s)} |",
              f"| CLIP preprocessing, full frames, 1 CPU process | {hms(prep_frames_s)} |", ""]
    text = "\n".join(lines)

    metrics = {"decode_thread_auto": float(thread_type == "AUTO"),
               "clip_best_batch": float(best["batch"]), "clip_img_per_s": best["per_s"],
               "clip_peak_mib": best["peak_alloc_mib"], "preprocess_img_per_s": prep["per_s"],
               "proj_clip_frames_h": clip_frames_s / 3600, "proj_clip_with_crops_h": clip_crops_s / 3600}
    for (m, label), r in results.items():
        key = label.split(",")[0].replace(" ", "_")
        metrics[f"decode_{m}_{key}_x_realtime"] = r["video_s"] / r["seconds"]
    for label, s in proj.items():
        metrics[f"proj_decode_{label.split(',')[0].replace(' ', '_')}_h"] = s / 3600
    row = log_experiment("none", metrics, notes=f"step 1.4 throughput, seed {args.seed}, {args.clips} clips/modality, "
                         f"{len(frames)} CLIP frames, untrained {model_name} fp16")
    text += f"\nexperiments.csv row: {row['timestamp']}, git {row['git_hash']}, config {row['config_hash'][:12]}.\n"
    args.report.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
