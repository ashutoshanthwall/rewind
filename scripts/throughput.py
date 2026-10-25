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
import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import av
import open_clip
import torch
import yaml
from PIL import Image

from rewind.experiment_log import DEFAULT_CSV, log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.video import keep_step

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


def decode(path: Path, n_frames: int, thread_type: str, step: int | None = None, width: int | None = None,
           on_keep=None) -> tuple[int, float]:
    """Decode every frame sequentially; every `step`-th frame (from frame 0) is converted to an RGB array (scaled
    down to `width`, aspect kept, if given; never scaled up) and passed to on_keep(index, array).
    Returns (frames kept, seconds)."""
    kept, t0 = 0, time.perf_counter()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = thread_type
        if width is not None and width >= stream.codec_context.width:
            width = None  # e.g. IR 352 px wide: keep native size
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


def load_clip(model_name: str):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None, device="cuda",
                                                                 precision="fp16")
    return model.eval(), preprocess


def clip_benchmark(model, batch: torch.Tensor, batch_sizes: list[int]) -> list[dict]:
    """Image-encoder throughput per batch size on preprocessed images (host->GPU copy included)."""
    device = torch.device("cuda")
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
    return rows


def hms(seconds: float) -> str:
    m = round(seconds / 60)
    return f"{m // 60} h {m % 60:02d} min"


NOTES_PREFIX = "step 1.4 throughput"
# Metric names of earlier versions of this script for the same quantity. The IR 5 fps mode is left out on purpose:
# before git 3f50859 it scaled IR frames up to 640 px, so its old values are not comparable.
OLD_KEYS = {"decode_rgb_keep_1fps_full_x_realtime": "decode_rgb_keep_1_fps_x_realtime",
            "decode_rgb_keep_5fps_max640_x_realtime": "decode_rgb_keep_5_fps_x_realtime",
            "decode_ir_keep_1fps_full_x_realtime": "decode_ir_keep_1_fps_x_realtime"}


def history(csv_path: Path, keys: list[str]) -> list[dict]:
    """Rows of this benchmark in experiments.csv: {'timestamp', 'git', key: value or None}."""
    out = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if not row["notes"].startswith(NOTES_PREFIX):
                continue
            m = json.loads(row["metrics"])
            out.append({"timestamp": row["timestamp"], "git": row["git_hash"][:7],
                        **{k: m.get(k, m.get(OLD_KEYS.get(k, ""))) for k in keys}})
    return out


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("no 'model name' in /proc/cpuinfo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "throughput.md")
    parser.add_argument("--clips", type=int, default=12, help="clips per modality (12 x 5 min = 1 h)")
    parser.add_argument("--clip-frames", type=int, default=1000, help="frames for the CLIP benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--hash-frames", type=int, default=300, help="frames compared between threading modes")
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()
    load_start = os.getloadavg()
    stamp = run_stamp(args.config)  # code + settings at the start (see run_stamp)

    cfg = yaml.safe_load(args.config.read_text())
    app_fps, motion_fps, motion_w = cfg["sampling"]["appearance_fps"], cfg["motion"]["fps"], cfg["motion"]["downscale_width"]
    model_name = cfg["appearance"]["model"]
    if "TBD" in (app_fps, motion_fps, motion_w, model_name):
        raise ValueError("config values used here are TBD")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    conn = connect(args.db)
    videos = load_videos(conn)
    gb_per_h = {m: gb for m, gb in conn.execute(
        "SELECT modality, SUM(size_bytes) / 1e9 / (SUM(duration_s) / 3600) FROM videos GROUP BY modality")}
    root = REPO_ROOT

    # Threading: same frames either way? Which is faster? (one RGB clip, decode only)
    probe = pick_clips(videos, "rgb", 1, args.seed)[0]
    same = frame_hashes(root / probe.path, args.hash_frames, "SLICE") == frame_hashes(root / probe.path, args.hash_frames, "AUTO")
    if not same:
        raise RuntimeError(f"{probe.clip}: frame threading gives different frames than slice threading")
    threading = {t: decode(root / probe.path, probe.n_frames, t)[1] for t in ("SLICE", "AUTO")}
    thread_type = min(threading, key=threading.get)
    print(f"threading on {probe.clip}: {threading} -> {thread_type} (first {args.hash_frames} frames identical)", flush=True)

    modes = [("decode_only", "decode only", None, None),
             ("keep_1fps_full", f"keep {app_fps} fps, full resolution", app_fps, None),
             (f"keep_{motion_fps}fps_max{motion_w}", f"keep {motion_fps} fps, at most {motion_w} px wide", motion_fps, motion_w)]
    results = {}
    for modality in ("rgb", "ir"):
        clips = pick_clips(videos, modality, args.clips, args.seed)
        video_s = sum(v.n_frames / v.fps for v in clips)
        for key, label, rate, width in modes:
            seconds = kept = 0
            for v in clips:
                k, s = decode(root / v.path, v.n_frames, thread_type, keep_step(v.fps, rate) if rate else None, width)
                kept, seconds = kept + k, seconds + s
            results[(modality, key)] = {"label": label, "video_s": video_s, "seconds": seconds, "kept": kept,
                                        "x": video_s / seconds, "frames": sum(v.n_frames for v in clips),
                                        "clips": [v.clip for v in clips]}
            print(f"{modality} {label}: {video_s / 3600:.2f} h of video in {seconds:.1f} s "
                  f"({video_s / seconds:.0f}x realtime, {results[(modality, key)]['frames'] / seconds:.0f} frames/s)", flush=True)

    # CLIP: 1 fps frames from the RGB clips, preprocessed as they are kept (1,000 full frames would need 6 GB of RAM).
    model, preprocess = load_clip(model_name)
    tensors, prep_s, frame_size = [], [0.0], []

    def keep(i, arr):
        if len(tensors) < args.clip_frames:
            t0 = time.perf_counter()
            tensors.append(preprocess(Image.fromarray(arr)))
            prep_s[0] += time.perf_counter() - t0
            frame_size[:] = [f"{arr.shape[1]}x{arr.shape[0]}"]

    for v in pick_clips(videos, "rgb", args.clips, args.seed):
        decode(root / v.path, v.n_frames, thread_type, keep_step(v.fps, app_fps), None, on_keep=keep)
        if len(tensors) >= args.clip_frames:
            break
    if len(tensors) < args.clip_frames:
        raise ValueError(f"only {len(tensors)} frames for the CLIP benchmark")
    prep = {"images": len(tensors), "seconds": prep_s[0], "per_s": len(tensors) / prep_s[0], "input": frame_size[0],
            "transform": str(preprocess).replace("\n", " ")}
    clip_rows = clip_benchmark(model, torch.stack(tensors), args.batch_sizes)
    best = max(clip_rows, key=lambda r: r["per_s"])
    for r in clip_rows:
        print(f"CLIP batch {r['batch']:>3}: {r['per_s']:.0f} img/s, peak {r['peak_alloc_mib']:.0f} MiB allocated", flush=True)
    print(f"preprocess (CPU, 1 process): {prep['per_s']:.0f} img/s", flush=True)
    load_end = os.getloadavg()

    # Log this run, then read all runs of this benchmark for the spread.
    total = {m: sum(v.n_frames / v.fps for v in videos if v.modality == m) for m in ("rgb", "ir")}
    n_1fps = sum(-(-v.n_frames // keep_step(v.fps, app_fps)) for v in videos)
    max_crops = cfg["appearance"]["max_crops_per_frame"]
    proj = {key: sum(total[m] / results[(m, key)]["x"] for m in total) for key, _, _, _ in modes}
    clip_frames_s, clip_crops_s = n_1fps / best["per_s"], n_1fps * (1 + max_crops) / best["per_s"]
    prep_frames_s = n_1fps / prep["per_s"]
    metrics = {"decode_thread_auto": float(thread_type == "AUTO"), "load1_start": load_start[0], "load1_end": load_end[0],
               "clip_best_batch": float(best["batch"]), "clip_img_per_s": best["per_s"],
               "clip_peak_mib": best["peak_alloc_mib"], "preprocess_img_per_s": prep["per_s"],
               "proj_clip_frames_h": clip_frames_s / 3600, "proj_clip_with_crops_h": clip_crops_s / 3600,
               "proj_preprocess_frames_h": prep_frames_s / 3600}
    for (m, key), r in results.items():
        metrics[f"decode_{m}_{key}_x_realtime"] = r["x"]
    for key, s in proj.items():
        metrics[f"proj_decode_{key}_h"] = s / 3600
    row = log_experiment("none", metrics, notes=f"{NOTES_PREFIX}, seed {args.seed}, {args.clips} clips/modality, "
                         f"{len(tensors)} CLIP frames, untrained {model_name} fp16", stamp=stamp)
    hist_keys = [f"decode_{m}_{key}_x_realtime" for m in ("rgb", "ir") for key, _, _, _ in modes] + ["clip_img_per_s"]
    runs = history(DEFAULT_CSV, hist_keys)
    slowest = {k: min(r[k] for r in runs if r[k] is not None) for k in hist_keys}
    proj_slow = {key: sum(total[m] / slowest[f"decode_{m}_{key}_x_realtime"] for m in total) for key, _, _, _ in modes}

    plateau = [r["batch"] for r in clip_rows if r["per_s"] >= 0.98 * best["per_s"]]
    biggest = max(clip_rows, key=lambda r: r["batch"])
    lines = [
        "# Throughput (step 1.4)", "",
        f"Generated by `uv run python -m scripts.throughput` (seed {args.seed}). CPU {cpu_model()}, "
        f"{torch.cuda.get_device_name(0)} ({torch.cuda.mem_get_info()[1] / 2**30:.1f} GiB). torch {torch.__version__}, "
        f"open_clip {open_clip.__version__}, PyAV {av.__version__}. Load average (1/5/15 min) at start "
        f"{'/'.join(f'{x:.2f}' for x in load_start)}, at end {'/'.join(f'{x:.2f}' for x in load_end)} "
        "(the benchmark's own threads raise it). Projection basis: the whole selection, "
        f"{total['rgb'] / 3600:.2f} h RGB (1920x1080) + {total['ir'] / 3600:.2f} h IR (352x240) = "
        f"{(total['rgb'] + total['ir']) / 3600:.2f} h.", "",
        "## Decoding (CPU, sequential from frame 0)", "",
        f"Threading compared on {probe.clip}, decode only: slice {threading['SLICE']:.1f} s, frame (AUTO) "
        f"{threading['AUTO']:.1f} s; the first {args.hash_frames} frames are identical (hash) → **{thread_type}** used "
        "below" + (" (frame threading: decoding uses several cores, so other jobs running alongside slow it)"
                   if thread_type == "AUTO" else "") + ". Every clip decoded to exactly its manifest frame count.", "",
        "| modality | mode | video | wall time | × realtime | decoded frames/s | kept frames |",
        "|---|---|---|---|---|---|---|"]
    for (m, key), r in results.items():
        lines.append(f"| {m} | {r['label']} | {r['video_s'] / 3600:.2f} h | {r['seconds']:.0f} s | "
                     f"{r['x']:.0f}× | {r['frames'] / r['seconds']:.0f} | {r['kept']} |")
    lines += ["", "Clips: RGB " + ", ".join(results[("rgb", "decode_only")]["clips"]) + "; IR "
              + ", ".join(results[("ir", "decode_only")]["clips"]) + ".", "",
              "## CLIP image encoder (GPU)", "",
              f"`{model_name}`, open_clip `precision=\"fp16\"` (mixed precision), untrained weights (same compute as any "
              f"tag), {len(tensors)} real 1 fps frames, host→GPU copy included, after 2 warm-up batches.", "",
              "| batch | img/s | wall time | peak allocated (torch) | device memory in use |", "|---|---|---|---|---|"]
    for r in clip_rows:
        lines.append(f"| {r['batch']} | {r['per_s']:.0f} | {r['seconds']:.2f} s | {r['peak_alloc_mib']:.0f} MiB | "
                     f"{r['device_used_mib']:.0f} / {r['total_mib']:.0f} MiB |")
    lines += ["", f"Fastest in this run: batch {best['batch']}, {best['per_s']:.0f} img/s; within 2 % of it: batch "
              f"{', '.join(map(str, plateau))}. **VRAM is not the limit**: the largest batch tried ({biggest['batch']}) "
              f"used {biggest['device_used_mib']:.0f} of {biggest['total_mib']:.0f} MiB. Within the plateau, pick the "
              f"batch by memory headroom. Embedding dim {best['dim']}.", "",
              f"Preprocessing ({prep['transform']}) on the CPU in one process: **{prep['per_s']:.0f} img/s** "
              f"for {prep['input']} frames.", "",
              "## Run-to-run spread (all runs of this benchmark in results/experiments.csv)", "",
              "| run | git | " + " | ".join(k.removeprefix("decode_").removesuffix("_x_realtime") for k in hist_keys) + " |",
              "|---|---|" + "---|" * len(hist_keys)]
    for r in runs:
        lines.append(f"| {r['timestamp'][:19]} | {r['git']} | "
                     + " | ".join("–" if r[k] is None else f"{r[k]:.0f}" for k in hist_keys) + " |")
    lines += ["", "Decode columns: × realtime; clip_img_per_s: images/s. '–': not recorded in a comparable way "
              "(the IR 5 fps mode before git 3f50859 scaled frames up).", "",
              "## Projected times for the whole selection", "",
              "Each row is one stage timed on its own; no end-to-end run (decode + preprocessing + encoder together) "
              "was measured. 'Slowest run' uses the slowest × realtime recorded for each mode.", "",
              "| job | this run | slowest run |", "|---|---|---|"]
    for key, label, _, _ in modes:
        lines.append(f"| decode, {label} | {hms(proj[key])} | {hms(proj_slow[key])} |")
    lines += [f"| CLIP encoder, full frames only ({n_1fps:,} frames at {app_fps} fps, before dedup) | {hms(clip_frames_s)} | "
              f"{hms(n_1fps / slowest['clip_img_per_s'])} |",
              f"| CLIP encoder, full frame + {max_crops} crops (upper bound, {n_1fps * (1 + max_crops):,} images) | "
              f"{hms(clip_crops_s)} | {hms(n_1fps * (1 + max_crops) / slowest['clip_img_per_s'])} |",
              f"| CLIP preprocessing, full frames, 1 CPU process (at the 1080p rate; IR frames are cheaper) | "
              f"{hms(prep_frames_s)} | – |",
              f"| CLIP preprocessing, {max_crops} crops per frame | not measured | – |", "",
              "## Caveats", "",
              f"- Clips may already be in the page cache (the same seeded clips as earlier runs); not controlled. "
              f"RGB video is {gb_per_h['rgb']:.1f} GB per hour, IR {gb_per_h['ir']:.2f} GB per hour.",
              "- The CPU is a hybrid design (performance and efficiency cores) in a laptop: decode speed depends on which "
              "cores the OS uses and on temperature. See the run-to-run table.",
              "- CLIP weights untrained: the weight tag (appearance.pretrained, TBD) does not change the compute.",
              "- Preprocessing is the model's default: shortest side to 224, then a **centre crop to 224x224**, which "
              "cuts the left and right ~22 % of a 16:9 frame. Whether full frames should be padded or tiled instead is "
              "a step 4.2 decision. It ran in one process; splitting it over CPU workers was not measured.",
              "- Not measured here: the detector (Gate 3.3), MOG2 itself, crop extraction and crop preprocessing, the "
              "action branch.", "",
              f"experiments.csv row: {row['timestamp']}, git {row['git_hash']}, config {row['config_hash'][:12]}.", ""]
    text = "\n".join(lines)
    args.report.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
