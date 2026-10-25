"""Step 1.2: make browser playback proxies (H.264 MP4, <= 720p, +faststart) for every video in the manifest.

Frame-exact: proxy frame i is original frame i (the i-th decoded frame, see rewind/ingest/manifest.py) and is
timestamped i / fps. Each proxy is written to "<clip>.mp4.part" and renamed only after these checks:
  - same frame count as the original (header and packet count), same fps, duration within proxy.max_duration_diff_s;
  - frame i has presentation time exactly i / fps (sorted packet pts), so playback time maps back to a frame index;
  - content alignment: over the first proxy.align_check_frames decoded frames (small grayscale images), the
    frame-to-frame difference images of proxy and original must correlate best at shift 0 among shifts
    -align_max_offset..+align_max_offset. Differences are used, not the frames themselves: in near-static CCTV
    the per-frame error is dominated by compression noise and barely depends on the shift (G340 test clip:
    mean abs diff 0.982 at shift 0 vs 0.978 at +1), while motion between frames pins the shift
    (correlation 0.55 at 0 vs 0.30 next best).
Then a row goes into the `proxies` table of data/rewind.sqlite. Resumable: videos with a proxy row are skipped
(file presence and size re-checked); a leftover .part is overwritten. A finished .mp4 without a row (a run stopped
between rename and insert) is an error: its encoding settings are unknown, so it is not adopted.
A clip that fails (ffmpeg error, frame count, alignment) is logged, gets no row, and the run continues; all failures
are listed at the end and the exit code is 1. Re-running retries exactly those clips. Disk, config and database
problems stop the run at once. Progress goes to the terminal and to data/proxy/make_proxies.log.
Originals are only read.

Run from the repo root:
  uv run python -m scripts.make_proxies                      all videos (long job: run in your own terminal)
  uv run python -m scripts.make_proxies --clips <clip> ...   only these clips (test run)
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import yaml

from rewind.ingest.manifest import connect

REPO_ROOT = Path(__file__).resolve().parent.parent


def log(msg: str, log_path: Path) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with log_path.open("a") as f:
        f.write(line + "\n")


def ffmpeg_command(src: Path, dst: Path, fps: float, cfg: dict) -> list[str]:
    # setpts numbers frames in decoder output (display) order: frame i gets pts i / fps, whatever pts the AVI
    # invented. passthrough keeps ffmpeg from dropping or duplicating frames to fit a frame-rate grid.
    # Integer rates only (all 300 files are 30/1): fps:g would round a rate like 30000/1001.
    if not float(fps).is_integer():
        raise ValueError(f"{src.name}: fps {fps} is not an integer; setpts needs an exact rational rate")
    vf = f"setpts=N/({fps:g}*TB),scale=-2:'min({cfg['max_height']},ih)'"
    return ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-vf", vf,
            "-fps_mode", "passthrough", "-c:v", cfg["codec"], "-preset", cfg["preset"], "-crf", str(cfg["crf"]),
            "-g", str(cfg["keyint_frames"]), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
            "-f", "mp4", str(dst)]


def probe_proxy(path: Path, fps: float) -> dict:
    """Header + packet facts of a proxy; frame i must have pts exactly i / fps (fps of the original, from the manifest)."""
    with av.open(str(path)) as container:
        if len(container.streams) != 1 or len(container.streams.video) != 1:
            raise ValueError(f"{path.name}: expected exactly one video stream, found {len(container.streams)}")
        stream = container.streams.video[0]
        n_header = stream.frames
        pts = sorted(packet.pts * stream.time_base for packet in container.demux(stream) if packet.size > 0)
        n_packets = len(pts)
        if n_header != n_packets:
            raise ValueError(f"{path.name}: header says {n_header} frames, file has {n_packets} packets")
        # Later steps map frame index -> proxy playback time as i / fps, so that must hold exactly. Not checked against
        # stream.average_rate: the container computes it from the duration, so a timestamp gap changes it too.
        rate = Fraction(fps)
        if pts != [i / rate for i in range(n_packets)]:
            first_bad = next(i for i, t in enumerate(pts) if t != i / rate)
            raise ValueError(f"{path.name}: frame {first_bad} has pts {float(pts[first_bad]):.6f} s, "
                             f"expected {float(first_bad / rate):.6f} s")
        return {"n_frames": n_packets, "fps": float(stream.average_rate),
                "duration_s": float(stream.duration * stream.time_base),
                "width": stream.codec_context.width, "height": stream.codec_context.height}


def decode_small(path: Path, n_frames: int, size: tuple[int, int]) -> np.ndarray:
    """First n_frames decoded frames (in decoder output order) as grayscale float32 images of size (w, h)."""
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(cv2.resize(frame.to_ndarray(format="gray"), size, interpolation=cv2.INTER_AREA))
            if len(frames) == n_frames:
                break
    if len(frames) < n_frames:
        raise ValueError(f"{path.name}: decoded only {len(frames)} of {n_frames} frames")
    return np.stack(frames).astype(np.float32)


def alignment_corr(orig: np.ndarray, proxy: np.ndarray, max_offset: int) -> dict[int, float]:
    """For each shift d: mean correlation between difference images orig[i+1]-orig[i] and proxy[i+d+1]-proxy[i+d].

    The same i are used for every d. A difference image with no change at all contributes 0.
    """
    n = min(len(orig), len(proxy)) - 1  # number of difference images
    if n <= 2 * max_offset:
        raise ValueError(f"Need more than {2 * max_offset + 1} frames to test shifts of +/-{max_offset}, got {n + 1}")
    diff_o = np.diff(orig[:n + 1], axis=0).reshape(n, -1)
    diff_p = np.diff(proxy[:n + 1], axis=0).reshape(n, -1)
    diff_o -= diff_o.mean(axis=1, keepdims=True)
    diff_p -= diff_p.mean(axis=1, keepdims=True)
    norm_o, norm_p = np.linalg.norm(diff_o, axis=1), np.linalg.norm(diff_p, axis=1)
    i = np.arange(max_offset, n - max_offset)
    out = {}
    for d in range(-max_offset, max_offset + 1):
        denom = norm_o[i] * norm_p[i + d]
        num = (diff_o[i] * diff_p[i + d]).sum(axis=1)
        out[d] = float(np.where(denom > 0, num / np.where(denom > 0, denom, 1), 0.0).mean())
    return out


def check_alignment(corr: dict[int, float], name: str) -> None:
    others = max(v for d, v in corr.items() if d != 0)
    if not corr[0] > others:
        raise ValueError(f"{name}: proxy frames are not aligned with the original (difference-image correlation "
                         f"by shift: { {d: round(v, 3) for d, v in corr.items()} })")


def verify_proxy(video: dict, proxy: Path, original: Path, cfg: dict) -> tuple[dict, dict[int, float]]:
    info = probe_proxy(proxy, video["fps"])
    if info["n_frames"] != video["n_frames"] or info["fps"] != video["fps"]:
        raise ValueError(f"{proxy.name}: {info['n_frames']} frames @ {info['fps']} fps, original has "
                         f"{video['n_frames']} @ {video['fps']}")
    if abs(info["duration_s"] - video["duration_s"]) > cfg["max_duration_diff_s"]:
        raise ValueError(f"{proxy.name}: duration {info['duration_s']:.3f} s vs original {video['duration_s']:.3f} s")
    w = cfg["align_check_width"]
    size = (w, round(video["height"] * w / video["width"]))
    n = min(cfg["align_check_frames"], video["n_frames"])
    corr = alignment_corr(decode_small(original, n, size), decode_small(proxy, n, size), cfg["align_max_offset"])
    check_alignment(corr, proxy.name)
    return info, corr


def make_one(video: dict, root: Path, proxy_dir: Path, cfg: dict) -> tuple[Path, dict, dict[int, float]]:
    """Encode to .part, verify, rename to .mp4. Raises if an unrecorded .mp4 is already there."""
    original = root / video["path"]
    final = proxy_dir / f"{video['clip']}.mp4"
    if final.exists():
        raise FileExistsError(f"{final} exists but is not recorded in the proxies table; its settings are unknown. "
                              "Check it and delete it by hand, then re-run.")
    part = final.with_name(final.name + ".part")
    result = subprocess.run(ffmpeg_command(original, part, video["fps"], cfg), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {original.name} (exit {result.returncode}):\n{result.stderr}")
    info, corr = verify_proxy(video, part, original, cfg)
    part.rename(final)
    return final, info, corr


def record(conn, video: dict, final: Path, root: Path, info: dict, corr: dict[int, float], cfg: dict) -> None:
    conn.execute(
        "INSERT INTO proxies (video_id, path, n_frames, fps, duration_s, width, height, size_bytes, align_corr,"
        " settings, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (video["video_id"], str(final.relative_to(root)), info["n_frames"], info["fps"], info["duration_s"],
         info["width"], info["height"], final.stat().st_size, json.dumps({str(d): v for d, v in corr.items()}),
         json.dumps(cfg, sort_keys=True), datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.commit()


def check_recorded(conn, root: Path) -> int:
    """Every recorded proxy must still exist with its recorded size. Returns how many are recorded."""
    rows = conn.execute("SELECT path, size_bytes FROM proxies").fetchall()
    for r in rows:
        path = root / r["path"]
        if not path.exists() or path.stat().st_size != r["size_bytes"]:
            raise FileNotFoundError(f"Recorded proxy {r['path']} is missing or changed size")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", nargs="+", help="only these clips (names without .r13.avi)")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--proxy-dir", type=Path, default=REPO_ROOT / "data" / "proxy")
    parser.add_argument("--min-free-gb", type=float, default=10.0, help="stop if free disk space falls below this")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["proxy"]
    if "TBD" in cfg.values():
        raise ValueError(f"proxy config has TBD values: {cfg}")
    if not args.db.exists():
        raise FileNotFoundError(f"{args.db} not found; build the manifest first (step 1.1)")
    if not args.proxy_dir.resolve().is_relative_to(REPO_ROOT):
        raise ValueError(f"--proxy-dir must be inside the repo ({REPO_ROOT}): paths are stored relative to it")
    args.proxy_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.proxy_dir / "make_proxies.log"
    conn = connect(args.db)
    n_done = check_recorded(conn, REPO_ROOT)

    todo = [dict(r) for r in conn.execute(
        "SELECT v.* FROM videos v LEFT JOIN proxies p USING (video_id) WHERE p.video_id IS NULL ORDER BY v.clip")]
    if args.clips:
        unknown = set(args.clips) - {r["clip"] for r in conn.execute("SELECT clip FROM videos")}
        if unknown:
            raise ValueError(f"Clips not in the manifest: {sorted(unknown)}")
        todo = [v for v in todo if v["clip"] in args.clips]
    log(f"proxies recorded: {n_done}; to make: {len(todo)} ({sum(v['duration_s'] for v in todo) / 3600:.2f} h of "
        f"video) -> {args.proxy_dir}; settings {json.dumps(cfg, sort_keys=True)}", log_path)

    t0, video_s, failed = time.monotonic(), 0.0, []
    for i, video in enumerate(todo, start=1):
        free_gb = shutil.disk_usage(args.proxy_dir).free / 1e9
        if free_gb < args.min_free_gb:
            raise RuntimeError(f"Only {free_gb:.1f} GB free (< --min-free-gb {args.min_free_gb}); stopping")
        t_file = time.monotonic()
        try:
            final, info, corr = make_one(video, REPO_ROOT, args.proxy_dir, cfg)
        except (ValueError, RuntimeError, FileExistsError) as e:  # this clip only; listed at the end, exit 1
            failed.append(video["clip"])
            log(f"[{i}/{len(todo)}] FAILED {video['clip']}: {e}", log_path)
            continue
        record(conn, video, final, REPO_ROOT, info, corr, cfg)
        video_s += video["duration_s"]
        elapsed = time.monotonic() - t0
        remaining = sum(v["duration_s"] for v in todo[i:])
        log(f"[{i}/{len(todo)}] {final.name}  {info['n_frames']} frames {info['width']}x{info['height']}  "
            f"{final.stat().st_size / 1e6:.1f} MB  {time.monotonic() - t_file:.1f} s  "
            f"align corr[0]={corr[0]:.3f} next-best={max(v for d, v in corr.items() if d != 0):.3f}  "
            f"ETA {remaining / (video_s / elapsed) / 60:.0f} min", log_path)
    total = conn.execute("SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM proxies").fetchone()
    log(f"done: {total[0]} proxies recorded, {total[1] / 1e9:.2f} GB; failed this run: {len(failed)}", log_path)
    if failed:
        log(f"FAILED clips (no proxy recorded; re-run to retry): {failed}", log_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
