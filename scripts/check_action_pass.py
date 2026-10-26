"""Check the outputs of scripts/embed_action.py over all footage (step 5.2; read-only).

For every clip of the manifest: .npy, .parquet and .done.json exist, no leftover *.tmp; the marker's settings hash
equals the current config's; rows = marker windows = len(window_starts(clip duration)); window_start equals
window_starts() and t_s = window_start + window_s / 2; video_id / camera_id / modality match the manifest; vectors are
float16, (n, dim) with one dim for all clips, finite, unit norm (|norm - 1| <= --norm-tol, fp16 rounding). Prints
totals (clips, windows, hours, size, wall time, peak VRAM). Exits non-zero if any check fails. No labels are read.

Run from the repo root:
  uv run python -m scripts.check_action_pass
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.embed.windows import window_starts
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT, display_path
from scripts.embed_action import out_dir, settings_hash


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    ap.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    ap.add_argument("--norm-tol", type=float, default=2e-3, help="max |L2 norm - 1| of an fp16 vector")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    act = cfg["action"]
    root = out_dir(REPO_ROOT / "artifacts", cfg)
    settings = settings_hash(cfg)
    videos = load_videos(connect(args.db))
    errors, dims = [], set()
    tot = {"clips": 0, "windows": 0, "video_s": 0.0, "seconds": 0.0, "bytes": 0, "peak_vram_mib": 0}
    leftovers = sorted(root.rglob("*.tmp"))
    if leftovers:
        errors.append(f"{len(leftovers)} leftover .tmp files, e.g. {leftovers[0]}")
    for v in videos:
        d = root / v.camera_id
        paths = {k: d / f"{v.clip}{ext}" for k, ext in (("npy", ".npy"), ("meta", ".parquet"), ("done", ".done.json"))}
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            errors.append(f"{v.clip}: missing {missing}")
            continue
        m = json.loads(paths["done"].read_text())
        x = np.load(paths["npy"])
        meta = pd.read_parquet(paths["meta"])
        starts = window_starts(v.n_frames / v.fps, act["window_s"], act["stride_s"])
        bad = []
        if m["settings"] != settings:
            bad.append(f"settings {m['settings']} != {settings}")
        if not (len(x) == len(meta) == m["windows"] == len(starts)):
            bad.append(f"rows: npy {len(x)}, parquet {len(meta)}, marker {m['windows']}, windows {len(starts)}")
        else:
            if not np.allclose(meta["window_start"].to_numpy(float), starts, atol=1e-9):
                bad.append("window_start != window_starts()")
            if not np.allclose(meta["t_s"].to_numpy(float), starts + act["window_s"] / 2, atol=1e-9):
                bad.append("t_s != window centre")
        if x.dtype != np.float16 or x.ndim != 2:
            bad.append(f"dtype/shape {x.dtype} {x.shape}")
        else:
            dims.add(x.shape[1])
            xf = x.astype(np.float32)
            if not np.isfinite(xf).all():
                bad.append("non-finite values")
            elif np.abs(np.linalg.norm(xf, axis=1) - 1).max() > args.norm_tol:
                bad.append(f"norm off by {np.abs(np.linalg.norm(xf, axis=1) - 1).max():.4f}")
        if set(meta["video_id"]) != {v.video_id} or set(meta["camera_id"]) != {v.camera_id} \
                or set(meta["modality"]) != {v.modality}:
            bad.append("video_id / camera_id / modality differ from the manifest")
        if bad:
            errors.append(f"{v.clip}: " + "; ".join(bad))
        tot["clips"] += 1
        tot["windows"] += len(x)
        tot["video_s"] += v.n_frames / v.fps
        tot["seconds"] += m["seconds"]
        tot["bytes"] += sum(p.stat().st_size for p in paths.values())
        tot["peak_vram_mib"] = max(tot["peak_vram_mib"], m["peak_vram_mib"])
    if len(dims) > 1:
        errors.append(f"more than one vector dim: {sorted(dims)}")
    print(f"{display_path(root)}: {tot['clips']}/{len(videos)} clips, {tot['windows']:,} windows, "
          f"{tot['video_s'] / 3600:.2f} h of video, dim {sorted(dims)}, {tot['bytes'] / 2 ** 20:.1f} MiB, "
          f"sum of per-clip wall time {tot['seconds'] / 60:.1f} min, peak VRAM {tot['peak_vram_mib']} MiB")
    if errors:
        print(f"FAIL: {len(errors)} problems", *errors[:20], sep="\n  ")
        sys.exit(1)
    print("PASS: every check")


if __name__ == "__main__":
    main()
