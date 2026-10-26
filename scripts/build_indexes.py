"""Step 7.1: build the per-camera, per-branch HNSW indexes of the C++ engine (rewind/search/engine.py layout).

Branches (rewind.search.engine.active_branches): appearance always, action if action.enabled. Vectors:
  appearance  the collected step 4.3 index (scripts/collect_appearance.py), rows of the config's variant
              (appearance.frame_mode / crop_source / max_crops_per_frame; mean-centered with the all-footage camera
              means if appearance.mean_center); 175,039 rows for pad/motion/raw
  action      the step 5.2 window vectors of every clip (scripts/embed_action.py), window time = centre
Each camera's rows -> float32, re-normalised -> vectors.f32 + manifest.json -> engine loads them back from the file
(SHA-256 + norm checks) -> HNSW build (search.hnsw, ef search.ef_search) -> graph.hnsw; meta.parquet maps row ->
metadata (+ wall_s for the time filter). Every file is written to a .tmp name first and renamed. All footage is
indexed (indexing reads footage of every split, never labels: CLAUDE.md 1.10).

Run from the repo root (the engine module must be built first, see rewind/search/engine.py BUILD_HINT):
  uv run python -m scripts.build_indexes [--branches appearance action] [--cameras G330 ...]
"""

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.embed.store import variant_mask
from rewind.experiment_log import run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT, display_path
from rewind.search.core import mean_center
from rewind.search.engine import (EngineConfig, active_branches, branch_model, engine_module, index_name,
                                  index_path, to_unit_float32, wall_seconds, write_store)
from scripts.collect_appearance import index_dir, load_means

APPEARANCE_COLUMNS = ["vector_id", "video_id", "camera_id", "modality", "frame", "t_s", "type", "view",
                      "crop_source", "crop_rank", "class", "score", "x1", "y1", "x2", "y2", "clip"]


def appearance_rows(cfg: dict, art: Path) -> tuple[np.ndarray, pd.DataFrame]:
    a = cfg["appearance"]
    d = index_dir(art, a["model"], a["pretrained"])
    meta = pd.read_parquet(d / "meta.parquet")
    keep = variant_mask(meta, a["frame_mode"], a["crop_source"], a["max_crops_per_frame"])
    vecs = np.load(d / "vectors.npy", mmap_mode="r")
    if len(vecs) != len(meta):
        raise ValueError(f"{d}: {len(vecs)} vectors for {len(meta)} metadata rows")
    m = meta[keep].reset_index(drop=True)[APPEARANCE_COLUMNS]
    x = np.asarray(vecs[np.flatnonzero(keep)])
    if a["mean_center"]:
        x = mean_center(x, m["camera_id"].to_numpy(),
                        load_means(art, a["model"], a["pretrained"], a["frame_mode"], a["crop_source"]))
    return x, m


def action_rows(cfg: dict, art: Path, videos: list) -> tuple[np.ndarray, pd.DataFrame]:
    from scripts.action_bakeoff import load_action          # imports torch; only needed for this branch
    from scripts.embed_action import model_key
    x, m, _ = load_action(art, model_key(cfg), videos, cfg)
    return x, m.reset_index(drop=True)


def with_wall_seconds(meta: pd.DataFrame, videos: list) -> pd.DataFrame:
    """wall_s = wall-clock seconds of the row's time (video start_time + t_s); checked to lie inside its video."""
    by_id = {v.video_id: v for v in videos}
    start = meta["video_id"].map(lambda i: wall_seconds(by_id[i].start_time))
    dur = meta["video_id"].map(lambda i: by_id[i].n_frames / by_id[i].fps)
    if meta["video_id"].map(lambda i: i not in by_id).any():
        raise ValueError("metadata rows of videos missing from the manifest")
    if ((meta["t_s"] < 0) | (meta["t_s"] > dur)).any():
        raise ValueError("t_s outside its video")
    out = meta.copy()
    out["wall_s"] = start + meta["t_s"]
    return out


def build_one(mod, ecfg: EngineConfig, branch: str, camera: str, model: str, x: np.ndarray, meta: pd.DataFrame,
              stamp: dict) -> dict:
    d = index_path(ecfg.index_dir, branch, camera)
    tmp = d.with_name(d.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    x32 = to_unit_float32(x)
    sha = write_store(tmp, x32, model)
    meta = meta.reset_index(drop=True)
    meta.insert(0, "row", np.arange(len(meta), dtype=np.int64))
    meta.to_parquet(tmp / "meta.parquet", index=False)
    idx = mod.NamedIndex(index_name(branch, camera), model)
    idx.load_vectors(str(tmp / "vectors.f32"), str(tmp / "manifest.json"), x32.shape[1], ecfg.norm_tolerance)
    idx.set_times(meta["wall_s"].to_numpy(np.float64))
    t0 = time.perf_counter()
    build_s = idx.build(ecfg.M, ecfg.ef_construction, ecfg.ef_search, ecfg.seed)
    idx.save(str(tmp / "graph.hnsw"))
    info = {"index": index_name(branch, camera), "model": model, "rows": int(len(x32)), "dim": int(x32.shape[1]),
            "sha256": sha, "build_s": round(build_s, 2), "build_and_save_s": round(time.perf_counter() - t0, 2),
            "max_norm_dev_fp16": float(np.abs(np.linalg.norm(np.asarray(x, np.float32), axis=1) - 1).max()),
            "hnsw": {"M": ecfg.M, "ef_construction": ecfg.ef_construction, "ef_search": ecfg.ef_search,
                     "seed": ecfg.seed},
            "git_hash": stamp["git_hash"], "config_hash": stamp["config_hash"]}
    (tmp / "build.json").write_text(json.dumps(info, indent=1))
    if d.exists():
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    tmp.rename(d)
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--branches", nargs="+")
    p.add_argument("--cameras", nargs="+")
    p.add_argument("--index-dir", type=Path, help="override search.index_dir (smoke tests)")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    stamp = run_stamp(args.config)
    ecfg = EngineConfig.from_config(cfg)
    if args.index_dir:
        ecfg = EngineConfig(**{**ecfg.__dict__, "index_dir": args.index_dir})
    branches = args.branches or active_branches(cfg)
    if set(branches) - set(active_branches(cfg)):
        raise ValueError(f"branches {branches} include one that is not active in the config")
    mod = engine_module()
    videos = load_videos(connect(args.db))
    for branch in branches:
        t0 = time.perf_counter()
        x, meta = appearance_rows(cfg, args.art) if branch == "appearance" else action_rows(cfg, args.art, videos)
        meta = with_wall_seconds(meta, videos)
        model = branch_model(cfg, branch)
        print(f"{branch}: {len(x):,} rows x {x.shape[1]} ({model}); loaded in {time.perf_counter() - t0:.0f} s",
              flush=True)
        for cam in sorted(meta["camera_id"].unique()):
            if args.cameras and cam not in args.cameras:
                continue
            rows = np.flatnonzero(meta["camera_id"].to_numpy() == cam)
            info = build_one(mod, ecfg, branch, cam, model, x[rows], meta.iloc[rows], stamp)
            print(f"  {info['index']}: {info['rows']:,} rows, build {info['build_s']:.1f} s, fp16 max |norm-1| "
                  f"{info['max_norm_dev_fp16']:.1e} -> {display_path(index_path(ecfg.index_dir, branch, cam))}",
                  flush=True)
    print("done")


if __name__ == "__main__":
    main()
