"""Step 4.3: collect the per-clip appearance embeddings of ALL footage into one index, with per-camera means.

Reads the step 4.2 outputs of every clip in the manifest (all splits: indexing uses footage, never labels) for one
(model, tag) and writes, under artifacts/embeddings/appearance/<model>__<tag>/index/:
  vectors.npy            (N, d) float16, raw (L2-normalised, not centered); row i has vector_id i
  meta.parquet           vector_id + the per-clip metadata + clip name; rows in manifest order (camera, start time)
  camera_means.parquet   per (frame_mode, crop_source) variant and camera: n vectors and the mean vector, over all
                         of that camera's vectors in the variant (CLAUDE.md step 4.3; no labels used)
  camera_type_means.parquet  the same per camera AND type (frame / crop): appearance.center_groups camera_type
                         (step 8.1 comparison, PROGRESS.md); column group = "<camera>/<type>"
  centered_<frame_mode>_<crop_source>.npy   the config's main variant, mean-centered and re-normalised (float16),
                         with centered_<...>_ids.npy (vector_ids of its rows)
appearance.mean_center decides which one search uses; both are kept for the 4.4 ablation (centering of other variants
is applied at load time from camera_means.parquet).

Run from the repo root (~1-2 min): uv run python -m scripts.collect_appearance
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.embed.crops import VARIANTS
from rewind.embed.store import FRAME_MODES, load_appearance, model_dir, variant_mask
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT, display_path
from rewind.search.core import camera_means, center_labels, mean_center

MEANS_FILE = {"camera": "camera_means.parquet", "camera_type": "camera_type_means.parquet"}


def index_dir(root: Path, model: str, tag: str) -> Path:
    return model_dir(root, model, tag) / "index"


def load_means(root: Path, model: str, tag: str, frame_mode: str, crop_source: str,
               center_groups: str = "camera") -> dict:
    """Group label (rewind.search.core.center_labels) -> mean vector of one variant, over all footage."""
    df = pd.read_parquet(index_dir(root, model, tag) / MEANS_FILE[center_groups])
    df = df[(df.frame_mode == frame_mode) & (df.crop_source == crop_source)]
    if df.empty:
        raise ValueError(f"no {center_groups} means for {frame_mode}/{crop_source}: run scripts.collect_appearance")
    key = "camera_id" if center_groups == "camera" else "group"
    return {getattr(r, key): np.asarray(r.mean, dtype=np.float32) for r in df.itertuples()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--clips", nargs="+", help="smoke test: only these clips (use a scratch --art)")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    a = cfg["appearance"]
    model, tag = a["model"], a["pretrained"]
    if "TBD" in (model, tag):
        raise ValueError("appearance.model / pretrained is TBD")
    videos = load_videos(connect(args.db))
    if args.clips:
        videos = [v for v in videos if v.clip in args.clips]
    vecs, meta = load_appearance(args.art, model, tag, videos)
    clip_of = {v.video_id: v.clip for v in videos}
    meta.insert(0, "vector_id", np.arange(len(meta), dtype=np.int64))
    meta["clip"] = meta["video_id"].map(clip_of)
    out = index_dir(args.art, model, tag)
    out.mkdir(parents=True, exist_ok=True)

    rows, type_rows = [], []
    cams = meta["camera_id"].to_numpy()
    groups = center_labels(cams, meta["type"].to_numpy(), "camera_type")
    for fm in FRAME_MODES:
        for cs in VARIANTS:
            keep = variant_mask(meta, fm, cs, a["max_crops_per_frame"])
            means = camera_means(vecs[keep], cams[keep])
            rows += [{"frame_mode": fm, "crop_source": cs, "camera_id": c, "n": int((cams[keep] == c).sum()),
                      "mean": m.tolist()} for c, m in means.items()]
            means = camera_means(vecs[keep], groups[keep])
            type_rows += [{"frame_mode": fm, "crop_source": cs, "group": g, "camera_id": g.split("/")[0],
                           "type": g.split("/")[1], "n": int((groups[keep] == g).sum()), "mean": m.tolist()}
                          for g, m in means.items()]
    means_df = pd.DataFrame(rows)

    fm, cs = a["frame_mode"], a["crop_source"]
    keep = variant_mask(meta, fm, cs, a["max_crops_per_frame"])
    main_means = {r["camera_id"]: np.asarray(r["mean"], np.float32) for r in rows
                  if r["frame_mode"] == fm and r["crop_source"] == cs}
    centered = mean_center(vecs[keep], cams[keep], main_means).astype(np.float16)

    for name, write in (("vectors.npy", lambda f: np.save(f, vecs)),
                        ("meta.parquet", lambda f: meta.to_parquet(f, index=False)),
                        ("camera_means.parquet", lambda f: means_df.to_parquet(f, index=False)),
                        ("camera_type_means.parquet", lambda f: pd.DataFrame(type_rows).to_parquet(f, index=False)),
                        (f"centered_{fm}_{cs}.npy", lambda f: np.save(f, centered)),
                        (f"centered_{fm}_{cs}_ids.npy", lambda f: np.save(f, meta["vector_id"].to_numpy()[keep]))):
        tmp = out / (name + ".tmp")
        with open(tmp, "wb") as f:
            write(f)
        tmp.rename(out / name)
    size = sum(f.stat().st_size for f in out.iterdir()) / 2 ** 30
    print(f"{len(meta):,} vectors from {meta.video_id.nunique()} clips; main variant {fm}/{cs}: {int(keep.sum()):,} "
          f"rows; {size:.2f} GiB in {display_path(out)}")
    print(meta.groupby(["camera_id", "type"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
