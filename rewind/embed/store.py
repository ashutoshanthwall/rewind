"""Reading step 4.2 appearance embeddings back, and choosing the vectors of one search variant (step 4.4)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rewind.embed.crops import VARIANTS
from rewind.ingest.manifest import Video

FRAME_MODES = {"tiles2": ("tile0", "tile1"), "pad": ("pad",)}


def model_dir(root: Path, model: str, tag: str) -> Path:
    return root / "embeddings" / "appearance" / f"{model}__{tag}"


def clip_paths(root: Path, model: str, tag: str, v: Video) -> dict[str, Path]:
    d = model_dir(root, model, tag) / v.camera_id
    return {"npy": d / f"{v.clip}.npy", "meta": d / f"{v.clip}.parquet", "done": d / f"{v.clip}.done.json"}


def load_appearance(root: Path, model: str, tag: str, videos: list[Video]) -> tuple[np.ndarray, pd.DataFrame]:
    """Vectors (fp16) and metadata of these videos, concatenated in the given order. Every clip must be done, all
    with one settings hash."""
    vecs, metas, settings = [], [], set()
    for v in videos:
        p = clip_paths(root, model, tag, v)
        if not p["done"].exists():
            raise FileNotFoundError(f"{v.clip}: no embeddings for {model}/{tag} ({p['done']})")
        settings.add(json.loads(p["done"].read_text())["settings"])
        x, m = np.load(p["npy"]), pd.read_parquet(p["meta"])
        if len(x) != len(m) or (m.video_id != v.video_id).any():
            raise ValueError(f"{v.clip}: vectors and metadata disagree")
        vecs.append(x)
        metas.append(m)
    if len(settings) != 1:
        raise ValueError(f"{model}/{tag}: clips embedded with {len(settings)} different settings: {sorted(settings)}")
    return np.concatenate(vecs), pd.concat(metas, ignore_index=True)


def variant_mask(meta: pd.DataFrame, frame_mode: str, crop_source: str, max_crops: int) -> np.ndarray:
    """Rows a search variant uses: the frame views of frame_mode, plus the crops of crop_source
    (rewind/embed/crops.variant, applied to the stored rows)."""
    if frame_mode not in FRAME_MODES:
        raise ValueError(f"unknown frame_mode {frame_mode!r}; expected one of {sorted(FRAME_MODES)}")
    if crop_source not in VARIANTS:
        raise ValueError(f"unknown crop_source {crop_source!r}; expected one of {VARIANTS}")
    keep = (meta["type"] == "frame") & meta["view"].isin(FRAME_MODES[frame_mode])
    crop = meta["type"] == "crop"
    det = crop & (meta["crop_source"] == "detection") & (meta["crop_rank"] < max_crops)
    mot = crop & (meta["crop_source"] == "motion")
    if crop_source == "motion":
        keep |= mot & (meta["crop_rank"] < max_crops)
    elif crop_source == "detection_then_motion":
        keep |= det
        frame_key = [meta["video_id"], meta["frame"]]
        n_det = det.groupby(frame_key).transform("sum")
        fill = mot & ~meta["dup"].astype(bool)
        order = meta[fill].sort_values(["video_id", "frame", "crop_rank"]).groupby(["video_id", "frame"]).cumcount()
        slot = pd.Series(np.inf, index=meta.index)
        slot[order.index] = order
        keep |= fill & (slot < (max_crops - n_det))
    return keep.to_numpy()
