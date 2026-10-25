"""Where scripts/index_pass.py keeps its per-clip outputs, and how later steps read them."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rewind.ingest.manifest import Video


def seconds_path(art: Path, v: Video) -> Path:
    return art / "motion" / v.camera_id / f"{v.clip}.seconds.parquet"


def boxes_path(art: Path, v: Video) -> Path:
    return art / "motion" / v.camera_id / f"{v.clip}.boxes.parquet"


def detections_path(art: Path, v: Video) -> Path:
    return art / "detections" / v.camera_id / f"{v.clip}.parquet"


def marker_path(art: Path, v: Video) -> Path:
    return art / "motion" / v.camera_id / f"{v.clip}.done.json"


def load_marker(art: Path, v: Video) -> dict:
    path = marker_path(art, v)
    if not path.exists():
        raise FileNotFoundError(f"{v.clip}: no done marker ({path}); has scripts.index_pass processed it?")
    return json.loads(path.read_text())


def load_activity(art: Path, v: Video) -> np.ndarray:
    """Per-second MOG2 activity of a finished clip (second i covers clip seconds [i, i + 1))."""
    load_marker(art, v)
    df = pd.read_parquet(seconds_path(art, v))
    if df["second"].tolist() != list(range(len(df))):
        raise ValueError(f"{v.clip}: per-second rows are not 0..{len(df) - 1}")
    return df["activity"].to_numpy()
