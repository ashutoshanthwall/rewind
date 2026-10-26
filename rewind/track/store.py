"""Where scripts/track_pass.py keeps its per-clip tracks, and how later steps read them."""

import json
from pathlib import Path

import pandas as pd

from rewind.ingest.manifest import Video
from rewind.track.tracks import TRACK_COLUMNS


def tracks_path(art: Path, v: Video) -> Path:
    return art / "tracks" / v.camera_id / f"{v.clip}.parquet"


def marker_path(art: Path, v: Video) -> Path:
    return art / "tracks" / v.camera_id / f"{v.clip}.done.json"


def load_tracks(art: Path, v: Video, settings: str) -> pd.DataFrame:
    """One clip's tracks; refuses a clip without a done marker or one written with other tracker settings."""
    marker = marker_path(art, v)
    if not marker.exists():
        raise FileNotFoundError(f"{v.clip}: no track marker ({marker}); has scripts.track_pass processed it?")
    m = json.loads(marker.read_text())
    if m["settings"] != settings:
        raise ValueError(f"{v.clip}: tracks have settings {m['settings']}, current config {settings}")
    df = pd.read_parquet(tracks_path(art, v))
    if list(df.columns) != TRACK_COLUMNS or len(df) != m["n_rows"]:
        raise ValueError(f"{v.clip}: tracks file does not match its marker")
    return df
