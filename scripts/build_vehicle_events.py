"""Step 6.1: apply the frozen vehicle_stops / vehicle_starts rules (config `events`) to the tracks of every clip and
store the detected events in SQLite table `detected_events` (data/rewind.sqlite).

Indexing, like detection and tracking: runs over every clip, reads no labels. The table is derived data: it is
rebuilt from scratch on every run (all rows replaced in one transaction). It holds NO track index: an event is a
camera, a time, a type, the vehicle's majority class and the best box of the evidence (rewind/track/tracks.py). The
crop of that box (crop_path) is filled in by step 6.2 (scripts/embed_events.py), which decodes the clips once.

Run from the repo root:
  uv run python -m scripts.build_vehicle_events
"""

import argparse
import hashlib
import json
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yaml

from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.motion import store as motion_store
from rewind.paths import REPO_ROOT
from rewind.track import store
from rewind.track.tracks import EVENT_TYPES, RuleParams, clip_events
from scripts.track_pass import settings_hash

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS detected_events (
    det_event_id  INTEGER PRIMARY KEY,
    video_id      INTEGER NOT NULL REFERENCES videos(video_id),
    camera_id     TEXT    NOT NULL,
    event_type    TEXT    NOT NULL CHECK (event_type IN ({', '.join(repr(t) for t in EVENT_TYPES)})),
    start_s       REAL    NOT NULL,          -- clip seconds (= proxy time), [start_s, end_s): the run transition
    end_s         REAL    NOT NULL,
    t_s           REAL    NOT NULL,          -- event time = end_s (first sample of the new run)
    start_time    TEXT    NOT NULL,          -- local wall-clock ISO
    end_time      TEXT    NOT NULL,
    class         TEXT    NOT NULL,          -- majority detector class of the evidence boxes
    box_frame     INTEGER NOT NULL,          -- frame of the best box (highest score near the event)
    box_t_s       REAL    NOT NULL,
    box_score     REAL    NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,   -- best box, original pixels
    crop_path     TEXT,                      -- set by step 6.2 (scripts/embed_events.py)
    settings      TEXT    NOT NULL           -- rules_settings(): tracker + rule parameters
) STRICT;
"""


def rules_settings(cfg: dict, track_settings: str) -> str:
    p = RuleParams.from_config(cfg)
    blob = json.dumps({"tracks": track_settings, "rules": p.__dict__}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    ap.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    args = ap.parse_args()
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    p = RuleParams.from_config(cfg)          # raises on TBD values
    art = REPO_ROOT / "artifacts"
    conn = connect(args.db)
    videos = load_videos(conn)
    det_settings = {motion_store.load_marker(art, v)["settings"] for v in videos}
    if len(det_settings) != 1:
        raise ValueError(f"detections with {len(det_settings)} settings")
    tset = settings_hash(cfg, det_settings.pop())
    settings = rules_settings(cfg, tset)

    t0 = time.perf_counter()
    rows = []
    for v in videos:
        ev = clip_events(store.load_tracks(art, v, tset), p)
        for e in ev.itertuples(index=False):
            rows.append((v.video_id, v.camera_id, e.event_type, e.start_s, e.end_s, e.t_s,
                         (v.start_time + timedelta(seconds=e.start_s)).isoformat(),
                         (v.start_time + timedelta(seconds=e.end_s)).isoformat(),
                         e[4],                  # column 'class' (a Python keyword: itertuples names it _4)
                         e.box_frame, e.box_t_s,
                         e.box_score, e.x1, e.y1, e.x2, e.y2, settings))
    with conn:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM detected_events")
        conn.executemany("INSERT INTO detected_events (video_id, camera_id, event_type, start_s, end_s, t_s, "
                         "start_time, end_time, class, box_frame, box_t_s, box_score, x1, y1, x2, y2, settings) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    df = pd.read_sql_query("SELECT camera_id, event_type, COUNT(*) AS n FROM detected_events GROUP BY 1, 2", conn)
    hours = {}
    for v in videos:
        hours[v.camera_id] = hours.get(v.camera_id, 0) + v.n_frames / v.fps / 3600
    df["per_hour"] = [n / hours[c] for c, n in zip(df["camera_id"], df["n"])]
    print(df.to_string(index=False))
    print(f"{len(rows)} detected events over {len(videos)} clips in {time.perf_counter() - t0:.0f} s; "
          f"settings {settings}")
    log_experiment("none", {"events": len(rows), **{f"n_{t}": int(df[df["event_type"] == t]["n"].sum())
                                                    for t in EVENT_TYPES}},
                   f"step 6.1 build_vehicle_events ({p}), settings {settings}", stamp=stamp)


if __name__ == "__main__":
    main()
