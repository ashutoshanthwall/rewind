"""Ingest manifest (step 1.1): one row per original video in data/rewind.sqlite, table `videos`.

Maps every (file, frame) to (camera, local wall-clock time) and records each original's SHA-256.

Time model (checked on all 300 selected files, 2026-10-25):
  - The filename gives the local wall-clock time of frame 0, to the second:
    date.HH-MM-SS(start).HH-MM-SS(end).site.camera.r13.avi
  - Frame f starts at start_time + f / fps. fps comes from the container (30/1 for every selected file).
  - Filename times are cut to whole seconds at both ends, so the filename duration is informational only
    (15 names say 299 s for 300.0-300.33 s of video). Duration comes from the container (n_frames / fps);
    container minus filename duration is -0.77 .. +1.33 s over the 300 files.
  VERIFY: the sub-second part of the start time is unknown, so wall-clock times are accurate to about +/- 1 s.
  VERIFY: times are naive local times. 2018-03-11 is the US daylight-saving switch date (no selected clip spans
  02:00-03:00), so never subtract times across dates; whether the camera clocks follow DST is unknown.

Frame index rule for every later step (checked on all 300 files): the files are H.264 with B-frames in AVI, which
stores no presentation timestamps. ffmpeg invents packet pts in decode order (pts = dts + 1), and the decoder
attaches them to frames it outputs in display order, so frame.pts / frame.time are scrambled (e.g. 1, 3, 2, 6) and
ffprobe shows pts=N/A. The frame index is therefore the count of decoded frames from 0, and its time is index / fps.
Never use frame.pts or frame.time, and don't seek by timestamp without checking the frame index you land on.
"""

import hashlib
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import av

VIDEO_SUFFIX = ".r13.avi"  # MEVA release r13 (dataset fact, drops-123-r13)
NAME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.([a-z]+)\.(G\d+)" + re.escape(VIDEO_SUFFIX) + "$")
# Snap seconds * fps to 1/10000 of a frame before flooring. datetime has microsecond resolution, so
# start + 1/30 s is stored as 33333 us, and 0.033333 * 30 = 0.99999 would otherwise floor to frame 0.
# The storage error is <= 0.5 us; the snap is 3.3 us at 30 fps. Consequence: an instant less than ~3 us before
# a frame boundary counts as the next frame (below what datetime can represent reliably).
FRAME_SNAP_DECIMALS = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id      INTEGER PRIMARY KEY,
    clip          TEXT    NOT NULL UNIQUE,   -- filename without .r13.avi
    path          TEXT    NOT NULL UNIQUE,   -- relative to the repo root
    size_bytes    INTEGER NOT NULL,
    sha256        TEXT    NOT NULL,
    camera_id     TEXT    NOT NULL,
    site          TEXT    NOT NULL,
    modality      TEXT    NOT NULL CHECK (modality IN ('rgb', 'ir')),
    start_time    TEXT    NOT NULL,          -- local wall-clock of frame 0, 'YYYY-MM-DDTHH:MM:SS' (filename)
    name_end_time TEXT    NOT NULL,          -- filename end time, truncated to the second (informational)
    duration_s    REAL    NOT NULL,          -- container: n_frames / fps
    n_frames      INTEGER NOT NULL,
    fps           REAL    NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    codec         TEXT    NOT NULL,
    ingested_at   TEXT    NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS proxies (
    video_id      INTEGER PRIMARY KEY REFERENCES videos(video_id),
    path          TEXT    NOT NULL UNIQUE,   -- relative to the repo root
    n_frames      INTEGER NOT NULL,          -- equal to videos.n_frames (checked); frame i is at i / fps
    fps           REAL    NOT NULL,
    duration_s    REAL    NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    size_bytes    INTEGER NOT NULL,
    align_corr    TEXT    NOT NULL,          -- JSON {shift: difference-image correlation} (alignment check)
    settings      TEXT    NOT NULL,          -- JSON of the proxy config used
    created_at    TEXT    NOT NULL
) STRICT;

-- Step 1.3 (rewind/ingest/annotations.py). Rebuilt together by scripts/build_events.py.
CREATE TABLE IF NOT EXISTS clip_labels (
    video_id      INTEGER PRIMARY KEY REFERENCES videos(video_id),
    label_status  TEXT    NOT NULL CHECK (label_status IN ('events', 'empty', 'none')),
    label_source  TEXT    NOT NULL CHECK (label_source IN ('kitware-eval', 'kitware-train', 'contrib', 'none')),
    n_events      INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS events (
    event_id      INTEGER PRIMARY KEY,
    video_id      INTEGER NOT NULL REFERENCES videos(video_id),
    camera_id     TEXT    NOT NULL,
    activity_type TEXT    NOT NULL,
    start_frame   INTEGER NOT NULL,          -- 0-based decoded-frame index, inclusive
    end_frame     INTEGER NOT NULL,          -- inclusive
    start_s       REAL    NOT NULL,          -- seconds in the clip (= proxy playback time): [start_s, end_s)
    end_s         REAL    NOT NULL,
    start_time    TEXT    NOT NULL,          -- local wall-clock ISO, accurate to about +/- 1 s (see above)
    end_time      TEXT    NOT NULL,          -- exclusive (just after end_frame)
    label_source  TEXT    NOT NULL CHECK (label_source IN ('kitware-eval', 'kitware-train', 'contrib')),
    sources       TEXT    NOT NULL,          -- JSON list: Kitware folder, or the contrib sources merged into this event
    src_ids       TEXT    NOT NULL,          -- JSON list of original activity ids, as strings (some are 64-bit hashes)
    src_status    TEXT,                      -- Kitware only; meaning undocumented (VERIFY)
    actor_types   TEXT,                      -- JSON list of Kitware object types; NULL for contrib (all 'Person')
    n_actors      INTEGER NOT NULL,
    n_actors_with_boxes INTEGER NOT NULL,
    at_clip_start INTEGER NOT NULL CHECK (at_clip_start IN (0, 1)),   -- start_frame == 0
    at_clip_end   INTEGER NOT NULL CHECK (at_clip_end IN (0, 1)),     -- end_frame == n_frames - 1
    group_id      INTEGER NOT NULL,          -- smallest event_id of the pieces of one activity cut at clip boundaries
    CHECK (0 <= start_frame AND start_frame <= end_frame)
) STRICT;

CREATE TABLE IF NOT EXISTS event_boxes (
    event_id      INTEGER NOT NULL REFERENCES events(event_id),
    actor         INTEGER NOT NULL,          -- index within the event (0, 1, ...), never an identity
    frame         INTEGER NOT NULL,
    x1 INTEGER NOT NULL, y1 INTEGER NOT NULL, x2 INTEGER NOT NULL, y2 INTEGER NOT NULL,  -- as annotated, not clipped
    PRIMARY KEY (event_id, actor, frame)
) STRICT, WITHOUT ROWID;
"""


@dataclass(frozen=True)
class VideoName:
    clip: str
    camera_id: str
    site: str
    start_time: datetime
    name_end_time: datetime


def parse_video_filename(filename: str) -> VideoName:
    """Parse e.g. '2018-03-07.10-55-06.11-00-05.school.G336.r13.avi'. Raises ValueError if malformed."""
    m = NAME_RE.match(filename)
    if not m:
        raise ValueError(f"Not a MEVA video name (date.HH-MM-SS.HH-MM-SS.site.Gnnn{VIDEO_SUFFIX}): {filename!r}")
    date, start, end, site, camera = m.groups()
    # strptime also rejects impossible dates and times (2018-02-30, 25-00-00).
    start_time = datetime.strptime(f"{date} {start}", "%Y-%m-%d %H-%M-%S")
    end_time = datetime.strptime(f"{date} {end}", "%Y-%m-%d %H-%M-%S")
    if end_time < start_time:  # e.g. 23-55-00.00-00-00: the clip ends after midnight
        end_time += timedelta(days=1)
    return VideoName(filename.removesuffix(VIDEO_SUFFIX), camera, site, start_time, end_time)


def frame_to_seconds(frame: int, fps: float) -> float:
    """Seconds from frame 0 to the start of `frame`."""
    if frame < 0:
        raise ValueError(f"Negative frame index: {frame}")
    return frame / fps


def seconds_to_frame(seconds: float, fps: float) -> int:
    """Index of the frame shown at `seconds` after frame 0 (frame f covers [f/fps, (f+1)/fps))."""
    if seconds < 0:
        raise ValueError(f"Negative time offset: {seconds}")
    return math.floor(round(seconds * fps, FRAME_SNAP_DECIMALS))


@dataclass(frozen=True)
class Video:
    """One manifest row, with frame <-> wall-clock conversion bounded to the frames that exist."""
    video_id: int
    clip: str
    path: str
    camera_id: str
    modality: str
    start_time: datetime
    n_frames: int
    fps: float

    @property
    def end_time(self) -> datetime:
        """Wall-clock instant just after the last frame (exclusive)."""
        return self.start_time + timedelta(seconds=self.n_frames / self.fps)

    def frame_time(self, frame: int) -> datetime:
        if not 0 <= frame < self.n_frames:
            raise ValueError(f"Frame {frame} outside {self.clip} (0..{self.n_frames - 1})")
        return self.start_time + timedelta(seconds=frame_to_seconds(frame, self.fps))

    def frame_end_time(self, frame: int) -> datetime:
        """Wall-clock instant just after `frame` (exclusive end of an inclusive frame range ending at `frame`).

        One rounding to the microsecond, like end_time: frame_time(f) + 1/fps would round twice and can land 1 us
        early, so the last frame's end would not equal end_time."""
        self.frame_time(frame)  # bounds check
        return self.start_time + timedelta(seconds=(frame + 1) / self.fps)

    def frame_at(self, t: datetime) -> int:
        frame = seconds_to_frame((t - self.start_time).total_seconds(), self.fps) if t >= self.start_time else -1
        if not 0 <= frame < self.n_frames:
            raise ValueError(f"{t.isoformat()} is outside {self.clip} [{self.start_time.isoformat()}, "
                             f"{self.end_time.isoformat()})")
        return frame


def probe_video(path: Path) -> dict:
    """Read stream facts from the container and demux every packet (no decoding) to check the frame count.

    Raises if the file breaks our assumptions: one video stream, constant frame rate, and exactly n_frames
    packets with decode timestamps 0..n_frames-1 (one packet per frame, none missing).
    """
    with av.open(str(path)) as container:
        if len(container.streams) != 1 or len(container.streams.video) != 1:
            raise ValueError(f"{path.name}: expected exactly one (video) stream, found {len(container.streams)}")
        stream = container.streams.video[0]
        if stream.average_rate != stream.base_rate:
            raise ValueError(f"{path.name}: variable frame rate (average {stream.average_rate}, base {stream.base_rate})")
        fps = float(stream.average_rate)
        n_frames = stream.frames
        stream_s = float(stream.duration * stream.time_base)
        if n_frames <= 0 or abs(stream_s - n_frames / fps) > 1 / fps:
            raise ValueError(f"{path.name}: header frame count {n_frames} does not match stream duration {stream_s} s")
        dts = [packet.dts for packet in container.demux(stream) if packet.size > 0]  # size 0 = end-of-stream flush
        if dts != list(range(n_frames)):
            raise ValueError(f"{path.name}: {len(dts)} packets with decode timestamps {dts[:3]}..{dts[-3:]}, "
                             f"expected 0..{n_frames - 1}")
        return {"n_frames": n_frames, "fps": fps, "duration_s": n_frames / fps,
                "width": stream.codec_context.width, "height": stream.codec_context.height,
                "codec": stream.codec_context.name}


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def load_videos(conn: sqlite3.Connection) -> list[Video]:
    rows = conn.execute("SELECT video_id, clip, path, camera_id, modality, start_time, n_frames, fps "
                        "FROM videos ORDER BY camera_id, start_time").fetchall()
    return [Video(r["video_id"], r["clip"], r["path"], r["camera_id"], r["modality"],
                  datetime.fromisoformat(r["start_time"]), r["n_frames"], r["fps"]) for r in rows]
