"""Evaluation splits (step 2.1): time blocks, all cameras of a block together, assigned to tuning / dev / test.

CLAUDE.md 1.10: split by time block, never by event. All cameras of a block go to the same split, because MEVA
cameras overlap and one activity seen by two cameras must not leak across splits.

Method (config section `splits`):
  1. Sessions. Per date, selected clips of all cameras whose wall-clock spans touch (gap <= ingest.max_name_duration_diff_s)
     form one session of continuous selected footage (20-50 min here). Between sessions is time that is either not
     recorded or not selected (only annotated clips were selected, Gate 0.3). E.g. on 03-07 MEVA has clips for
     17:10-17:20 that were not selected, so the 16:50 and 17:20 sessions are one continuous recording ~5 min apart:
     the separation between two sessions is the time gap (>= ~5 min here), not a guaranteed recording break.
  2. Blocks. A session of length L is cut into n = max(1, round(L / block_target_min)) blocks of about equal length.
     Each cut is moved to the nearest clip start in the session, and each clip goes to the block that holds its
     midpoint. The clips of one 5-min slot start up to ~8 s apart on different cameras; they land in one block.
  3. Assignment. Every assignment of blocks to splits that meets the requirements (`meets_requirements`) is
     listed in a fixed order and the seed picks one, uniformly. Only label metadata is used, never search results.
  4. Boundary events. Blocks of one session touch and the camera clocks are only accurate to about +/- 1 s, so an
     activity at a cut between two different splits can be visible in both. Such events are flagged, not deleted
     (`boundary_events`); query writing (2.2) skips them and evaluation (2.3) must treat them as "don't care".

Rule 7 (the test split is sacred): `split_video_ids` refuses the test split unless the caller says allow_test=True,
which only step 8.2 may do.
"""

import itertools
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from rewind.ingest.annotations import VEHICLE_START_STOP
from rewind.ingest.manifest import Video

SPLITS = ("tuning", "dev", "test")  # fixed by the evaluation protocol (CLAUDE.md 1.10), not a tunable
KITWARE = ("kitware-eval", "kitware-train")

SCHEMA = """
CREATE TABLE IF NOT EXISTS blocks (
    block_id      TEXT    PRIMARY KEY,       -- 'YYYY-MM-DD_HHMMSS' of the block's earliest clip start
    session       INTEGER NOT NULL,          -- 0-based, chronological; a session is continuous footage
    start_time    TEXT    NOT NULL,          -- earliest clip start (local wall-clock ISO)
    end_time      TEXT    NOT NULL,          -- latest clip end (exclusive)
    split         TEXT    NOT NULL CHECK (split IN ('tuning', 'dev', 'test')),
    n_videos      INTEGER NOT NULL,
    hours         REAL    NOT NULL           -- sum of clip durations (camera-hours)
) STRICT;
CREATE TABLE IF NOT EXISTS video_blocks (
    video_id      INTEGER PRIMARY KEY REFERENCES videos(video_id),
    block_id      TEXT    NOT NULL REFERENCES blocks(block_id)
) STRICT;
CREATE TABLE IF NOT EXISTS event_splits (
    event_id      INTEGER PRIMARY KEY REFERENCES events(event_id),
    split         TEXT    NOT NULL CHECK (split IN ('tuning', 'dev', 'test')),  -- split of the event's clip
    boundary      TEXT                       -- NULL, or why the event may be visible in another split too
) STRICT;
"""


@dataclass(frozen=True)
class Block:
    block_id: str
    session: int
    start_time: datetime   # earliest clip start
    end_time: datetime     # latest clip end (exclusive)
    video_ids: tuple[int, ...]


@dataclass
class BlockStats:
    """Label facts of a block (or a whole split). An activity cut at a clip boundary counts once, in the clip
    where it starts (its first piece, event_id == group_id)."""
    hours: float = 0.0
    eval_hours: float = 0.0        # clips labelled by Kitware evaluation-level annotation (incl. 'empty')
    kitware_hours: float = 0.0     # clips labelled by Kitware, either level
    types: Counter = field(default_factory=Counter)   # activity_type -> activities (all label sources)
    rgb_vehicle: Counter = field(default_factory=Counter)  # vehicle_starts / vehicle_stops, Kitware (RGB) only
    ir_activities: int = 0         # third-party (contrib) labelled IR activities; labels incomplete (recall only)
    ir_vehicle: int = 0            # of those, vehicle_starts + vehicle_stops
    rgb_cameras: frozenset = frozenset()                  # RGB cameras with footage here
    camera_activities: Counter = field(default_factory=Counter)  # RGB camera -> Kitware activities

    def __add__(self, other: "BlockStats") -> "BlockStats":
        return BlockStats(self.hours + other.hours, self.eval_hours + other.eval_hours,
                          self.kitware_hours + other.kitware_hours, self.types + other.types,
                          self.rgb_vehicle + other.rgb_vehicle, self.ir_activities + other.ir_activities,
                          self.ir_vehicle + other.ir_vehicle, self.rgb_cameras | other.rgb_cameras,
                          self.camera_activities + other.camera_activities)


def find_sessions(videos: list[Video], max_gap_s: float) -> list[list[Video]]:
    """Group clips of all cameras into continuous sessions, chronologically. Never joins two dates."""
    if not videos:
        raise ValueError("No videos to split")
    out: list[list[Video]] = []
    for _, day in itertools.groupby(sorted(videos, key=lambda v: (v.start_time, v.camera_id)),
                                    key=lambda v: v.start_time.date()):
        current, end = [], None
        for v in day:
            if current and (v.start_time - end).total_seconds() > max_gap_s:
                out.append(current)
                current, end = [], None
            current.append(v)
            end = v.end_time if end is None else max(end, v.end_time)
        out.append(current)
    return out


def midpoint(v: Video) -> datetime:
    return v.start_time + (v.end_time - v.start_time) / 2


def cut_blocks(sessions: list[list[Video]], target_s: float) -> list[Block]:
    """Cut each session into max(1, round(length / target_s)) blocks of about equal length (see module doc).

    round() rounds halves to even (45 min / 30 -> 2, 75 min / 30 -> 2); no real session is a tie (20-50 min)."""
    if target_s <= 0:
        raise ValueError(f"block target must be positive, got {target_s}")
    blocks = []
    for s, clips in enumerate(sessions):
        start, end = min(v.start_time for v in clips), max(v.end_time for v in clips)
        n = max(1, round((end - start).total_seconds() / target_s))
        starts = sorted({v.start_time for v in clips})
        ideal = [start + (end - start) * i / n for i in range(1, n)]
        cuts = sorted({min(starts, key=lambda t: (abs((t - c).total_seconds()), t)) for c in ideal})
        if len(cuts) != n - 1 or (cuts and cuts[0] == start):
            raise ValueError(f"session {s}: {n} blocks wanted, but cuts {ideal} snap to clip starts {cuts}; "
                             "clips are too long for block_target_min")
        bounds = [start] + cuts + [end]  # a clip's midpoint is always before the session end
        for lo, hi in itertools.pairwise(bounds):
            members = [v for v in clips if lo <= midpoint(v) < hi]
            if not members:
                raise ValueError(f"session {s}: empty block [{lo}, {hi}); cuts {cuts} are too close")
            b_start = min(v.start_time for v in members)
            blocks.append(Block(b_start.strftime("%Y-%m-%d_%H%M%S"), s, b_start, max(v.end_time for v in members),
                                tuple(sorted(v.video_id for v in members))))
    ids = [v.video_id for c in sessions for v in c]
    placed = [i for b in blocks for i in b.video_ids]
    if len({b.block_id for b in blocks}) != len(blocks):
        raise AssertionError(f"duplicate block ids: {sorted(b.block_id for b in blocks)}")
    if sorted(placed) != sorted(ids):
        raise AssertionError(f"{len(ids)} clips but {len(placed)} placed in blocks (a clip lost or placed twice)")
    return blocks


def meets_requirements(by_split: dict[str, BlockStats], cfg: dict, overall_eval_share: float,
                       rgb_cameras: frozenset) -> list[tuple[str, str]]:
    """Unmet requirements of one assignment as (requirement, actual value); empty list = valid.

    - tuning and dev each within max_fraction_diff of their target fraction of all footage hours (test takes the
      rest, so it lands within 2 x max_fraction_diff of its target);
    - every split: at least min_activity_types activity types, min_vehicle_starts / min_vehicle_stops RGB
      vehicle_starts / vehicle_stops, min_ir_activities labelled IR activities;
    - Gate 0.3 ("evaluation-level footage to dev and test wherever possible; tuning may take training-level"):
      tuning's share of evaluation-level hours among its Kitware-labelled hours is at most the overall share;
    - every split in camera_coverage_splits has at least min_activities_per_rgb_camera Kitware activities on each
      camera of rgb_cameras (all RGB cameras), so every scene of test is also in dev.
    """
    total = sum(st.hours for st in by_split.values())
    problems = []
    for split in ("tuning", "dev"):
        frac = by_split[split].hours / total
        if abs(frac - cfg["target_fraction"][split]) > cfg["max_fraction_diff"]:
            problems.append((f"{split} fraction", f"{frac:.3f}"))
    for split in SPLITS:
        st = by_split[split]
        if len(st.types) < cfg["min_activity_types"]:
            problems.append((f"{split} activity types", str(len(st.types))))
        for kind, key in (("vehicle_starts", "min_vehicle_starts"), ("vehicle_stops", "min_vehicle_stops")):
            if st.rgb_vehicle[kind] < cfg[key]:
                problems.append((f"{split} {kind}", str(st.rgb_vehicle[kind])))
        if st.ir_activities < cfg["min_ir_activities"]:
            problems.append((f"{split} IR activities", str(st.ir_activities)))
    for split in cfg["camera_coverage_splits"]:
        for camera in sorted(rgb_cameras):
            n = by_split[split].camera_activities[camera]
            if n < cfg["min_activities_per_rgb_camera"]:
                problems.append((f"{split} activities on an RGB camera", f"{camera} {n}"))
    tuning = by_split["tuning"]
    if tuning.kitware_hours == 0:
        problems.append(("tuning evaluation-level share", "no Kitware-labelled footage"))
    elif tuning.eval_hours / tuning.kitware_hours > overall_eval_share:
        problems.append(("tuning evaluation-level share", f"{tuning.eval_hours / tuning.kitware_hours:.3f}"))
    return problems


def block_stats(blocks: list[Block], videos: list[Video], labels: dict[int, str], events) -> dict[str, BlockStats]:
    """BlockStats per block. labels: video_id -> clip_labels.label_source. events: rows with event_id, group_id,
    video_id, activity_type, label_source. A cut activity counts once, at the piece with event_id == group_id
    (the smallest id of the group; on the real data always its earliest piece)."""
    block_of = {vid: b.block_id for b in blocks for vid in b.video_ids}
    stats = {b.block_id: BlockStats() for b in blocks}
    camera = {v.video_id: v.camera_id for v in videos}
    for v in videos:
        st, hours = stats[block_of[v.video_id]], v.n_frames / v.fps / 3600
        st.hours += hours
        if v.modality == "rgb":
            st.rgb_cameras |= {v.camera_id}
        if labels[v.video_id] in KITWARE:
            st.kitware_hours += hours
        if labels[v.video_id] == "kitware-eval":
            st.eval_hours += hours
    for e in events:
        if e["event_id"] != e["group_id"]:
            continue
        st, kind = stats[block_of[e["video_id"]]], e["activity_type"]
        st.types[kind] += 1
        if e["label_source"] == "contrib":
            st.ir_activities += 1
            st.ir_vehicle += kind in VEHICLE_START_STOP
        else:
            st.camera_activities[camera[e["video_id"]]] += 1
            if kind in VEHICLE_START_STOP:
                st.rgb_vehicle[kind] += 1
    return stats


def split_stats(assignment: dict[str, str], stats: dict[str, BlockStats]) -> dict[str, BlockStats]:
    by_split = {s: BlockStats() for s in SPLITS}
    for block_id, split in assignment.items():
        by_split[split] = by_split[split] + stats[block_id]
    return by_split


def all_assignments(block_ids: list[str]) -> list[dict[str, str]]:
    """Every way to give each block one split with all three splits non-empty, in a fixed order."""
    ids = sorted(block_ids)
    out = []
    for labels in itertools.product(SPLITS, repeat=len(ids)):
        if "tuning" in labels and "dev" in labels and "test" in labels:
            out.append(dict(zip(ids, labels)))
    return out


def choose_assignment(stats: dict[str, BlockStats], cfg: dict) -> tuple[dict[str, str], int, int, Counter]:
    """(chosen assignment, number of valid assignments, number of assignments checked, how often each requirement
    failed over all assignments).

    Raises if no assignment meets the requirements. Exhaustive (3^n): fine for the ~10 blocks here; refuses above
    max_blocks_enumerated rather than run for hours."""
    fractions = cfg["target_fraction"]
    if set(fractions) != set(SPLITS) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError(f"splits.target_fraction must give {SPLITS} summing to 1, got {fractions}")
    if len(stats) > cfg["max_blocks_enumerated"]:
        raise ValueError(f"{len(stats)} blocks: 3^{len(stats)} assignments is too many to enumerate")
    total = sum(st.hours for st in stats.values())
    overall = sum(st.eval_hours for st in stats.values()) / sum(st.kitware_hours for st in stats.values())
    rgb_cameras = frozenset().union(*(st.rgb_cameras for st in stats.values()))
    valid, failures = [], Counter()
    candidates = all_assignments(list(stats))
    for assignment in candidates:
        problems = meets_requirements(split_stats(assignment, stats), cfg, overall, rgb_cameras)
        if problems:
            failures.update(requirement for requirement, _ in problems)
        else:
            valid.append(assignment)
    if not valid:
        raise ValueError(f"No assignment of {len(stats)} blocks ({total:.2f} h) meets the requirements; "
                         f"failures: {dict(failures)}")
    return random.Random(cfg["seed"]).choice(valid), len(valid), len(candidates), failures


@dataclass(frozen=True)
class EventSpan:
    event_id: int
    group_id: int
    block_id: str
    start_time: datetime
    end_time: datetime   # exclusive


def boundary_events(events: list[EventSpan], blocks: list[Block], assignment: dict[str, str],
                    pad_s: float) -> dict[int, str]:
    """{event_id: reason} for events that may be visible in a split other than their own.

    - 'overlaps <block>': the event's wall-clock span, widened by pad_s on both sides (clock accuracy), overlaps the
      span of a block in another split (some camera there may show the same moment);
    - 'group spans splits': pieces of one activity cut at clip boundaries are in different splits.
    Every piece of a group with a flagged piece is flagged too ('group: ...').
    """
    pad = timedelta(seconds=pad_s)
    reasons: dict[int, str] = {}
    for e in events:
        own = assignment[e.block_id]
        for b in blocks:
            if assignment[b.block_id] != own and e.start_time - pad < b.end_time and b.start_time < e.end_time + pad:
                reasons[e.event_id] = f"overlaps {b.block_id}"
                break
    groups: dict[int, list[EventSpan]] = {}
    for e in events:
        groups.setdefault(e.group_id, []).append(e)
    for members in groups.values():
        if len({assignment[e.block_id] for e in members}) > 1:
            for e in members:
                reasons.setdefault(e.event_id, "group spans splits")
        flagged = [reasons[e.event_id] for e in members if e.event_id in reasons]
        if flagged:
            for e in members:
                reasons.setdefault(e.event_id, f"group: {flagged[0]}")
    return reasons


def check_split_access(split: str, allow_test: bool) -> None:
    """The test split is refused unless allow_test=True (step 8.2 only, CLAUDE.md 1.1.7)."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    if split == "test" and not allow_test:
        raise PermissionError("The test split is read only by step 8.2 (CLAUDE.md 1.1.7); pass allow_test=True there")


def split_video_ids(conn: sqlite3.Connection, split: str, allow_test: bool = False) -> list[int]:
    """Video ids of one split (test refused unless allow_test, see check_split_access)."""
    check_split_access(split, allow_test)
    rows = conn.execute("SELECT vb.video_id FROM video_blocks vb JOIN blocks b USING (block_id) WHERE b.split = ? "
                        "ORDER BY vb.video_id", (split,)).fetchall()
    if not rows:
        raise ValueError(f"No videos in split {split!r}: has scripts.make_splits been run?")
    return [r[0] for r in rows]


def split_event_ids(conn: sqlite3.Connection, split: str, allow_test: bool = False,
                    include_boundary: bool = False) -> list[int]:
    """Event ids of one split (test refused unless allow_test). Boundary-flagged events are left out unless
    include_boundary: they may be visible in another split, so no query may be written about them (2.2)."""
    check_split_access(split, allow_test)
    if not conn.execute("SELECT COUNT(*) FROM event_splits").fetchone()[0]:
        raise ValueError("event_splits is empty: has scripts.make_splits been run?")
    sql = "SELECT event_id FROM event_splits WHERE split = ?" + ("" if include_boundary else " AND boundary IS NULL")
    return [r[0] for r in conn.execute(sql + " ORDER BY event_id", (split,))]
