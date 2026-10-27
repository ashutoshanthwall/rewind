"""Step 2.1: sessions, blocks, assignment requirements, boundary events, the test-split guard, and the committed split.

Toy cases have hand-computed answers in comments. The last tests check the real committed split
(eval/splits/*.csv): no block in two splits, every clip in exactly one block.
"""

import csv
import itertools
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from rewind.eval.splits import (SCHEMA, SPLITS, Block, BlockStats, EventSpan, block_stats, boundary_events,
                                choose_assignment, cut_blocks, find_sessions, split_event_ids, split_video_ids)
from rewind.ingest.manifest import Video, connect
from scripts import make_splits

REPO_ROOT = Path(__file__).resolve().parent.parent
T0 = datetime(2018, 3, 11, 10, 0, 0)


def clip(vid: int, cam: str, start: datetime, seconds: float = 300.0, modality: str = "rgb") -> Video:
    return Video(vid, f"c{vid}", f"p{vid}", cam, modality, start, round(seconds * 30), 30.0)


def slots(n: int, offsets: dict[str, int], t0: datetime = T0) -> list[Video]:
    """n back-to-back 5-min slots; camera cam starts offsets[cam] seconds after each slot start."""
    out = []
    for i in range(n):
        for cam, off in offsets.items():
            out.append(clip(len(out), cam, t0 + timedelta(seconds=300 * i + off)))
    return out


# --- sessions -----------------------------------------------------------------------------------------------------

def test_sessions_split_on_gaps_larger_than_tolerance():
    # A: 10:00:00-10:05:00, then 10:05:02-10:10:02 (gap 2 s = tolerance -> same session), then 10:10:05
    # (gap 3 s -> new session).
    a = clip(1, "A", T0)
    b = clip(2, "A", T0 + timedelta(seconds=302))           # gap 2.0 s
    c = clip(3, "A", T0 + timedelta(seconds=605))           # previous ends 10:10:02 -> gap 3.0 s
    sessions = find_sessions([c, a, b], max_gap_s=2.0)
    assert [[v.video_id for v in s] for s in sessions] == [[1, 2], [3]]


def test_sessions_span_all_cameras_and_never_cross_dates():
    # B covers the gap A leaves (10:05-10:10), so A's two clips are one session. The clip starting at midnight
    # touches the previous day's last clip (gap 0) but is on another date: separate session.
    a1, a2 = clip(1, "A", T0), clip(2, "A", T0 + timedelta(minutes=10))
    b1 = clip(3, "B", T0 + timedelta(minutes=5))
    late = clip(4, "A", datetime(2018, 3, 11, 23, 55))
    midnight = clip(5, "A", datetime(2018, 3, 12, 0, 0))
    sessions = find_sessions([a1, a2, b1, late, midnight], max_gap_s=2.0)
    assert [sorted(v.video_id for v in s) for s in sessions] == [[1, 2, 3], [4], [5]]


def test_session_end_is_the_running_maximum():
    # A long clip 10:00-10:30, a short one 10:01-10:06, then 10:10: the session runs to 10:30, so 10:10 joins it.
    videos = [clip(1, "A", T0, 30 * 60), clip(2, "B", T0 + timedelta(minutes=1)), clip(3, "C", T0 + timedelta(minutes=10))]
    assert [[v.video_id for v in s] for s in find_sessions(videos, max_gap_s=2.0)] == [[1, 2, 3]]


def test_sessions_refuse_empty_input():
    with pytest.raises(ValueError, match="No videos"):
        find_sessions([], max_gap_s=2.0)


# --- blocks -------------------------------------------------------------------------------------------------------

def test_fifty_minute_session_is_cut_in_two_and_cameras_stay_together():
    # 10 slots, camera B starts 8 s after A: session 10:00:00-10:50:08 (50.13 min) -> round(50.13/30) = 2 blocks.
    # Ideal cut 10:25:04 -> nearest clip starts 10:25:00 and 10:25:08 are both 4 s away; the earlier wins (either
    # gives the same blocks under the midpoint rule).
    # B's clip 10:20:08-10:25:08 has midpoint 10:22:38 < 10:25:00 -> first block, with A's 10:20 clip.
    videos = slots(10, {"A": 0, "B": 8})
    blocks = cut_blocks(find_sessions(videos, 2.0), target_s=30 * 60)
    assert len(blocks) == 2
    first, second = blocks
    assert first.start_time == T0 and second.start_time == T0 + timedelta(minutes=25)
    assert first.end_time == T0 + timedelta(minutes=25, seconds=8)       # B's 10:20:08 clip ends 10:25:08
    assert second.end_time == T0 + timedelta(minutes=50, seconds=8)
    assert first.video_ids == tuple(range(10)) and second.video_ids == tuple(range(10, 20))
    assert first.block_id == "2018-03-11_100000" and second.block_id == "2018-03-11_102500"


def test_midpoint_rule_keeps_a_slot_together_when_the_cut_is_a_late_camera_start():
    # As above plus one clip on camera C at 10:50: session 10:00-10:55 -> 2 blocks, ideal cut 10:27:30. Nearest clip
    # start is B's 10:25:08 (2:22 away; 10:25:00 and 10:30:00 are 2:30 away). A's 10:25 clip has midpoint 10:27:30
    # >= 10:25:08 -> second block, with B's 10:25:08 clip: the slot stays together. (Assigning by start time would
    # put A's 10:25 clip in the first block and B's in the second: blocks overlapping by 4:52.)
    videos = slots(10, {"A": 0, "B": 8}) + [clip(20, "C", T0 + timedelta(minutes=50))]
    first, second = cut_blocks(find_sessions(videos, 2.0), target_s=30 * 60)
    assert first.video_ids == tuple(range(10))                          # slots 10:00-10:20, both cameras
    assert first.end_time == T0 + timedelta(minutes=25, seconds=8)
    assert second.start_time == T0 + timedelta(minutes=25)               # A's 10:25 clip
    assert second.video_ids == tuple(range(10, 21))
    assert second.end_time == T0 + timedelta(minutes=55)


@pytest.mark.parametrize("n_slots, expected_blocks", [
    (4, 1),    # 20 min: round(0.67) = 1
    (8, 1),    # 40 min: round(1.33) = 1
    (9, 2),    # 45 min: round(1.5) = 2 (Python rounds half to even: 2)
    (13, 2),   # 65 min: round(2.17) = 2
    (17, 3),   # 85 min: round(2.83) = 3
])
def test_number_of_blocks_per_session(n_slots, expected_blocks):
    videos = slots(n_slots, {"A": 0})
    blocks = cut_blocks(find_sessions(videos, 2.0), target_s=30 * 60)
    assert len(blocks) == expected_blocks
    placed = [vid for b in blocks for vid in b.video_ids]
    assert sorted(placed) == sorted(v.video_id for v in videos)          # every clip in exactly one block


def test_three_blocks_are_cut_at_nearest_clip_starts():
    # 17 slots of camera A only: 10:00-11:25 (85 min) -> 3 blocks. Ideal cuts at 28:20 and 56:40 after 10:00;
    # nearest clip starts are 10:30 (1:40 away, vs 3:20 for 10:25) and 10:55 (1:40 away, vs 3:20 for 11:00).
    blocks = cut_blocks(find_sessions(slots(17, {"A": 0}), 2.0), target_s=30 * 60)
    assert [b.start_time.strftime("%H:%M") for b in blocks] == ["10:00", "10:30", "10:55"]
    assert [len(b.video_ids) for b in blocks] == [6, 5, 6]


def test_blocks_refuse_cuts_that_collapse_onto_one_clip_start():
    # Two 45-min clips (90 min) with a 30-min target want 3 blocks, but the only inner clip start is 10:45: both
    # ideal cuts (10:30, 11:00) snap to it, which would silently give 2 blocks.
    videos = [clip(1, "A", T0, 45 * 60), clip(2, "A", T0 + timedelta(minutes=45), 45 * 60)]
    with pytest.raises(ValueError, match="3 blocks wanted"):
        cut_blocks(find_sessions(videos, 2.0), target_s=30 * 60)


def test_blocks_refuse_non_positive_target():
    with pytest.raises(ValueError, match="positive"):
        cut_blocks(find_sessions(slots(2, {"A": 0}), 2.0), target_s=0)


# --- assignment ---------------------------------------------------------------------------------------------------

CFG = {"seed": 7, "target_fraction": {"tuning": 0.15, "dev": 0.15, "test": 0.70}, "max_fraction_diff": 0.05,
       "min_activity_types": 0, "min_vehicle_starts": 0, "min_vehicle_stops": 0, "min_ir_activities": 0,
       "min_activities_per_rgb_camera": 0, "camera_coverage_splits": ["dev", "test"], "max_blocks_enumerated": 12}


def stats(hours: float, eval_hours: float, starts: int = 5, types: int = 3, ir: int = 5,
          cameras: dict[str, int] | None = None, stops: int | None = None) -> BlockStats:
    cameras = {"A": 30} if cameras is None else cameras
    return BlockStats(hours=hours, eval_hours=eval_hours, kitware_hours=hours,
                      types=Counter({f"t{i}": 1 for i in range(types)}),
                      rgb_vehicle=Counter({"vehicle_starts": starts, "vehicle_stops": starts if stops is None else stops}),
                      ir_activities=ir,
                      rgb_cameras=frozenset(cameras), camera_activities=Counter(cameras))


def test_only_assignment_meeting_fractions_and_gate_rule_is_chosen():
    # Total 10 h. Tuning and dev must each hold 1.0-2.0 h -> one of A, B each (1.5 h = 15%); C+D (7 h) go to test.
    # Overall evaluation-level share = (1.5 + 0 + 1.75 + 1.75) / 10 = 0.5. Tuning = A has share 1.0 > 0.5 (fails);
    # tuning = B has share 0 (ok). Exactly one valid assignment: B tuning, A dev, C and D test.
    blocks = {"A": stats(1.5, 1.5), "B": stats(1.5, 0.0), "C": stats(3.5, 1.75), "D": stats(3.5, 1.75)}
    chosen, n_valid, n_checked, failures = choose_assignment(blocks, CFG)
    assert n_valid == 1
    assert n_checked == 3 ** 4 - 3 * 2 ** 4 + 3      # all three splits non-empty: 81 - 48 + 3 = 36
    assert chosen == {"A": "dev", "B": "tuning", "C": "test", "D": "test"}
    # The rule fails iff A is in tuning and B is not (shares: A 1.0, A+C 0.65, A+D 0.65, A+C+D 0.59; with B 0.5).
    # A tuning, B dev/test (2), C and D anywhere (9) = 18, minus no dev (B, C, D in tuning/test: 4) and no test
    # (B, C, D in tuning/dev: 4) = 10 assignments.
    assert failures["tuning evaluation-level share"] == 10


def test_minimum_counts_are_enforced_for_every_split():
    # Same blocks, but B (the only possible tuning block) has no vehicle starts -> nothing is valid.
    blocks = {"A": stats(1.5, 1.5), "B": stats(1.5, 0.0, starts=0), "C": stats(3.5, 1.75), "D": stats(3.5, 1.75)}
    with pytest.raises(ValueError, match="No assignment"):
        choose_assignment(blocks, {**CFG, "min_vehicle_starts": 1})


@pytest.mark.parametrize("requirement, cfg_key, block_kwargs", [
    ("test vehicle_starts", "min_vehicle_starts", {"starts": 0, "stops": 5}),
    ("test vehicle_stops", "min_vehicle_stops", {"starts": 5, "stops": 0}),
    ("test IR activities", "min_ir_activities", {"ir": 0}),
    ("test activity types", "min_activity_types", {"types": 2}),
    ("test activities on an RGB camera", "min_activities_per_rgb_camera", {"cameras": {"A": 30, "G": 0}}),
])
def test_each_minimum_is_enforced_on_the_test_split(requirement, cfg_key, block_kwargs):
    # The 4-block case above: the only valid assignment puts C and D in test. Make C and D (all of test) miss one
    # minimum (threshold 1, or 3 types, or 20 per camera); A and B meet it -> nothing is valid, and the failure is
    # reported for the test split.
    ok = {"cameras": {"A": 30, "G": 50}} if cfg_key == "min_activities_per_rgb_camera" else {}
    blocks = {"A": stats(1.5, 1.5, **ok), "B": stats(1.5, 0.0, **ok),
              "C": stats(3.5, 1.75, **block_kwargs), "D": stats(3.5, 1.75, **block_kwargs)}
    threshold = {"min_activity_types": 3, "min_activities_per_rgb_camera": 20}.get(cfg_key, 1)
    with pytest.raises(ValueError, match=requirement):
        choose_assignment(blocks, {**CFG, cfg_key: threshold})


def test_per_camera_minimum_is_inclusive():
    # As above, but C and D have 10 activities each on G: test has exactly 20 = the minimum -> valid (1 assignment).
    ok = {"cameras": {"A": 30, "G": 50}}
    low = {"cameras": {"A": 30, "G": 10}}
    blocks = {"A": stats(1.5, 1.5, **ok), "B": stats(1.5, 0.0, **ok),
              "C": stats(3.5, 1.75, **low), "D": stats(3.5, 1.75, **low)}
    _, n_valid, _, _ = choose_assignment(blocks, {**CFG, "min_activities_per_rgb_camera": 20})
    assert n_valid == 1


def test_every_rgb_camera_needs_activities_in_coverage_splits():
    # Seven blocks as in the seed test below (42 valid without the rule), but camera G only has activities in
    # B0 (50) and B1 (50); B2..B5 and S have none on G (G has footage everywhere). With >= 20 per camera in dev
    # and test, dev must be B0 or B1 (dev is a single block) and test must hold the other -> tuning is one of the
    # other 5 blocks: 2 (dev choices) x 5 (tuning choices) = 10 valid. Tuning is not a coverage split.
    blocks = {f"B{i}": stats(1.5, 0.75, cameras={"A": 30, "G": 50 if i < 2 else 0}) for i in range(6)}
    blocks |= {"S": stats(1.0, 0.5, cameras={"A": 30, "G": 0})}
    chosen, n_valid, _, failures = choose_assignment(blocks, {**CFG, "min_activities_per_rgb_camera": 20})
    assert n_valid == 10
    assert chosen["B0"] != chosen["B1"] and {chosen["B0"], chosen["B1"]} == {"dev", "test"}
    assert failures["dev activities on an RGB camera"] > 0


def test_seed_picks_among_valid_assignments_reproducibly():
    # Six 1.5 h blocks + one 1 h block (10 h); all evaluation shares equal, so only fractions matter.
    # Tuning: one 1.5 h block or the 1 h block (1.0-2.0 h). Valid assignments: ordered (tuning, dev) pairs of distinct
    # blocks among 7 = 42, test gets the other 5 (6.5 or 7.0 h).
    blocks = {f"B{i}": stats(1.5, 0.75) for i in range(6)} | {"S": stats(1.0, 0.5)}
    first, n_valid, _, _ = choose_assignment(blocks, CFG)
    again, _, _, _ = choose_assignment(blocks, CFG)
    assert n_valid == 42
    assert first == again
    assert Counter(first.values()) == Counter({"tuning": 1, "dev": 1, "test": 5})
    other_seeds = {tuple(sorted(choose_assignment(blocks, {**CFG, "seed": s})[0].items())) for s in range(20)}
    assert len(other_seeds) > 1                                # the seed really chooses


@pytest.mark.parametrize("fractions", [{"tuning": 0.15, "dev": 0.15, "test": 0.60},
                                       {"tuning": 0.15, "dev": 0.15, "tset": 0.70}])
def test_target_fractions_must_name_all_splits_and_sum_to_one(fractions):
    blocks = {"A": stats(1.5, 1.5), "B": stats(1.5, 0.0), "C": stats(3.5, 1.75), "D": stats(3.5, 1.75)}
    with pytest.raises(ValueError, match="target_fraction"):
        choose_assignment(blocks, {**CFG, "target_fraction": fractions})


def test_enumeration_refuses_too_many_blocks():
    blocks = {f"B{i}": stats(1.0, 0.5) for i in range(13)}
    with pytest.raises(ValueError, match="too many"):
        choose_assignment(blocks, CFG)


def test_block_stats_counts_hours_labels_and_activities_once():
    # One block, three 300 s clips: kitware-eval RGB, kitware-train RGB, contrib IR.
    # Hours 3 x 300 s = 0.25 h; Kitware 0.1667 h, of which evaluation-level 0.0833 h.
    # Events: a vehicle_stops activity cut into 2 pieces (ids 10, 11, group 10) -> 1 activity; a Kitware
    # vehicle_starts (12); an IR (contrib) vehicle_starts (13) -> IR, not RGB; an IR person_talks_to_person (14).
    videos = [clip(1, "A", T0), clip(2, "B", T0), clip(3, "C", T0, modality="ir")]
    blocks = cut_blocks(find_sessions(videos, 2.0), target_s=30 * 60)
    labels = {1: "kitware-eval", 2: "kitware-train", 3: "contrib"}
    ev = lambda i, g, vid, kind, src: {"event_id": i, "group_id": g, "video_id": vid, "activity_type": kind,
                                       "label_source": src}
    events = [ev(10, 10, 1, "vehicle_stops", "kitware-eval"), ev(11, 10, 1, "vehicle_stops", "kitware-eval"),
              ev(12, 12, 2, "vehicle_starts", "kitware-train"), ev(13, 13, 3, "vehicle_starts", "contrib"),
              ev(14, 14, 3, "person_talks_to_person", "contrib")]
    (st,) = block_stats(blocks, videos, labels, events).values()
    assert st.hours == pytest.approx(0.25)
    assert st.kitware_hours == pytest.approx(1 / 6) and st.eval_hours == pytest.approx(1 / 12)
    assert st.types == Counter({"vehicle_stops": 1, "vehicle_starts": 2, "person_talks_to_person": 1})
    assert st.rgb_vehicle == Counter({"vehicle_stops": 1, "vehicle_starts": 1})
    assert st.ir_activities == 2 and st.ir_vehicle == 1
    assert st.rgb_cameras == {"A", "B"}                                   # C is IR
    assert st.camera_activities == Counter({"A": 1, "B": 1})            # Kitware activities only (10 once, 12)


# --- boundary events ----------------------------------------------------------------------------------------------

def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def test_boundary_events_overlap_pad_and_groups():
    # X = [10:00, 10:25:08) tuning, Y = [10:25:00, 10:50) dev (blocks of one session touch), pad 2 s.
    x = Block("X", 0, at(0), at(25) + timedelta(seconds=8), (1,))
    y = Block("Y", 0, at(25), at(50), (2,))
    assignment = {"X": "tuning", "Y": "dev"}
    s = timedelta(seconds=1)
    events = [
        EventSpan(1, 1, "X", at(24) + 59 * s, at(25) + 3 * s),           # inside Y's span -> flagged
        EventSpan(2, 2, "X", at(20), at(22)),                              # far from Y -> not flagged
        EventSpan(3, 3, "X", at(24) + 50 * s, at(24) + 57.5 * s),          # ends 10:24:57.5 + 2 s pad < 10:25 -> not
        EventSpan(4, 4, "X", at(24) + 50 * s, at(24) + 58.5 * s),          # 10:24:58.5 + 2 s > 10:25 -> flagged
        EventSpan(5, 5, "X", at(10), at(11)),                              # group 5 spans X (tuning) and Y (dev)
        EventSpan(6, 5, "Y", at(40), at(41)),
        EventSpan(7, 7, "Y", at(30), at(31)),                              # same group as a flagged piece
        EventSpan(8, 7, "Y", at(25), at(25) + 4 * s),                      # overlaps X's span
        EventSpan(9, 9, "X", at(24) + 50 * s, at(24) + 58 * s),            # end + pad = 10:25:00 = Y start: not (exclusive)
        EventSpan(10, 10, "Y", at(25) + 10 * s, at(26)),                   # start - pad = 10:25:08 = X end: not
    ]
    reasons = boundary_events(events, [x, y], assignment, pad_s=2.0)
    assert reasons == {1: "overlaps Y", 4: "overlaps Y", 5: "group spans splits", 6: "group spans splits",
                       8: "overlaps X", 7: "group: overlaps X"}
    # Same split on both sides: nothing can leak, nothing is flagged.
    assert boundary_events(events, [x, y], {"X": "test", "Y": "test"}, pad_s=2.0) == {}


# --- test-split guard ---------------------------------------------------------------------------------------------

def test_split_video_ids_refuses_test_unless_allowed():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO blocks VALUES (?, 0, '', '', ?, 1, 1.0)", [("b1", "tuning"), ("b2", "test")])
    conn.executemany("INSERT INTO video_blocks VALUES (?, ?)", [(11, "b1"), (12, "b1"), (21, "b2")])
    assert split_video_ids(conn, "tuning") == [11, 12]
    with pytest.raises(PermissionError, match="8.2"):
        split_video_ids(conn, "test")
    assert split_video_ids(conn, "test", allow_test=True) == [21]
    with pytest.raises(ValueError, match="No videos"):
        split_video_ids(conn, "dev")
    with pytest.raises(ValueError, match="Unknown split"):
        split_video_ids(conn, "train")


def test_split_event_ids_refuses_test_and_leaves_out_boundary_events():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    with pytest.raises(ValueError, match="empty"):
        split_event_ids(conn, "dev")
    conn.executemany("INSERT INTO event_splits VALUES (?, ?, ?)",
                     [(1, "dev", None), (2, "dev", "overlaps X"), (3, "test", None)])
    assert split_event_ids(conn, "dev") == [1]
    assert split_event_ids(conn, "dev", include_boundary=True) == [1, 2]
    with pytest.raises(PermissionError):
        split_event_ids(conn, "test")
    assert split_event_ids(conn, "test", allow_test=True) == [3]


# --- the committed split ------------------------------------------------------------------------------------------

def read_csv(name: str) -> list[dict]:
    path = REPO_ROOT / "eval" / "splits" / name
    assert path.exists(), f"{path} missing: run `uv run python -m scripts.make_splits` (step 2.1)"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_committed_split_no_block_in_two_splits():
    # The generator gives each block one split by construction (a dict); this checks the committed files themselves
    # (e.g. a hand edit). The wall-clock overlap test below is the check against leaks.
    blocks, clips = read_csv("blocks.csv"), read_csv("clips.csv")
    block_split = {}
    for b in blocks:
        assert b["block_id"] not in block_split, f"block {b['block_id']} listed twice"
        block_split[b["block_id"]] = b["split"]
    splits_per_block: dict[str, set] = {}
    for c in clips:
        splits_per_block.setdefault(c["block_id"], set()).add(c["split"])
    assert {b: s for b, s in splits_per_block.items() if len(s) > 1} == {}       # no block in two splits
    assert {b: s.pop() for b, s in splits_per_block.items()} == block_split      # clips agree with blocks.csv
    assert set(block_split.values()) == set(SPLITS)


def test_committed_split_every_clip_in_one_block_and_counts_match():
    blocks, clips = read_csv("blocks.csv"), read_csv("clips.csv")
    names = [c["clip"] for c in clips]
    assert len(names) == len(set(names))
    per_block = Counter(c["block_id"] for c in clips)
    assert {b["block_id"]: int(b["n_videos"]) for b in blocks} == dict(per_block)


def test_committed_split_fractions_within_config_band():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())["splits"]
    hours = Counter()
    for b in read_csv("blocks.csv"):
        hours[b["split"]] += float(b["hours"])
    total = sum(hours.values())
    for split in ("tuning", "dev"):
        assert abs(hours[split] / total - cfg["target_fraction"][split]) <= cfg["max_fraction_diff"]


def test_committed_split_blocks_of_different_splits_never_overlap_in_time():
    # The leak that matters: one wall-clock moment in two splits (another camera may show it), padded by the clock
    # accuracy (splits.boundary_pad_s). Only blocks of one session may touch; their events at the cut are
    # boundary-flagged (stored flags checked in the DB test below). Blocks of different sessions must never overlap.
    pad = timedelta(seconds=yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())["splits"]["boundary_pad_s"])
    blocks = [(b["block_id"], b["split"], datetime.fromisoformat(b["start_time"]), datetime.fromisoformat(b["end_time"]),
               b["session"]) for b in read_csv("blocks.csv")]
    leaks = [(a[0], b[0]) for a, b in itertools.combinations(blocks, 2)
             if a[1] != b[1] and a[2] - pad < b[3] and b[2] < a[3] + pad and a[4] != b[4]]
    assert leaks == []


def test_database_split_matches_committed_split():
    db = REPO_ROOT / "data" / "rewind.sqlite"
    if not db.exists():
        pytest.skip("data/rewind.sqlite not present (fresh clone)")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    in_db = dict(conn.execute("SELECT v.clip, b.split FROM videos v JOIN video_blocks USING (video_id) "
                              "JOIN blocks b USING (block_id)").fetchall())
    assert in_db == {c["clip"]: c["split"] for c in read_csv("clips.csv")}
    n_videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    assert len(in_db) == n_videos
    # every event has the split of its clip's block
    mismatched = conn.execute("SELECT COUNT(*) FROM events e LEFT JOIN event_splits s USING (event_id) "
                              "JOIN video_blocks USING (video_id) JOIN blocks b USING (block_id) "
                              "WHERE s.split IS NULL OR s.split != b.split").fetchone()[0]
    assert mismatched == 0
    # the stored boundary flags are what boundary_events gives for the stored split
    assert stored_flags(conn) == recomputed_flags(conn, pad_s=yaml.safe_load(
        (REPO_ROOT / "configs" / "config.yaml").read_text())["splits"]["boundary_pad_s"])


def stored_flags(conn) -> dict[int, str]:
    return dict(conn.execute("SELECT event_id, boundary FROM event_splits WHERE boundary IS NOT NULL"))


def recomputed_flags(conn, pad_s: float) -> dict[int, str]:
    """boundary_events over the blocks, clip memberships and events stored in the DB."""
    rows = conn.execute("SELECT block_id, session, start_time, end_time, split FROM blocks").fetchall()
    members: dict[str, list[int]] = {}
    for vid, bid in conn.execute("SELECT video_id, block_id FROM video_blocks"):
        members.setdefault(bid, []).append(vid)
    blocks = [Block(b, s, datetime.fromisoformat(a), datetime.fromisoformat(e), tuple(sorted(members[b])))
              for b, s, a, e, _ in rows]
    block_of = {vid: b.block_id for b in blocks for vid in b.video_ids}
    spans = [EventSpan(e, g, block_of[v], datetime.fromisoformat(a), datetime.fromisoformat(z)) for e, g, v, a, z in
             conn.execute("SELECT event_id, group_id, video_id, start_time, end_time FROM events")]
    return boundary_events(spans, blocks, {r[0]: r[4] for r in rows}, pad_s)


# --- the script, end to end on a toy database ---------------------------------------------------------------------

TOY_CONFIG = """
ingest: {max_name_duration_diff_s: 2.0}
splits:
  seed: 2                        # puts the 15:00 block in test and the 15:05 block in dev (boundary flags exercised)
  block_target_min: 5
  target_fraction: {tuning: 0.15, dev: 0.15, test: 0.70}
  max_fraction_diff: 0.05
  min_activity_types: 0
  min_vehicle_starts: 0
  min_vehicle_stops: 0
  min_ir_activities: 0
  min_activities_per_rgb_camera: 0
  camera_coverage_splits: [dev, test]
  max_blocks_enumerated: 12
  boundary_pad_s: 2.0
"""


def toy_db(path: Path) -> None:
    """9 clips of camera A (300 s, kitware-eval): 7 single-clip sessions 08:00..14:00, and 15:00 + 15:05 back to back
    (one session, cut into 2 blocks at 15:05 with a 5-min target) -> 9 blocks of 1/9 = 11.1 % each. One activity
    is cut at 15:05 (2 pieces, group 8), one sits at the cut (15:04:59-15:05:00), one is far from it."""
    conn = connect(path)
    starts = [datetime(2018, 3, 11, h) for h in range(8, 15)] + [datetime(2018, 3, 11, 15), datetime(2018, 3, 11, 15, 5)]
    for vid, st in enumerate(starts, start=1):
        conn.execute("INSERT INTO videos VALUES (?, ?, ?, 1, 'x', 'A', 'site', 'rgb', ?, ?, 300.0, 9000, 30.0, "
                     "1920, 1080, 'h264', 'now')", (vid, f"c{vid}", f"p{vid}", st.isoformat(),
                                                     (st + timedelta(seconds=300)).isoformat()))
        conn.execute("INSERT INTO clip_labels VALUES (?, 'events', 'kitware-eval', 1)", (vid,))
    def event(eid, vid, s0, s1, group):
        st = starts[vid - 1]
        conn.execute("INSERT INTO events VALUES (?, ?, 'A', 'person_sits_down', ?, ?, ?, ?, ?, ?, 'kitware-eval', "
                     "'[]', '[]', NULL, NULL, 1, 1, 0, 0, ?)",
                     (eid, vid, s0 * 30, s1 * 30 - 1, s0, s1, (st + timedelta(seconds=s0)).isoformat(),
                      (st + timedelta(seconds=s1)).isoformat(), group))
    event(8, 8, 290, 300, 8)
    event(9, 9, 0, 10, 8)
    event(10, 8, 299, 300, 10)
    event(11, 1, 100, 110, 11)
    conn.commit()
    conn.close()


def run_script(monkeypatch, tmp_path: Path, *extra: str) -> None:
    monkeypatch.setattr(make_splits, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(make_splits, "SPLITS_DIR", tmp_path / "eval" / "splits")
    monkeypatch.setattr(make_splits, "QUERIES_DIR", tmp_path / "eval" / "queries")
    monkeypatch.setattr(make_splits, "git_hash", lambda _: "toy")  # tmp_path is not a git repo
    monkeypatch.setattr(sys, "argv", ["make_splits", "--db", str(tmp_path / "toy.sqlite"), "--config",
                                      str(tmp_path / "config.yaml"), "--report", str(tmp_path / "report.md"), *extra])
    make_splits.main()


def test_script_end_to_end_on_toy_database(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(TOY_CONFIG)
    toy_db(tmp_path / "toy.sqlite")
    run_script(monkeypatch, tmp_path)
    conn = sqlite3.connect(tmp_path / "toy.sqlite")
    db_split = dict(conn.execute("SELECT v.clip, b.split FROM videos v JOIN video_blocks USING (video_id) "
                                 "JOIN blocks b USING (block_id)"))
    clips_csv = tmp_path / "eval" / "splits" / "clips.csv"
    with open(clips_csv, newline="") as f:
        assert {r["clip"]: r["split"] for r in csv.DictReader(f)} == db_split
    assert len(db_split) == 9 and Counter(db_split.values()) == Counter({"tuning": 1, "dev": 1, "test": 7})
    # boundary column = boundary_events recomputed from what was written
    flagged = stored_flags(conn)
    assert flagged == recomputed_flags(conn, pad_s=2.0)
    assignment = dict(conn.execute("SELECT block_id, split FROM blocks"))
    # By hand (seed 2): 15:00 block test, 15:05 block dev. Event 8 (15:04:50-15:05:00) and 10 (15:04:59-15:05:00)
    # end + 2 s pad past the dev block's start; event 9 (15:05:00-15:05:10) starts - 2 s before the test block's end.
    assert assignment["2018-03-11_150000"] == "test" and assignment["2018-03-11_150500"] == "dev"
    assert flagged == {8: "overlaps 2018-03-11_150500", 9: "overlaps 2018-03-11_150000",
                       10: "overlaps 2018-03-11_150500"}
    assert len(assignment) == 9
    first_csv = clips_csv.read_bytes()

    with pytest.raises(SystemExit, match="already exists"):
        run_script(monkeypatch, tmp_path)
    worksheet = tmp_path / "eval" / "queries" / "worksheet_dev.csv"
    worksheet.parent.mkdir(parents=True)
    worksheet.write_text("x")
    with pytest.raises(SystemExit, match="Not rebuilding"):
        run_script(monkeypatch, tmp_path, "--rebuild")
    worksheet.unlink()
    run_script(monkeypatch, tmp_path, "--rebuild")                     # same seed and data -> same split
    assert clips_csv.read_bytes() == first_csv


def test_restore_from_committed_reproduces_the_split(tmp_path):
    # pre-Phase-10 review, repro-2: a fresh clone fills the split tables from eval/splits/*.csv (make_splits
    # --from-committed); on a copy of the real DB with those tables emptied it must give back the same rows
    import shutil
    import subprocess
    import sys
    db = REPO_ROOT / "data" / "rewind.sqlite"
    if not db.exists():
        pytest.skip("data/rewind.sqlite not present (fresh clone)")
    copy = tmp_path / "db.sqlite"
    shutil.copy(db, copy)
    with sqlite3.connect(copy) as c:
        for t in ("event_splits", "video_blocks", "blocks"):
            c.execute(f"DELETE FROM {t}")
    run = lambda: subprocess.run([sys.executable, "-m", "scripts.make_splits", "--from-committed", "--db", str(copy)],
                                 cwd=REPO_ROOT, capture_output=True, text=True)
    r = run()
    assert r.returncode == 0, r.stderr
    a, b = sqlite3.connect(f"file:{db}?mode=ro", uri=True), sqlite3.connect(copy)
    for t, k in (("blocks", "block_id"), ("video_blocks", "video_id"), ("event_splits", "event_id")):
        assert a.execute(f"SELECT * FROM {t} ORDER BY {k}").fetchall() == b.execute(f"SELECT * FROM {t} ORDER BY {k}").fetchall(), t
    r = run()                                               # a second run refuses: the tables hold a split
    assert r.returncode != 0 and "already holds a split" in r.stderr + r.stdout
