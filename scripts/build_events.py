"""Step 1.3: build the annotated events tables in data/rewind.sqlite from the MEVA annotations.

Tables (schema in rewind/ingest/manifest.py): `clip_labels` (one row per video: events / empty / none, and which
annotation it comes from), `events` (one row per annotated activity within one clip, frames and wall-clock times),
`event_boxes` (per-frame actor boxes of each event). Conventions and checks: rewind/ingest/annotations.py.
  - RGB clips: Kitware labels only (evaluation-level `kitware`, training-level `kitware-meva-training`).
  - IR clips: third-party contrib labels only (annotations.contrib_sources), merged by annotations.contrib_merge_min_box_iou.
  - Pieces of one activity cut at clip boundaries share a group_id (annotations.boundary_merge_min_box_iou).
All tables are written in one transaction. If they already have rows the script stops, unless --rebuild is given,
which deletes the old rows of these three tables first. A short run (a few minutes): parsing ~1 GB of geom files.

Run from the repo root:
  uv run python -m scripts.build_events                 build, print counts, write results/reports/events_1.3.md
  uv run python -m scripts.build_events --spot-check    print 5 random events to check by eye on the proxies
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

from rewind.ingest.annotations import (VEHICLE_START_STOP, Event, activity_names, boundary_links, contrib_files, group_ids, kitware_events,
                                       load_json, load_kpf, merge_contrib, parse_contrib_json,
                                       parse_kitware_activities, parse_kitware_types, piece, read_geom)
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import display_path
from scripts.build_manifest import ANNOT_ROOT, KITWARE_SOURCES, MEVA_REPO
from scripts.select_footage import markdown_table

REPO_ROOT = Path(__file__).resolve().parent.parent
LABEL_SOURCE = {"kitware": "kitware-eval", "kitware-meva-training": "kitware-train"}
FROZEN_QUERIES = REPO_ROOT / "eval" / "queries" / "HASHES"  # written at Gate 2.2


def kitware_index(annot_root: Path) -> dict[str, tuple[str, Path]]:
    """{clip: (folder, path of <clip>.activities.yml)} over both Kitware folders."""
    index = {}
    for folder in KITWARE_SOURCES:
        for path in (annot_root / folder).rglob("*.activities.yml"):
            clip = path.name.removesuffix(".activities.yml")
            if clip in index:
                raise ValueError(f"{clip} is annotated in both {index[clip][0]} and {folder}")
            index[clip] = (folder, path)
    return index


def clip_events(video, kitware: dict, contrib: dict, known: set[str], rename: dict[str, str],
                cfg: dict, stats: Counter) -> tuple[list[Event], str, str]:
    """Events of one video and its (label_status, label_source)."""
    if video.modality == "rgb":
        if video.clip not in kitware:
            raise ValueError(f"RGB clip {video.clip} has no Kitware annotation (1.1 checked that every RGB clip has one)")
        folder, act_path = kitware[video.clip]
        stem = act_path.with_name(video.clip)
        acts, empty = parse_kitware_activities(load_kpf(act_path), known, act_path.name)
        if empty:
            stats["kitware_empty_clips"] += 1
            return [], "empty", LABEL_SOURCE[folder]
        types = parse_kitware_types(load_kpf(stem.with_name(f"{video.clip}.types.yml")), video.clip)
        tracks = read_geom(stem.with_name(f"{video.clip}.geom.yml"))
        events = kitware_events(video.clip, LABEL_SOURCE[folder], folder, acts, types, tracks, video.n_frames)
        stats["kitware_actor_span_outside_activity"] += sum(
            not (act["start"] <= s and e <= act["end"]) for act in acts for _, s, e in act["actors"])
        return events, "events", LABEL_SOURCE[folder]
    # IR: contrib labels only. A Kitware file on an IR clip is only ever an empty_37 marker (docs/meva_notes.md).
    kitware_marker = video.clip in kitware
    if kitware_marker:
        acts, empty = parse_kitware_activities(load_kpf(kitware[video.clip][1]), known, video.clip)
        if acts:
            raise ValueError(f"IR clip {video.clip} has Kitware activities; decide how to use them before building")
    records = []
    for source, path in contrib.get(video.clip, []):
        events, dropped = parse_contrib_json(load_json(path), video.clip, source, rename, known, video.n_frames)
        stats["contrib_records"] += len(events)
        stats["contrib_zero_length_dropped"] += dropped
        records += events
    events = merge_contrib(records, cfg["contrib_merge_min_box_iou"]) if records else []
    stats["contrib_records_merged_away"] += len(records) - len(events)
    if events:
        return events, "events", "contrib"
    # Contrib labels are incomplete and Kitware has no real IR labels, so no event is 'none', never 'empty'.
    # Why, for the report: no annotation at all / a Kitware empty_37 marker (meaning VERIFY) / contrib file, 0 records.
    reason = ("kitware_empty37_marker" if kitware_marker
              else "contrib_file_without_records" if video.clip in contrib else "no_annotation_file")
    stats[f"ir_none_{reason}"] += 1
    return [], "none", "none"


def build(conn, videos, kitware, contrib, known, rename, cfg, max_gap_s: float) -> Counter:
    stats = Counter()
    pieces, event_id = {}, 0
    for video in sorted(videos, key=lambda v: v.clip):
        events, status, source = clip_events(video, kitware, contrib, known, rename, cfg, stats)
        conn.execute("INSERT INTO clip_labels VALUES (?, ?, ?, ?)", (video.video_id, status, source, len(events)))
        pieces[video.clip] = []
        for ev in sorted(events, key=lambda e: (e.start_frame, e.end_frame, e.activity_type)):
            event_id += 1
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, video.video_id, video.camera_id, ev.activity_type, ev.start_frame, ev.end_frame,
                 ev.start_frame / video.fps, (ev.end_frame + 1) / video.fps,
                 video.frame_time(ev.start_frame).isoformat(timespec="microseconds"),
                 video.frame_end_time(ev.end_frame).isoformat(timespec="microseconds"),
                 ev.label_source, json.dumps(ev.sources), json.dumps(ev.src_ids), ev.src_status,
                 None if ev.actor_types is None else json.dumps(ev.actor_types),
                 len(ev.boxes), sum(bool(t) for t in ev.boxes),
                 int(ev.start_frame == 0), int(ev.end_frame == video.n_frames - 1), event_id))
            conn.executemany("INSERT INTO event_boxes VALUES (?, ?, ?, ?, ?, ?, ?)",
                             ((event_id, actor, frame, *box) for actor, track in enumerate(ev.boxes)
                              for frame, box in track.items()))
            pieces[video.clip].append(piece(event_id, ev))
        stats["events"] += len(events)

    # Link pieces of activities cut at clip boundaries: consecutive clips of one camera, back to back.
    links = []
    by_camera = defaultdict(list)
    for v in videos:
        by_camera[v.camera_id].append(v)
    for cam_videos in by_camera.values():
        cam_videos.sort(key=lambda v: v.start_time)
        for a, b in zip(cam_videos, cam_videos[1:]):
            if abs((b.start_time - a.end_time).total_seconds()) <= max_gap_s:
                stats["back_to_back_clip_pairs"] += 1
                links += boundary_links(pieces[a.clip], a.n_frames, pieces[b.clip], cfg["boundary_merge_min_box_iou"])
    ids = [p.event_id for ps in pieces.values() for p in ps]
    groups = group_ids(ids, links)
    conn.executemany("UPDATE events SET group_id = ? WHERE event_id = ?",
                     ((g, e) for e, g in groups.items() if g != e))
    stats["boundary_links"] = len(links)
    stats["groups"] = len(set(groups.values()))
    return stats


def report(conn, stats: Counter, seconds: float) -> str:
    ev = pd.read_sql("SELECT e.*, v.modality FROM events e JOIN videos v USING (video_id)", conn)
    labels = pd.read_sql("SELECT l.*, v.camera_id FROM clip_labels l JOIN videos v USING (video_id)", conn)
    per_type = ev.pivot_table(index="activity_type", columns="camera_id", values="event_id", aggfunc="count",
                              fill_value=0, margins=True, margins_name="total")
    per_type = per_type.sort_values("total", ascending=False)
    groups = ev.groupby("activity_type")["group_id"].nunique()
    groups["total"] = ev["group_id"].nunique()
    groups = groups.rename("groups (cut pieces linked)")
    status = labels.pivot_table(index="camera_id", columns=["label_status", "label_source"], values="video_id",
                                aggfunc="count", fill_value=0)
    status.columns = [f"{s} ({src})" for s, src in status.columns]
    flags = ev.groupby("camera_id")[["at_clip_start", "at_clip_end"]].sum()
    src_status = ev[ev.label_source != "contrib"]["src_status"].value_counts()
    vehicle = ev[ev.activity_type.isin(VEHICLE_START_STOP)].pivot_table(
        index="camera_id", columns="activity_type", values="event_id", aggfunc="count", fill_value=0)
    lines = [
        "# Annotated events (step 1.3)", "",
        "Generated by `uv run python -m scripts.build_events`. One event = one annotated activity record within one",
        "5-min clip of one camera. The same real activity can appear on several cameras and, when cut at a clip",
        "boundary, as 2+ pieces (linked by group_id). RGB: Kitware labels. IR (G474): third-party labels, incomplete",
        "(recall-type metrics only, Gate 0.3).", "",
        f"Build time {seconds:.0f} s. Build statistics: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())), "",
        "## Clips by label status", "", markdown_table(status), "",
        "## Events per activity type and camera", "", markdown_table(per_type.join(groups)), "",
        "## Vehicle start/stop events per camera", "", markdown_table(vehicle), "",
        "## Events cut at clip boundaries (start on frame 0 / end on the last frame)", "", markdown_table(flags), "",
        "## Kitware src_status (meaning undocumented, VERIFY; not filtered)", "", markdown_table(src_status), "",
    ]
    return "\n".join(lines)


def fmt_s(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:05.2f}"


def spot_check(conn, seed: int) -> str:
    """5 random events to check by eye: a vehicle start/stop and another activity from each Kitware level (the
    levels come from different annotation pipelines, incl. both KPF line styles) and one IR (contrib) event."""
    rng = random.Random(seed)
    rows = [dict(r) for r in conn.execute(
        "SELECT e.*, v.clip, v.modality, v.width, v.height, v.fps, p.path AS proxy, p.width AS pw "
        "FROM events e JOIN videos v USING (video_id) JOIN proxies p USING (video_id) ORDER BY e.event_id")]
    vehicle = lambda r: r["activity_type"] in VEHICLE_START_STOP
    pools = [(f"vehicle start/stop, {level}", [r for r in rows if vehicle(r) and r["label_source"] == level], 1)
             for level in ("kitware-eval", "kitware-train")]
    pools += [("IR, third-party label", [r for r in rows if r["label_source"] == "contrib"], 1)]
    pools += [(f"other activity, {level}", [r for r in rows if not vehicle(r) and r["label_source"] == level], 1)
              for level in ("kitware-eval", "kitware-train")]
    out = [f"Spot check, seed {seed}. Times are proxy playback times (= frame / fps). Boxes are in proxy pixels",
           "(x1,y1)-(x2,y2) from the top-left corner, at the event's middle frame; the ffplay command draws them (fixed",
           "position, red) and a playback clock (top left). ffplay starts at the keyframe before -ss (up to 8 s early):",
           "wait for the clock to reach the in-file time. Keys: space = pause, left/right = -/+10 s, q = quit.", ""]
    k = 0
    for label, pool, n in pools:
        for r in rng.sample(pool, n):
            k += 1
            mid = (r["start_frame"] + r["end_frame"]) // 2
            scale = r["pw"] / r["width"]
            boxes = [tuple(round(v * scale) for v in b) for b in conn.execute(
                "SELECT x1, y1, x2, y2 FROM event_boxes WHERE event_id = ? AND frame = ?", (r["event_id"], mid))]
            box_txt = "; ".join(f"({b[0]},{b[1]})-({b[2]},{b[3]})" for b in boxes) or "no boxes"
            vf = ",".join([f"drawbox=x={b[0]}:y={b[1]}:w={b[2] - b[0]}:h={b[3] - b[1]}:color=red:t=2" for b in boxes]
                          + ["drawtext=text='%{pts\\:hms}':x=8:y=8:fontsize=24:fontcolor=yellow:box=1:boxcolor=black"])
            seek = max(0.0, r["start_s"] - 3)
            out += [f"{k}. [{label}] {r['activity_type']}  camera {r['camera_id']}  event_id {r['event_id']}  "
                    f"({r['label_source']}, {', '.join(json.loads(r['sources']))})",
                    f"   proxy {r['proxy']}",
                    f"   in file {fmt_s(r['start_s'])} -> {fmt_s(r['end_s'])} ({r['end_s'] - r['start_s']:.1f} s; "
                    f"frames {r['start_frame']}-{r['end_frame']})   wall-clock {r['start_time'][:19]}",
                    f"   boxes at {fmt_s(mid / r['fps'])}: {box_txt}"
                    + ("   [starts on the clip's first frame: may continue from the previous clip]" if r["at_clip_start"] else "")
                    + ("   [ends on the clip's last frame: may continue in the next clip]" if r["at_clip_end"] else "")]
            if r["label_source"] == "contrib":
                out.append("   [third-party label: start/end can differ from Kitware's by ~1-3 s, and CMU-v2 boxes can "
                           "run ahead of the object (VERIFY); judge the time span, not the box]")
            out += [f'   ffplay -ss {seek:.1f} -vf "{vf}" {r["proxy"]}', ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "events_1.3.md")
    parser.add_argument("--rebuild", action="store_true", help="delete existing rows of the three tables first")
    parser.add_argument("--spot-check", action="store_true", help="only print random events to check by eye")
    parser.add_argument("--seed", type=int, default=13, help="seed for --spot-check")
    args = parser.parse_args()

    conn = connect(args.db)
    if args.spot_check:
        print(spot_check(conn, seed=args.seed))
        return
    config = yaml.safe_load(args.config.read_text())
    cfg, max_gap_s = config["annotations"], config["ingest"]["max_name_duration_diff_s"]
    existing = conn.execute("SELECT (SELECT COUNT(*) FROM events) + (SELECT COUNT(*) FROM clip_labels) "
                            "+ (SELECT COUNT(*) FROM event_boxes)").fetchone()[0]
    if existing and not args.rebuild:
        raise SystemExit(f"events tables already have {existing} rows; pass --rebuild to replace them")
    if args.rebuild and FROZEN_QUERIES.exists():
        raise SystemExit(f"{FROZEN_QUERIES.relative_to(REPO_ROOT)} exists: frozen queries reference event ids, and a "
                         "rebuild can renumber them. Not rebuilding.")
    videos = load_videos(conn)
    known, rename = activity_names(MEVA_REPO)
    kitware = kitware_index(ANNOT_ROOT)
    ir_clips = {v.clip for v in videos if v.modality == "ir"}
    contrib = contrib_files(ANNOT_ROOT / "contrib", cfg["contrib_sources"], ir_clips)

    t0 = time.monotonic()
    with conn:  # one transaction: all or nothing
        if args.rebuild:
            for table in ("event_boxes", "events", "clip_labels"):
                conn.execute(f"DELETE FROM {table}")
        stats = build(conn, videos, kitware, contrib, known, rename, cfg, max_gap_s)
        text = report(conn, stats, time.monotonic() - t0)  # inside the transaction: a report error rolls back
    args.report.write_text(text)
    print(text)
    print(f"\nwrote {display_path(args.report)}")


if __name__ == "__main__":
    main()
