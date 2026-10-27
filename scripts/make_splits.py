"""Step 2.1: cut the footage into ~30-min time blocks and assign each block (all its cameras) to tuning/dev/test.

Method and rules: rewind/eval/splits.py. Settings: config section `splits`.
Writes, in one go:
  - data/rewind.sqlite tables `blocks`, `video_blocks`, `event_splits` (schema in rewind/eval/splits.py);
  - eval/splits/blocks.csv and eval/splits/clips.csv: the frozen split, committed to git (data/ is not);
  - results/reports/splits_2.1.md.
Refuses to run again unless --rebuild is given, and refuses --rebuild once queries are frozen (Gate 2.2): the
queries are written against these splits. Reads only label metadata, never video or search results. A short run.

--from-committed (fresh clone, pre-Phase-10 review repro-2): fill the three DB tables from the committed
eval/splits/*.csv instead of choosing a split. The blocks are recomputed with the same code and must equal the
committed ones (block ids, clips, hours); the assignment is read from blocks.csv; boundary flags are recomputed with
the same rule. Writes no CSV and no report; refuses if the tables already hold a split.

Run from the repo root:
  uv run python -m scripts.make_splits                    # step 2.1 (once)
  uv run python -m scripts.make_splits --from-committed   # fresh clone: restore the frozen split
"""

import argparse
import csv
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from rewind.eval.splits import (SCHEMA, SPLITS, EventSpan, block_stats, boundary_events, choose_assignment,
                                cut_blocks, find_sessions, meets_requirements, split_stats)
from rewind.ingest.annotations import VEHICLE_START_STOP
from rewind.experiment_log import config_hash, git_hash
from rewind.ingest.manifest import connect, load_videos
from scripts.select_footage import markdown_table

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "eval" / "splits"
QUERIES_DIR = REPO_ROOT / "eval" / "queries"  # worksheets (2.2 part 1) and HASHES (Gate 2.2) are written here


def report(blocks, assignment, stats, n_valid: int, n_checked: int, failures: Counter, cfg: dict, conn,
           provenance: str) -> str:
    by_split = split_stats(assignment, stats)
    cameras = frozenset().union(*(st.rgb_cameras for st in stats.values()))
    total = sum(st.hours for st in by_split.values())
    ev = pd.read_sql("SELECT e.event_id, e.group_id, e.activity_type, e.label_source, v.modality, s.split, s.boundary "
                     "FROM events e JOIN videos v USING (video_id) JOIN event_splits s USING (event_id)", conn)
    vids = pd.read_sql("SELECT v.video_id, v.modality, v.duration_s, b.split FROM videos v "
                       "JOIN video_blocks USING (video_id) JOIN blocks b USING (block_id)", conn)
    ev["activity"] = ev.event_id == ev.group_id
    rows = []
    for split in SPLITS:
        st, e, vs = by_split[split], ev[ev.split == split], vids[vids.split == split]
        act = e[e.activity]
        rows.append({
            "split": split, "blocks": sum(s == split for s in assignment.values()),
            "hours": round(st.hours, 2), "share": f"{st.hours / total:.1%}",
            "RGB h": round(vs[vs.modality == "rgb"].duration_s.sum() / 3600, 2),
            "IR h": round(vs[vs.modality == "ir"].duration_s.sum() / 3600, 2),
            "eval-level h": round(st.eval_hours, 2), "train-level h": round(st.kitware_hours - st.eval_hours, 2),
            "activities": len(act), "eval-level": int((act.label_source == "kitware-eval").sum()),
            "train-level": int((act.label_source == "kitware-train").sum()),
            "types": len(st.types), "RGB vehicle_starts": st.rgb_vehicle["vehicle_starts"],
            "RGB vehicle_stops": st.rgb_vehicle["vehicle_stops"], "IR activities": st.ir_activities,
            "IR vehicle start/stop": st.ir_vehicle,
            **{f"{c} activities": st.camera_activities[c] for c in sorted(cameras)},
            "boundary-flagged pieces": int(e.boundary.notna().sum()),
        })
    per_split = pd.DataFrame(rows).set_index("split")
    per_block = pd.DataFrame([{
        "block": b.block_id, "session": b.session, "start": b.start_time.strftime("%H:%M:%S"),
        "end": b.end_time.strftime("%H:%M:%S"), "min": round((b.end_time - b.start_time).total_seconds() / 60, 1),
        "clips": len(b.video_ids), "hours": round(stats[b.block_id].hours, 2),
        "eval-level share": f"{stats[b.block_id].eval_hours / stats[b.block_id].kitware_hours:.0%}",
        "activities": sum(stats[b.block_id].types.values()), "types": len(stats[b.block_id].types),
        "RGB veh. starts/stops": "{}/{}".format(*(stats[b.block_id].rgb_vehicle[k] for k in VEHICLE_START_STOP)),
        "IR activities": stats[b.block_id].ir_activities, "split": assignment[b.block_id]} for b in blocks]
    ).set_index("block")
    acts = ev[ev.activity]
    per_type = acts.pivot_table(index="activity_type", columns="split", values="event_id", aggfunc="count",
                                fill_value=0).reindex(columns=list(SPLITS), fill_value=0)  # a split may have none
    per_type["total"] = per_type.sum(axis=1)
    per_type = per_type.sort_values("total", ascending=False)
    flagged = ev[ev.boundary.notna()].groupby(["split", "boundary"]).size().rename("pieces")
    overall = sum(s.eval_hours for s in stats.values()) / sum(s.kitware_hours for s in stats.values())
    unmet = meets_requirements(by_split, cfg, overall, cameras)
    fail_txt = ", ".join(f"{k}: {v}" for k, v in failures.most_common()) or "none"
    requirements = (["tuning fraction", "dev fraction", "tuning evaluation-level share"]
                    + [f"{s} {r}" for s in SPLITS for r in ("activity types", *VEHICLE_START_STOP, "IR activities")]
                    + [f"{s} activities on an RGB camera" for s in cfg["camera_coverage_splits"]])
    never = [r for r in requirements if not failures[r]]
    lines = [
        "# Evaluation splits (step 2.1)", "",
        "Generated by `uv run python -m scripts.make_splits`. " + provenance, "",
        "Split by time block, never by event; all cameras of a block are in one split (CLAUDE.md 1.10). Method:",
        "`rewind/eval/splits.py`. Hours are camera-hours (sum of clip durations). An *activity* is one annotated",
        "event, counted once if it was cut at a clip boundary (pieces share a group_id). IR labels are third-party and",
        "incomplete: recall-type metrics only (Gate 0.3).", "",
        f"Seed {cfg['seed']}. {n_valid} of {n_checked} assignments (all three splits non-empty) meet the requirements;",
        "the seed picked one uniformly.",
        f"Requirement failures over all assignments: {fail_txt}.",
        f"Requirements that never failed (so did not shape the split, only checked it): {', '.join(never) or 'none'}.",
        f"Chosen assignment, unmet requirements: {unmet or 'none'}.",
        f"Gate 0.3 rule: tuning evaluation-level share {by_split['tuning'].eval_hours / by_split['tuning'].kitware_hours:.0%}"
        f" <= overall {overall:.0%}.", "",
        "## Per split", "", markdown_table(per_split.T.rename_axis("")), "",
        "## Per block", "", markdown_table(per_block), "",
        "## Boundary-flagged event pieces (may be visible in another split; skip at 2.2, don't-care at 2.3)", "",
        markdown_table(flagged.to_frame()) if len(flagged) else "None.", "",
        "## Activities per type and split", "", markdown_table(per_type), "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "splits_2.1.md")
    parser.add_argument("--rebuild", action="store_true", help="replace an existing split (before Gate 2.2 only)")
    parser.add_argument("--from-committed", action="store_true",
                        help="fill the DB tables from the committed eval/splits/*.csv (fresh clone)")
    args = parser.parse_args()
    if args.from_committed:
        if args.rebuild:
            raise SystemExit("--from-committed and --rebuild exclude each other")
        return restore(args)

    config = yaml.safe_load(args.config.read_text())
    cfg, max_gap_s = config["splits"], config["ingest"]["max_name_duration_diff_s"]
    committed = [SPLITS_DIR / "blocks.csv", SPLITS_DIR / "clips.csv"]
    ro = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)  # check before creating any table
    existing = (ro.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'blocks'").fetchone()[0]
                and ro.execute("SELECT COUNT(*) FROM blocks").fetchone()[0])  # empty tables from a failed run: ok
    ro.close()
    if (existing or any(p.exists() for p in committed)) and not args.rebuild:
        raise SystemExit("A split already exists (blocks table or eval/splits/*.csv); pass --rebuild to replace it")
    written = sorted(QUERIES_DIR.glob("worksheet_*.csv")) + sorted(QUERIES_DIR.glob("HASHES"))
    if args.rebuild and written:
        raise SystemExit(f"{', '.join(str(p.relative_to(REPO_ROOT)) for p in written)} exist: queries are written "
                         "against this split. Not rebuilding.")
    conn = connect(args.db)
    conn.executescript(SCHEMA)

    videos = load_videos(conn)
    labels = {r["video_id"]: r["label_source"] for r in conn.execute("SELECT video_id, label_source FROM clip_labels")}
    missing = [v.clip for v in videos if v.video_id not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} videos have no clip_labels row (run scripts.build_events): {missing[:3]}")
    events = conn.execute("SELECT event_id, video_id, group_id, activity_type, label_source, start_time, end_time "
                          "FROM events").fetchall()

    blocks = cut_blocks(find_sessions(videos, max_gap_s), cfg["block_target_min"] * 60)
    stats = block_stats(blocks, videos, labels, events)
    assignment, n_valid, n_checked, failures = choose_assignment(stats, cfg)
    block_of = {vid: b.block_id for b in blocks for vid in b.video_ids}
    spans = [EventSpan(e["event_id"], e["group_id"], block_of[e["video_id"]],
                       datetime.fromisoformat(e["start_time"]), datetime.fromisoformat(e["end_time"])) for e in events]
    boundary = boundary_events(spans, blocks, assignment, cfg["boundary_pad_s"])

    with conn:  # one transaction: all or nothing
        for table in ("event_splits", "video_blocks", "blocks"):
            conn.execute(f"DELETE FROM {table}")
        for b in blocks:
            conn.execute("INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (b.block_id, b.session, b.start_time.isoformat(), b.end_time.isoformat(),
                          assignment[b.block_id], len(b.video_ids), stats[b.block_id].hours))
            conn.executemany("INSERT INTO video_blocks VALUES (?, ?)", [(vid, b.block_id) for vid in b.video_ids])
        conn.executemany("INSERT INTO event_splits VALUES (?, ?, ?)",
                         [(s.event_id, assignment[s.block_id], boundary.get(s.event_id)) for s in spans])
        # Invariants, checked on what was written: every video in one block, every block in one split.
        n_unplaced = conn.execute("SELECT COUNT(*) FROM videos WHERE video_id NOT IN "
                                  "(SELECT video_id FROM video_blocks)").fetchone()[0]
        if n_unplaced:
            raise AssertionError(f"{n_unplaced} videos are in no block")
        provenance = (f"git {git_hash(REPO_ROOT)}, config sha256 {config_hash(args.config)[:12]}, "
                      f"{datetime.now().isoformat(timespec='seconds')}.")
        text = report(blocks, assignment, stats, n_valid, n_checked, failures, cfg, conn, provenance)
        # CSVs are written inside the transaction (to .tmp, renamed after the commit): an error rolls the DB back.
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = [p.with_suffix(".csv.tmp") for p in committed]
        write_csvs(tmp, blocks, assignment, stats, videos)
    for t, p in zip(tmp, committed):
        t.replace(p)
    args.report.write_text(text)
    print(text)
    print(f"\nwrote {', '.join(str(p.relative_to(REPO_ROOT)) for p in committed + [args.report])} and 3 DB tables")


def restore(args) -> None:
    """--from-committed: the committed split into the DB tables, after checking the recomputed blocks match it."""
    config = yaml.safe_load(args.config.read_text())
    cfg, max_gap_s = config["splits"], config["ingest"]["max_name_duration_diff_s"]
    with open(SPLITS_DIR / "blocks.csv", newline="") as f:
        cblocks = {r["block_id"]: r for r in csv.DictReader(f)}
    with open(SPLITS_DIR / "clips.csv", newline="") as f:
        cclips = {r["clip"]: r for r in csv.DictReader(f)}
    conn = connect(args.db)
    conn.executescript(SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]:
        raise SystemExit("The DB already holds a split (blocks table not empty); nothing restored")
    videos = load_videos(conn)
    labels = {r["video_id"]: r["label_source"] for r in conn.execute("SELECT video_id, label_source FROM clip_labels")}
    events = conn.execute("SELECT event_id, video_id, group_id, activity_type, label_source, start_time, end_time "
                          "FROM events").fetchall()
    blocks = cut_blocks(find_sessions(videos, max_gap_s), cfg["block_target_min"] * 60)
    stats = block_stats(blocks, videos, labels, events)
    clip_of = {v.video_id: v.clip for v in videos}
    problems = []
    if {b.block_id for b in blocks} != set(cblocks):
        problems.append(f"block ids differ: {sorted({b.block_id for b in blocks} ^ set(cblocks))[:5]}")
    if set(clip_of.values()) != set(cclips):
        problems.append(f"clips differ: {sorted(set(clip_of.values()) ^ set(cclips))[:5]}")
    for b in blocks:
        c = cblocks.get(b.block_id)
        if c is None:
            continue
        if (int(c["n_videos"]) != len(b.video_ids) or c["start_time"] != b.start_time.isoformat()
                or c["end_time"] != b.end_time.isoformat() or c["hours"] != f"{stats[b.block_id].hours:.4f}"):
            problems.append(f"block {b.block_id} differs from blocks.csv")
        for vid in b.video_ids:
            if cclips.get(clip_of[vid], {}).get("block_id") != b.block_id:
                problems.append(f"{clip_of[vid]} is not in block {b.block_id} in clips.csv")
    if problems:
        raise SystemExit("Recomputed blocks do not match the committed split; nothing restored:\n  " + "\n  ".join(problems[:10]))
    assignment = {bid: r["split"] for bid, r in cblocks.items()}
    if set(assignment.values()) - set(SPLITS):
        raise SystemExit(f"unknown split names in blocks.csv: {set(assignment.values()) - set(SPLITS)}")
    block_of = {vid: b.block_id for b in blocks for vid in b.video_ids}
    spans = [EventSpan(e["event_id"], e["group_id"], block_of[e["video_id"]],
                       datetime.fromisoformat(e["start_time"]), datetime.fromisoformat(e["end_time"])) for e in events]
    boundary = boundary_events(spans, blocks, assignment, cfg["boundary_pad_s"])
    with conn:
        for b in blocks:
            conn.execute("INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (b.block_id, b.session, b.start_time.isoformat(), b.end_time.isoformat(),
                          assignment[b.block_id], len(b.video_ids), stats[b.block_id].hours))
            conn.executemany("INSERT INTO video_blocks VALUES (?, ?)", [(vid, b.block_id) for vid in b.video_ids])
        conn.executemany("INSERT INTO event_splits VALUES (?, ?, ?)",
                         [(s.event_id, assignment[s.block_id], boundary.get(s.event_id)) for s in spans])
    n = Counter(assignment.values())
    print(f"restored the committed split: {len(blocks)} blocks ({dict(n)}), {len(block_of)} clips, {len(spans)} "
          f"events ({len(boundary)} boundary-flagged)")


def write_csvs(paths: list[Path], blocks, assignment: dict[str, str], stats, videos) -> None:
    """eval/splits/blocks.csv (one row per block) and clips.csv (one row per video), the committed split."""
    block_of = {vid: b.block_id for b in blocks for vid in b.video_ids}
    with open(paths[0], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block_id", "session", "start_time", "end_time", "split", "n_videos", "hours"])
        for b in blocks:
            w.writerow([b.block_id, b.session, b.start_time.isoformat(), b.end_time.isoformat(),
                        assignment[b.block_id], len(b.video_ids), f"{stats[b.block_id].hours:.4f}"])
    clip_of = {v.video_id: v for v in videos}
    with open(paths[1], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "camera_id", "modality", "block_id", "split"])
        for vid in sorted(block_of, key=lambda i: (clip_of[i].start_time, clip_of[i].camera_id)):
            v = clip_of[vid]
            w.writerow([v.clip, v.camera_id, v.modality, block_of[vid], assignment[block_of[vid]]])


if __name__ == "__main__":
    main()
