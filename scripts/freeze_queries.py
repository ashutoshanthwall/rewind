"""Step 2.2 part 2 (Gate 2.2): validate the worksheets, write the frozen query files and their SHA-256 hashes.

Writes eval/queries/queries_<split>.csv (rewind.eval.queries.FROZEN_COLUMNS; category rows have empty event
columns: their ground truth is every activity of the type in the split) and eval/queries/HASHES (sha256sum format)
covering the frozen files, the worksheets and replacements.csv. Runs once: refuses if HASHES exists. After this,
nothing in eval/queries/ may change (tests/test_frozen_queries.py fails if it does).

Checks, all raising: every row filled (query, kind, qtype, specificity, modality valid), unique query ids,
witness-wording rules (rewind.eval.queries.text_problems); specific rows: the event exists, is in the row's split,
is not boundary-flagged, and clip / src_ids / times / camera / type in the worksheet match the DB, vehicle_event
rows are vehicle_starts / vehicle_stops; category rows: the type has enough eligible RGB activities in the split.

Run from the repo root:
  uv run python -m scripts.freeze_queries
"""

import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path

import yaml

from rewind.eval.queries import EVENT_COLUMNS, FROZEN_COLUMNS, KINDS, QTYPES, sha256, text_problems
from rewind.eval.splits import SPLITS
from rewind.ingest.annotations import VEHICLE_START_STOP
from scripts.make_worksheets import candidates

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_DIR = REPO_ROOT / "eval" / "queries"


def validate(rows: list[dict], split: str, conn, cfg: dict) -> list[str]:
    problems = []
    rgb_counts = Counter(c.activity_type for c in candidates(conn, split, cfg["min_event_s"]) if c.modality == "rgb")
    for r in rows:
        where = r["query_id"]
        bad = lambda msg: problems.append(f"{where}: {msg}")
        if r["split"] != split or not where.startswith(split[:3] + "-"):
            bad(f"split {r['split']!r} in worksheet_{split}.csv")
        if r["kind"] not in KINDS or r["qtype"] not in QTYPES or r["specificity"] not in cfg["specificity"] \
                or r["modality"] not in ("rgb", "ir"):
            bad(f"invalid kind/qtype/specificity/modality {r['kind']}/{r['qtype']}/{r['specificity']}/{r['modality']}")
        problems += [f"{where}: {p}" for p in text_problems(r)]
        if (r["qtype"] == "vehicle_event") != (r["activity_type"] in VEHICLE_START_STOP):
            bad(f"qtype {r['qtype']} with activity {r['activity_type']}")
        if r["kind"] == "category":
            if rgb_counts[r["activity_type"]] < cfg["min_category_instances"]:
                bad(f"category {r['activity_type']} has {rgb_counts[r['activity_type']]} eligible RGB activities")
            continue
        ev = conn.execute("SELECT e.*, v.clip, s.split, s.boundary FROM events e JOIN videos v USING (video_id) "
                          "JOIN event_splits s USING (event_id) WHERE e.event_id = ?", (int(r["event_id"]),)).fetchone()
        if ev is None:
            bad(f"event {r['event_id']} not in the DB")
            continue
        if ev["split"] != split or ev["boundary"] is not None:
            bad(f"event in split {ev['split']}, boundary {ev['boundary']}")
        for col in ("clip", "src_ids", "camera_id", "activity_type", "label_source", "start_time", "end_time"):
            if str(ev[col]) != r[col]:
                bad(f"{col} {r[col]!r} != DB {ev[col]!r}")
        for col in ("start_s", "end_s"):
            if abs(ev[col] - float(r[col])) > 1e-3:
                bad(f"{col} {r[col]} != DB {ev[col]}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    args = parser.parse_args()
    if (QUERIES_DIR / "HASHES").exists():
        raise SystemExit("eval/queries/HASHES exists: the queries are already frozen. Never re-freeze.")
    cfg = yaml.safe_load(args.config.read_text())["queries"]
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    sheets = {s: list(csv.DictReader(open(QUERIES_DIR / f"worksheet_{s}.csv", newline=""))) for s in SPLITS}
    problems = [p for s in SPLITS for p in validate(sheets[s], s, conn, cfg)]
    ids = Counter(r["query_id"] for rows in sheets.values() for r in rows)
    problems += [f"{q}: duplicate query_id" for q, n in ids.items() if n > 1]
    if problems:
        raise SystemExit("Not freezing:\n  " + "\n  ".join(problems))

    files = []
    for split in SPLITS:
        path = QUERIES_DIR / f"queries_{split}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FROZEN_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in sheets[split]:
                w.writerow({**r, **({c: "" for c in EVENT_COLUMNS} if r["kind"] == "category" else {})})
        files.append(path)
    files += [QUERIES_DIR / f"worksheet_{s}.csv" for s in SPLITS] + [QUERIES_DIR / "replacements.csv"]
    (QUERIES_DIR / "HASHES").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))

    summary = Counter((r["split"], r["kind"], r["qtype"]) for rows in sheets.values() for r in rows)
    for key in sorted(summary):
        print(*key, summary[key])
    print(f"froze {sum(ids.values())} queries: {', '.join(p.name for p in files)} -> eval/queries/HASHES")


if __name__ == "__main__":
    main()
