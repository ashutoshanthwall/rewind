"""Step 2.2 part 1: query worksheets per split, with one keyframe sheet per row. No query text is written here.

Which events, kinds, qtypes and specificity levels: rewind/eval/queries.py (config section `queries`).
Writes eval/queries/worksheet_<split>.csv (column `query` empty) and eval/queries/keyframes/<query_id>.jpg:
the event's middle frame on top (label boxes drawn), its first and last frame and a zoom on the boxes below.
Reads the proxies (frame i = original frame i, checked at step 1.2), decoding sequentially (frame index rule, 1.1).
Reads the test split's event list (allow_test=True): queries for test are written now, before any search code
exists; nothing is searched. Refuses to overwrite worksheets; refuses entirely once eval/queries/HASHES exists.

Run from the repo root:
  uv run python -m scripts.make_worksheets
  uv run python -m scripts.make_worksheets --replace tes-s20 --reason "..."   one unusable row: new event (same
      modality and type, seeded), new keyframe sheet, query blanked; logged in eval/queries/replacements.csv
  uv run python -m scripts.make_worksheets --sheets-only [--out-dir DIR]   only (re)draw the keyframe sheets of the
      existing (frozen) worksheets, e.g. after a fresh clone; never touches a CSV, allowed after Gate 2.2

Boxes: each actor's box is the annotated box at the nearest frame that has one (actors need not be boxed on every
frame). For the 60 frozen rows, 10 actor boxes in 7 rows came from 7-591 frames away (secondary actors, or the
first/last panel); the sheets don't mark this. The queries were written from these exact sheets, so the drawing is
kept as it was: --sheets-only reproduces them byte for byte.
"""

import argparse
import csv
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import av
import cv2
import numpy as np
import yaml

from rewind.eval.queries import Candidate, Slot, plan_split, replacement
from rewind.eval.splits import SPLITS, split_event_ids

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_DIR = REPO_ROOT / "eval" / "queries"
COLUMNS = ["query_id", "split", "kind", "qtype", "specificity", "modality", "activity_type", "query",
           "event_id", "clip", "src_ids", "camera_id", "label_source", "start_time", "end_time", "start_s", "end_s",
           "n_actors", "actor_types", "proxy", "keyframe"]
BOX_COLORS = [(0, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255)]  # BGR, one per actor index (drawing only)


def candidates(conn: sqlite3.Connection, split: str, min_event_s: float) -> list[Candidate]:
    ids = set(split_event_ids(conn, split, allow_test=(split == "test")))  # boundary-flagged events left out
    rows = conn.execute(
        "SELECT e.event_id, e.activity_type, e.camera_id, v.modality FROM events e JOIN videos v USING (video_id) "
        "WHERE e.event_id = e.group_id AND e.at_clip_start = 0 AND e.at_clip_end = 0 "
        "AND e.n_actors_with_boxes > 0 AND e.end_s - e.start_s >= ?", (min_event_s,)).fetchall()
    return [Candidate(*r) for r in rows if r[0] in ids]


def boxes_near(conn, event_id: int, frame: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Per actor, its box at the frame nearest to `frame` (actors' boxes need not cover every frame)."""
    out = []
    for (actor,) in conn.execute("SELECT DISTINCT actor FROM event_boxes WHERE event_id = ? ORDER BY actor", (event_id,)):
        r = conn.execute("SELECT x1, y1, x2, y2 FROM event_boxes WHERE event_id = ? AND actor = ? "
                         "ORDER BY ABS(frame - ?) LIMIT 1", (event_id, actor, frame)).fetchone()
        out.append((actor, tuple(r)))
    return out


def decode_frames(proxy: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Frames by index, decoding sequentially from frame 0 (never frame.pts)."""
    got, last = {}, max(wanted)
    with av.open(str(proxy)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                got[i] = frame.to_ndarray(format="bgr24")
            if i >= last:
                break
    missing = wanted - set(got)
    if missing:
        raise ValueError(f"{proxy.name}: frames {sorted(missing)[:5]} not decoded (file shorter than expected)")
    return got


def draw(img: np.ndarray, boxes, scale: float) -> np.ndarray:
    img = img.copy()
    for actor, (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (round(x1 * scale), round(y1 * scale)), (round(x2 * scale), round(y2 * scale)),
                      BOX_COLORS[actor % len(BOX_COLORS)], 2)
    return img


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize to fit inside w x h keeping aspect, centred on black."""
    s = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def sheet(frames: dict[int, np.ndarray], ev, boxes: dict[int, list], scale: float, width: int) -> np.ndarray:
    s, m, e = ev["start_frame"], (ev["start_frame"] + ev["end_frame"]) // 2, ev["end_frame"]
    h, w = frames[m].shape[:2]
    top_h, pw = round(width * h / w), width // 3
    ph = round(pw * h / w)
    top = label(fit(draw(frames[m], boxes[m], scale), width, top_h),
                f"middle {m / ev['fps']:.1f}s  {ev['activity_type']}  {ev['camera_id']}")
    xs = [c for _, b in boxes[m] for c in (b[0], b[2])]
    ys = [c for _, b in boxes[m] for c in (b[1], b[3])]
    bw, bh = (max(xs) - min(xs)) * scale, (max(ys) - min(ys)) * scale
    pad = 0.3 * max(bw, bh, 40)
    x1, y1 = max(0, round(min(xs) * scale - pad)), max(0, round(min(ys) * scale - pad))
    x2, y2 = min(w, round(max(xs) * scale + pad)), min(h, round(max(ys) * scale + pad))
    inside = x2 > x1 and y2 > y1  # False only if every box lies outside the image (never in the 60 frozen rows)
    zoom = frames[m][y1:y2, x1:x2] if inside else frames[m]
    row = [label(fit(draw(frames[s], boxes[s], scale), pw, ph), f"start {s / ev['fps']:.1f}s"),
           label(fit(draw(frames[e], boxes[e], scale), pw, ph), f"end {e / ev['fps']:.1f}s"),
           label(fit(zoom, width - 2 * pw, ph), "zoom (middle, no boxes)" if inside else "boxes outside image")]
    return np.vstack([top, np.hstack(row)])


def event_row(conn, event_id: int):
    return conn.execute("SELECT e.*, v.clip, v.width, v.fps, p.path AS proxy, p.width AS pw FROM events e "
                        "JOIN videos v USING (video_id) JOIN proxies p USING (video_id) WHERE e.event_id = ?",
                        (event_id,)).fetchone()


def redraw_sheets(args, cfg: dict) -> None:
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for split in SPLITS for r in csv.DictReader(open(QUERIES_DIR / f"worksheet_{split}.csv", newline=""))]
    for i, r in enumerate(rows, 1):
        path = write_sheet(conn, r["query_id"], event_row(conn, int(r["event_id"])), cfg["strip_width"], args.out_dir)
        print(f"[{i}/{len(rows)}] {path}", flush=True)


def write_sheet(conn, query_id: str, ev, width: int, out_dir: Path = QUERIES_DIR / "keyframes") -> Path:
    idx = {ev["start_frame"], (ev["start_frame"] + ev["end_frame"]) // 2, ev["end_frame"]}
    frames = decode_frames(REPO_ROOT / ev["proxy"], idx)
    boxes = {f: boxes_near(conn, ev["event_id"], f) for f in idx}
    keyframe = out_dir / f"{query_id}.jpg"
    if not cv2.imwrite(str(keyframe), sheet(frames, ev, boxes, ev["pw"] / ev["width"], width),
                       [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f"could not write {keyframe}")
    return keyframe


def row_fields(ev, keyframe: Path) -> dict:
    return {"event_id": ev["event_id"], "clip": ev["clip"], "src_ids": ev["src_ids"], "camera_id": ev["camera_id"],
            "label_source": ev["label_source"], "start_time": ev["start_time"], "end_time": ev["end_time"],
            "start_s": f"{ev['start_s']:.3f}", "end_s": f"{ev['end_s']:.3f}", "n_actors": ev["n_actors"],
            "actor_types": ev["actor_types"] or "", "proxy": ev["proxy"],
            "keyframe": str(keyframe.relative_to(REPO_ROOT))}


def replace_row(args, cfg: dict) -> None:
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    sheets = {split: list(csv.DictReader(open(QUERIES_DIR / f"worksheet_{split}.csv", newline=""))) for split in SPLITS}
    log = QUERIES_DIR / "replacements.csv"
    replaced = set()
    if log.exists():
        replaced = {int(r["old_event_id"]) for r in csv.DictReader(open(log, newline=""))}
    used = {int(r["event_id"]) for rows in sheets.values() for r in rows} | replaced
    split, row = next(((s, r) for s, rows in sheets.items() for r in rows if r["query_id"] == args.replace), (None, None))
    if row is None:
        raise SystemExit(f"no row {args.replace}")
    slot = Slot(row["query_id"], split, row["kind"], row["qtype"], row["specificity"], row["modality"],
                int(row["event_id"]), row["activity_type"])
    new = replacement(slot, candidates(conn, split, cfg["min_event_s"]), used, cfg["seed"])
    ev = event_row(conn, new.event_id)
    row.update(row_fields(ev, write_sheet(conn, slot.query_id, ev, cfg["strip_width"])), query="")
    with open(QUERIES_DIR / f"worksheet_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(sheets[split])
    new_log = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["query_id", "old_event_id", "new_event_id", "reason"])
        w.writerow([slot.query_id, slot.event_id, new.event_id, args.reason])
    print(f"{slot.query_id}: event {slot.event_id} -> {new.event_id} ({ev['activity_type']}, {ev['clip']}, "
          f"{ev['start_s']:.1f}-{ev['end_s']:.1f} s); keyframe rewritten, query blanked")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--replace", help="query_id of a row whose event is unusable")
    parser.add_argument("--reason", help="why the event is unusable (required with --replace)")
    parser.add_argument("--sheets-only", action="store_true", help="only redraw keyframe sheets of existing worksheets")
    parser.add_argument("--out-dir", type=Path, default=QUERIES_DIR / "keyframes", help="for --sheets-only")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["queries"]
    if args.sheets_only:
        redraw_sheets(args, cfg)
        return
    if (QUERIES_DIR / "HASHES").exists():
        raise SystemExit("eval/queries/HASHES exists: queries are frozen (Gate 2.2). Not touching the worksheets.")
    if args.replace:
        if not args.reason:
            raise SystemExit("--replace needs --reason")
        replace_row(args, cfg)
        return
    existing = sorted(QUERIES_DIR.glob("worksheet_*.csv"))
    if existing:
        raise SystemExit(f"{[p.name for p in existing]} exist; delete them by hand to regenerate (queries in them "
                         "would be lost)")
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rng = random.Random(cfg["seed"])
    slots = []
    for split in SPLITS:
        cands = candidates(conn, split, cfg["min_event_s"])
        counts = Counter(c.activity_type for c in cands if c.modality == "rgb")
        slots += plan_split(split, cands, counts, cfg, rng)

    events = {s.event_id: conn.execute(
        "SELECT e.*, v.clip, v.width, v.fps, p.path AS proxy, p.width AS pw FROM events e "
        "JOIN videos v USING (video_id) JOIN proxies p USING (video_id) WHERE e.event_id = ?",
        (s.event_id,)).fetchone() for s in slots}
    by_proxy = defaultdict(set)
    for ev in events.values():
        by_proxy[ev["proxy"]] |= {ev["start_frame"], (ev["start_frame"] + ev["end_frame"]) // 2, ev["end_frame"]}
    (QUERIES_DIR / "keyframes").mkdir(parents=True, exist_ok=True)
    frames = {}
    for i, (proxy, wanted) in enumerate(sorted(by_proxy.items()), 1):
        print(f"[{i}/{len(by_proxy)}] decoding {proxy} ({len(wanted)} frames)", flush=True)
        frames[proxy] = decode_frames(REPO_ROOT / proxy, wanted)

    rows = defaultdict(list)
    for s in slots:
        ev = events[s.event_id]
        idx = {ev["start_frame"], (ev["start_frame"] + ev["end_frame"]) // 2, ev["end_frame"]}
        boxes = {f: boxes_near(conn, s.event_id, f) for f in idx}
        img = sheet(frames[ev["proxy"]], ev, boxes, ev["pw"] / ev["width"], cfg["strip_width"])
        keyframe = QUERIES_DIR / "keyframes" / f"{s.query_id}.jpg"
        if not cv2.imwrite(str(keyframe), img, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise OSError(f"could not write {keyframe}")
        rows[s.split].append({
            "query_id": s.query_id, "split": s.split, "kind": s.kind, "qtype": s.qtype,
            "specificity": s.specificity, "modality": s.modality, "activity_type": s.activity_type, "query": "",
            "event_id": s.event_id, "clip": ev["clip"], "src_ids": ev["src_ids"], "camera_id": ev["camera_id"],
            "label_source": ev["label_source"], "start_time": ev["start_time"], "end_time": ev["end_time"],
            "start_s": f"{ev['start_s']:.3f}", "end_s": f"{ev['end_s']:.3f}", "n_actors": ev["n_actors"],
            "actor_types": ev["actor_types"] or "", "proxy": ev["proxy"],
            "keyframe": str(keyframe.relative_to(REPO_ROOT))})
    for split in SPLITS:
        with open(QUERIES_DIR / f"worksheet_{split}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows[split])
    summary = Counter((s.split, s.kind, s.qtype, s.modality) for s in slots)
    for key in sorted(summary):
        print(*key, summary[key])
    print("specificity:", dict(Counter((s.split, s.specificity) for s in slots)))
    print(f"wrote {len(slots)} rows to eval/queries/worksheet_*.csv and {len(slots)} keyframe sheets")


if __name__ == "__main__":
    main()
