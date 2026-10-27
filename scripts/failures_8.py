"""Step 8.3: failure analysis of the single test run. Reads only the step 8.2 outputs, never the query files or the
split tables: results/reports/test_rows.csv, test_clips.json (each query's frozen fields + Rewind's top clips) and
test_event_ranks.csv; plus the query keyframe sheets (eval/queries/keyframes, drawn at step 2.2) and the playback
proxies (footage, no labels).

Failures = the --n specific queries with the most seconds watched by Rewind (misses first). For each:
  - the truth: camera, wall-clock span, clip and in-file seconds, keyframe sheet (eval/queries/keyframes/<id>.jpg);
  - what Rewind returned instead: its top 3 clips (camera, span, branches, best hit and box), and a thumbnail strip
    of their best-hit frames from the proxies with the hit box drawn (results/figures/failures/<id>_rewind.jpg);
  - the best rank of a vector inside the true event among all test vectors (exact, from 8.2), per branch.
Writes results/reports/failures.md with an empty `cause` column (filled by hand from the evidence) and
results/reports/failures_8.3.csv.

Run from the repo root:  uv run python -m scripts.failures_8 [--in-dir DIR --out-dir DIR]   (DIR: a rehearsal)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT, display_path
from rewind.video import frames_at
from scripts.select_footage import rows_table

CAUSES = ("object too small", "IR", "detector miss", "track broken", "action too brief", "ambiguous query",
          "dropped by dedup", "correct but unlabelled", "other")
THUMB_W = 400          # px per thumbnail in the strip
TOP = 3                # returned clips shown per failure


def best_hit(c: dict) -> tuple[float, dict | None]:
    """Time (s in file) and appearance hit of a clip's best appearance hit; else the first hit time, no box."""
    h = c["best"].get("appearance")
    return (h["t_s"], h) if h else (c["first_hit_s"], None)


def thumbs(clips: list[dict], videos: dict, proxies: dict) -> np.ndarray:
    tiles = []
    for c in clips:
        v = videos[c["video_id"]]
        pr = proxies[c["video_id"]]
        t, h = best_hit(c)
        f = min(int(round(t * pr["fps"])), pr["n_frames"] - 1)
        _, img = next(frames_at(REPO_ROOT / pr["path"], pr["n_frames"], {f}))
        if h is not None and h.get("x1") is not None:
            s = img.shape[1] / v.width
            cv2.rectangle(img, (int(h["x1"] * s), int(h["y1"] * s)), (int(h["x2"] * s), int(h["y2"] * s)),
                          (0, 200, 255), 3)
        img = cv2.resize(img, (THUMB_W, int(img.shape[0] * THUMB_W / img.shape[1])))
        cv2.putText(img, f"#{c['rank']} {c['camera_id']} {c['start'][11:19]}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    h = max(t.shape[0] for t in tiles)
    return np.hstack([np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--in-dir", type=Path, default=REPO_ROOT / "results" / "reports")
    p.add_argument("--out-dir", type=Path, help="default: results/reports (+ results/figures/failures)")
    p.add_argument("--n", type=int, default=10, help="failures listed (CLAUDE.md step 8.3: ~10)")
    args = p.parse_args()
    out = args.out_dir or REPO_ROOT / "results" / "reports"
    figs = (args.out_dir / "failures") if args.out_dir else REPO_ROOT / "results" / "figures" / "failures"
    figs.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.in_dir / "test_rows.csv")
    saved = json.loads((args.in_dir / "test_clips.json").read_text())
    ranks = pd.read_csv(args.in_dir / "test_event_ranks.csv")
    conn = connect(args.db)
    videos = {v.video_id: v for v in load_videos(conn)}
    proxies = {r[0]: {"path": r[1], "n_frames": r[2], "fps": r[3]}
               for r in conn.execute("SELECT video_id, path, n_frames, fps FROM proxies")}

    rw = rows[(rows.system == "rewind") & (rows.kind == "specific")].sort_values("seconds", ascending=False)
    worst = rw.head(args.n)
    table, csv_rows = [], []
    for r in worst.itertuples():
        q = saved[r.query_id]["query"]
        clips = saved[r.query_id]["clips"][:TOP]
        strip = figs / f"{r.query_id}_rewind.jpg"
        if clips:
            cv2.imwrite(str(strip), thumbs(clips, videos, proxies))
        rk = ranks[ranks.query_id == r.query_id]
        best = ", ".join(f"{x.branch} {x.best_rank} ({x.vectors_in_event} in event)" for x in rk.itertuples())
        span = f"{q['camera_id']} {q['start_time'][11:19]}–{q['end_time'][11:19]} ({q['clip']} {float(q['start_s']):.0f}–{float(q['end_s']):.0f} s)"
        ret = "<br>".join(f"#{c['rank']} {c['camera_id']} {c['start'][11:19]}–{c['end'][11:19]} [{'+'.join(c['branches'])}]"
                          + (f" {c['best']['appearance'].get('type')}"
                             f"{' ' + str(c['best']['appearance'].get('class')) if c['best']['appearance'].get('class') else ''}"
                             if 'appearance' in c['best'] else "") for c in clips)
        key = REPO_ROOT / "eval" / "queries" / "keyframes" / f"{r.query_id}.jpg"
        table.append([r.query_id, r.qtype, r.specificity, f"{r.modality} / {r.label_source}", q["activity_type"],
                      f"\"{q['query']}\"", span, f"{r.seconds:,.0f} s / " + ("miss" if r.hit_rank == -1 else f"rank {r.hit_rank}"),
                      best, ret, f"[truth]({display_path(key)}) · [returned]({display_path(strip)})", ""])
        csv_rows.append({"query_id": r.query_id, "qtype": r.qtype, "modality": r.modality, "seconds": r.seconds,
                         "hit_rank": r.hit_rank, "event_vector_ranks": best, "cause": ""})
    lines = [
        "# Step 8.3 — failure analysis (test)", "",
        f"Generated by `scripts/failures_8.py` from the step 8.2 outputs in `{display_path(args.in_dir)}` on "
        f"{datetime.now():%Y-%m-%d %H:%M}. The {len(worst)} specific test queries with the most seconds watched by "
        f"Rewind (misses first). Event-vector rank: best rank of a vector inside the true event among all test "
        f"vectors of the branch (exact search, before grouping; -1 = no vector inside the event). Thumbnails: the "
        f"query's keyframe sheet (truth) and the best-hit frame of Rewind's top {TOP} clips with the matched box.", "",
        f"Cause labels: {' / '.join(CAUSES)}.", "",
        rows_table(["query", "qtype", "specificity", "modality / labels", "activity", "query text", "truth",
                    "Rewind", "event-vector rank", f"Rewind top {TOP}", "thumbnails", "cause"], table), ""]
    (out / "failures.md").write_text("\n".join(lines))
    pd.DataFrame(csv_rows).to_csv(out / "failures_8.3.csv", index=False)
    print("\n".join(lines))
    print("report:", display_path(out / "failures.md"), "figures:", display_path(figs))


if __name__ == "__main__":
    main()
