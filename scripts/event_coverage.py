"""Finding F1 diagnostic (step 6.1, tuning only): is the vehicle stop/start recall lost in tracking or in the rule?

For every Kitware vehicle_starts / vehicle_stops label in the tuning clips that carry Kitware labels:
  - track coverage = share of the label's frames on the detection grid (frame % step == 0) where some track box has
    IoU >= events.coverage_check.min_iou with a label actor box; "tracked" if that share >= min_frame_share;
  - rule fires nearby = the frozen rules (config `events`, rewind/track/tracks.clip_events) detect an event of the same
    type in the same clip whose span is within metrics.event_tolerance_s of the label (any track: a loose upper bound
    on "this label could be matched").
Breakdowns per camera x type and per label box-height quantile. Information only: no gate or setting depends on it.
Writes results/reports/events_6.1_coverage.md and one experiments row. Tuning only; dev and test are refused.

Run from the repo root:
  uv run python -m scripts.event_coverage
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.eval.splits import split_video_ids
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.motion import store as motion_store
from rewind.paths import REPO_ROOT, display_path
from rewind.track import store
from rewind.track.tracks import EVENT_TYPES, RuleParams, clip_events, iou_matrix
from rewind.video import keep_step
from scripts.track_pass import settings_hash

SPLIT = "tuning"


def frame_coverage(label: pd.DataFrame, tracks: pd.DataFrame, min_iou: float) -> tuple[int, int]:
    """(grid frames of the label, frames where a track box has IoU >= min_iou with a label box of that frame).
    label: the label's boxes on grid frames (frame, x1..y2; several actors may share a frame); tracks: the clip's
    track rows (frame, x1..y2)."""
    n = covered = 0
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    for f, g in label.groupby("frame"):
        n += 1
        t = by_frame.get(f)
        if t is not None and iou_matrix(g[["x1", "y1", "x2", "y2"]].to_numpy(float),
                                        t[["x1", "y1", "x2", "y2"]].to_numpy(float)).max() >= min_iou:
            covered += 1
    return n, covered


def fires_near(det: pd.DataFrame, typ: str, start_s: float, end_s: float, tol_s: float) -> bool:
    """A detected event of this type whose [start_s, end_s) overlaps the label span widened by tol_s."""
    d = det[det["event_type"] == typ]
    return bool(((d["start_s"] < end_s + tol_s) & (d["end_s"] > start_s - tol_s)).any())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    ap.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    args = ap.parse_args()
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    cc, tol = cfg["events"]["coverage_check"], cfg["metrics"]["event_tolerance_s"]
    p = RuleParams.from_config(cfg)
    art = REPO_ROOT / "artifacts"
    conn = connect(args.db)
    ids = set(split_video_ids(conn, SPLIT))
    kit = {r[0] for r in conn.execute("SELECT video_id FROM clip_labels WHERE label_source LIKE 'kitware%'")}
    vids = {v.video_id: v for v in load_videos(conn) if v.video_id in ids & kit}
    det_settings = {motion_store.load_marker(art, v)["settings"] for v in vids.values()}
    if len(det_settings) != 1:
        raise ValueError(f"detections with {len(det_settings)} settings")
    tset = settings_hash(cfg, det_settings.pop())

    labels = pd.read_sql_query(
        f"SELECT event_id, video_id, camera_id, activity_type, start_s, end_s FROM events WHERE label_source != "
        f"'contrib' AND activity_type IN ({','.join('?' * len(EVENT_TYPES))})", conn, params=EVENT_TYPES)
    labels = labels[labels["video_id"].isin(vids)]
    boxes = pd.read_sql_query("SELECT event_id, frame, x1, y1, x2, y2 FROM event_boxes", conn)
    boxes = boxes[boxes["event_id"].isin(set(labels["event_id"]))]
    rows = []
    for vid, labs in labels.groupby("video_id"):
        v = vids[vid]
        step = keep_step(v.fps, cfg["detection"]["fps"])
        tracks = store.load_tracks(art, v, tset)
        det = clip_events(tracks, p)
        for e in labs.itertuples(index=False):
            b = boxes[(boxes["event_id"] == e.event_id) & (boxes["frame"] % step == 0)]
            n, cov = frame_coverage(b, tracks, cc["min_iou"])
            if n == 0:
                raise ValueError(f"label {e.event_id} has no box on the detection grid")
            rows.append((e.camera_id, e.activity_type, n, cov / n, fires_near(det, e.activity_type, e.start_s,
                                                                             e.end_s, tol),
                         float((b["y2"] - b["y1"]).median())))
    df = pd.DataFrame(rows, columns=["camera", "type", "grid_frames", "cover", "fires", "box_h"])
    df["tracked"] = df["cover"] >= cc["min_frame_share"]
    pct = lambda x: f"{100 * x:.0f} %"
    summary = {"labels": len(df), "tracked": df["tracked"].mean(), "fires_given_tracked": df[df["tracked"]]["fires"].mean(),
               "fires_given_untracked": df[~df["tracked"]]["fires"].mean()}
    by_cam = df.groupby(["camera", "type"]).agg(n=("tracked", "size"), tracked=("tracked", "mean"),
                                                fires=("fires", "mean"), median_box_h=("box_h", "median"))
    df["size_bin"] = pd.qcut(df["box_h"], cc["n_size_bins"])
    by_size = df.groupby("size_bin", observed=True).agg(n=("tracked", "size"), tracked=("tracked", "mean"),
                                                        fires=("fires", "mean"))
    lines = ["# Finding F1 — event branch: tracking coverage vs rule firing (tuning, Kitware RGB labels)", "",
             f"Generated by `scripts/event_coverage.py` at git `{stamp['git_hash']}`, config `{stamp['config_hash'][:12]}`. "
             f"Rule setting: `{p}`. Tracked = a track box with IoU >= {cc['min_iou']} on >= "
             f"{pct(cc['min_frame_share'])} of the label's 5 fps frames. Fires = a detected event of the same type "
             f"within ±{tol} s (any track; loose upper bound). Information only (no gate uses it).", "",
             f"- labels: **{summary['labels']}**; tracked: **{pct(summary['tracked'])}**",
             f"- rule fires near a tracked label: **{pct(summary['fires_given_tracked'])}**; near an untracked one: "
             f"{pct(summary['fires_given_untracked'])}", "",
             "| camera | type | labels | tracked | fires | median label box h (px) |", "|---|---|---|---|---|---|"]
    for (cam, typ), r in by_cam.iterrows():
        lines.append(f"| {cam} | {typ} | {int(r['n'])} | {pct(r['tracked'])} | {pct(r['fires'])} | {r['median_box_h']:.0f} |")
    lines += ["", "| label box height (px) | labels | tracked | fires |", "|---|---|---|---|"]
    for b, r in by_size.iterrows():
        lines.append(f"| {b} | {int(r['n'])} | {pct(r['tracked'])} | {pct(r['fires'])} |")
    out = REPO_ROOT / "results" / "reports" / "events_6.1_coverage.md"
    out.write_text("\n".join(lines) + "\n")
    log_experiment(SPLIT, {k: float(v) for k, v in summary.items()},
                   f"Finding F1 coverage diagnostic ({p}); report {display_path(out)}", stamp=stamp)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
