"""Step 3.1 check: MOG2 on ~10 minutes of one tuning camera, activity inside labelled events vs random windows.

Measure (replaces the eyeball check, CLAUDE.md rule 16): for every Kitware event in the chosen clips, its activity =
mean per-second activity over the seconds it overlaps; each event is paired with `random_windows_per_event` seeded
random windows of the same length in the same clips (a window never crosses a clip edge). Reported:
  - AUC = P(event activity > random window activity), ties count 1/2 (0.5 = no signal);
  - median activity in events vs random windows, overall and for events with / without a vehicle actor;
  - per-second: seconds covered by >= 1 event vs seconds covered by none (if both exist).
On a busy camera nearly every second is inside some event, so the whole-frame score cannot separate events from
random windows. Second, local measure (same place, other time): box coverage of an event = mean over its actor boxes
at the sampled frames of max over motion boxes of |actor box & motion box| / |actor box|. The null is the same
actor boxes moved, as one block, to a random start in the same clip that does not overlap the event. AUC as above.
Also writes a plot (activity over time, events as bars below) to results/figures/ and one experiments.csv row.
Tuning split only (split_video_ids refuses test).

Run from the repo root:
  uv run python -m scripts.check_motion --camera G436 --minutes 10
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from rewind.eval.splits import split_video_ids
from rewind.experiment_log import log_experiment
from rewind.ingest.manifest import connect, load_videos
from rewind.motion.mog2 import MotionConfig, per_second, score_clip
from rewind.paths import REPO_ROOT, display_path
from rewind.video import keep_step, sampled_frames, scaled_size


def pick_clips(videos, camera: str, minutes: float, max_gap_s: float) -> list:
    """The first back-to-back run of the camera's clips (chronological) that reaches `minutes`."""
    run = []
    for v in sorted((v for v in videos if v.camera_id == camera), key=lambda v: v.start_time):
        if run and abs((v.start_time - run[-1].end_time).total_seconds()) > max_gap_s:
            run = []
        run.append(v)
        if sum(x.n_frames / x.fps for x in run) >= minutes * 60 - max_gap_s:
            return run
    raise ValueError(f"no back-to-back run of {minutes} min for {camera} in these clips")


def auc(pos: list[float], neg: list[float]) -> float:
    """P(pos > neg) over all pairs, ties 1/2 (Mann-Whitney U / (n_pos * n_neg))."""
    neg_sorted = np.sort(np.asarray(neg))
    total = 0.0
    for p in pos:
        below = np.searchsorted(neg_sorted, p, side="left")
        ties = np.searchsorted(neg_sorted, p, side="right") - below
        total += below + 0.5 * ties
    return total / (len(pos) * len(neg))


def event_seconds(start_s: float, end_s: float, n_seconds: int) -> range:
    """Clip seconds overlapping the event span [start_s, end_s)."""
    return range(int(start_s), min(n_seconds, int(np.ceil(end_s))))


def coverage(actor: tuple[int, int, int, int], motion: list[tuple[int, int, int, int]]) -> float:
    """Largest fraction of the actor box covered by one motion box (0 if no motion box)."""
    ax1, ay1, ax2, ay2 = actor
    area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    if area == 0:
        return 0.0
    best = 0
    for bx1, by1, bx2, by2 in motion:
        best = max(best, max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1)))
    return best / area


def lanes(spans: list[tuple[float, float]]) -> list[int]:
    """Greedy interval packing: lane index per span so that spans in one lane do not overlap (for drawing)."""
    ends, out = [], []
    for s, e in spans:
        for i, end in enumerate(ends):
            if end <= s:
                ends[i] = e
                out.append(i)
                break
        else:
            ends.append(e)
            out.append(len(ends) - 1)
    return out


def plot(activity: np.ndarray, events: list[dict], title: str, path: Path) -> None:
    """Activity line on top, one bar per event below (vehicle actor: orange, else blue). OpenCV only."""
    W, top_h, lane_h, margin = 1600, 300, 6, 60
    order = sorted(events, key=lambda e: e["t0"])
    lane = lanes([(e["t0"], e["t1"]) for e in order])
    n_lanes = max(lane, default=-1) + 1
    H = top_h + n_lanes * lane_h + 2 * margin
    img = np.full((H, W + 2 * margin, 3), 255, np.uint8)
    T = len(activity)
    ymax = max(float(activity.max()), 1e-6)
    x = lambda t: int(margin + t / T * W)
    pts = np.array([[x(i + 0.5), int(margin + top_h - a / ymax * top_h)] for i, a in enumerate(activity)], np.int32)
    cv2.rectangle(img, (margin, margin), (margin + W, margin + top_h), (200, 200, 200), 1)
    cv2.polylines(img, [pts], False, (40, 40, 40), 1, cv2.LINE_AA)
    for e, ln in zip(order, lane):
        y = margin + top_h + 4 + ln * lane_h
        color = (0, 140, 255) if e["vehicle"] else (200, 120, 0)
        cv2.rectangle(img, (x(e["t0"]), y), (max(x(e["t1"]), x(e["t0"]) + 1), y + lane_h - 2), color, -1)
    for t in range(0, T + 1, 60):
        cv2.line(img, (x(t), margin + top_h), (x(t), margin + top_h + 6), (0, 0, 0), 1)
        cv2.putText(img, f"{t // 60} min", (x(t) - 15, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, f"{title}   activity max {ymax:.3f}   bars: events (orange = vehicle actor, blue = other)",
                (margin, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera", required=True)
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--report", type=Path, default=None, help="default results/reports/motion_3.1_<camera>.md")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()
    args.report = args.report or REPO_ROOT / "results" / "reports" / f"motion_3.1_{args.camera}.md"
    cfg = yaml.safe_load(args.config.read_text())
    mcfg, check = MotionConfig.from_config(cfg["motion"]), cfg["motion_check"]

    conn = connect(args.db)
    tuning = set(split_video_ids(conn, "tuning"))
    videos = [v for v in load_videos(conn) if v.video_id in tuning]
    clips = pick_clips(videos, args.camera, args.minutes, cfg["ingest"]["max_name_duration_diff_s"])

    per_clip, boxes_at, t0 = {}, {}, time.perf_counter()
    for v in clips:
        w, h = conn.execute("SELECT width, height FROM videos WHERE video_id = ?", (v.video_id,)).fetchone()
        samples = sampled_frames(REPO_ROOT / v.path, v.n_frames, keep_step(v.fps, mcfg.fps), mcfg.downscale_width)
        frames = score_clip(samples, mcfg, (w, h))
        per_clip[v.video_id] = per_second(frames, v.fps, v.n_frames)
        boxes_at[v.video_id] = {f.frame: f.boxes for f in frames}
        n_boxes = [len(f.boxes) for f in frames]
        print(f"{v.clip}: {len(frames)} samples, analysis size {scaled_size(w, h, mcfg.downscale_width)}, "
              f"activity mean {per_clip[v.video_id].mean():.4f}, boxes/frame mean {np.mean(n_boxes):.2f}", flush=True)
    elapsed = time.perf_counter() - t0
    video_s = sum(v.n_frames / v.fps for v in clips)

    events = []
    for v in clips:
        act = per_clip[v.video_id]
        for r in conn.execute("SELECT event_id, activity_type, start_s, end_s, actor_types FROM events "
                              "WHERE video_id = ? AND label_source != 'contrib'", (v.video_id,)):
            secs = event_seconds(r["start_s"], r["end_s"], len(act))
            events.append({"video_id": v.video_id, "type": r["activity_type"], "len": len(secs),
                           "activity": float(act[list(secs)].mean()),
                           "vehicle": "vehicle" in json.loads(r["actor_types"] or "[]"),
                           "secs": secs, "t0": r["start_s"], "t1": r["end_s"]})
    if not events:
        raise ValueError("no Kitware events in the chosen clips")

    rng = random.Random(check["seed"])
    lengths = {v.video_id: len(per_clip[v.video_id]) for v in clips}
    for e in events:
        fits = [vid for vid, n in lengths.items() if n >= e["len"]]
        e["random"] = []
        for _ in range(check["random_windows_per_event"]):
            vid = rng.choices(fits, weights=[lengths[f] - e["len"] + 1 for f in fits])[0]
            s = rng.randrange(lengths[vid] - e["len"] + 1)
            e["random"].append(float(per_clip[vid][s:s + e["len"]].mean()))

    def summary(sel: list[dict]) -> dict:
        pos = [e["activity"] for e in sel]
        neg = [x for e in sel for x in e["random"]]
        return {"n": len(sel), "auc": auc(pos, neg), "median_event": statistics.median(pos),
                "median_random": statistics.median(neg)}

    groups = {"all": events, "vehicle actor": [e for e in events if e["vehicle"]],
              "no vehicle actor": [e for e in events if not e["vehicle"]]}
    rows = {name: summary(sel) for name, sel in groups.items() if sel}

    covered, uncovered = [], []
    for v in clips:
        mask = np.zeros(lengths[v.video_id], bool)
        for e in events:
            if e["video_id"] == v.video_id:
                mask[list(e["secs"])] = True
        covered += per_clip[v.video_id][mask].tolist()
        uncovered += per_clip[v.video_id][~mask].tolist()
    seconds_covered = len(covered) / (len(covered) + len(uncovered))

    step = keep_step(clips[0].fps, mcfg.fps)
    local_pos, local_neg = {"all": [], "vehicle actor": [], "no vehicle actor": []}, {}
    for k in local_pos:
        local_neg[k] = []
    n_skipped = 0
    for v in clips:
        sampled = boxes_at[v.video_id]
        for r in conn.execute("SELECT e.event_id, e.start_frame, e.end_frame, e.actor_types FROM events e "
                              "WHERE e.video_id = ? AND e.label_source != 'contrib'", (v.video_id,)).fetchall():
            boxes = conn.execute("SELECT frame, x1, y1, x2, y2 FROM event_boxes WHERE event_id = ? AND frame % ? = 0",
                                 (r["event_id"], step)).fetchall()
            if not boxes:
                n_skipped += 1
                continue
            start, length = r["start_frame"], r["end_frame"] - r["start_frame"] + 1
            # Shifts are multiples of step, so the shifted sample frames stay on the sample grid; no overlap with
            # the event itself.
            starts = [t for t in range(start % step, v.n_frames - length + 1, step)
                      if t + length <= start or t >= start + length]
            if not starts:
                n_skipped += 1
                continue
            pos = statistics.mean(coverage(tuple(b[1:]), sampled[b[0]]) for b in boxes)
            neg = []
            for _ in range(check["random_windows_per_event"]):
                shift = rng.choice(starts) - start
                neg.append(statistics.mean(coverage(tuple(b[1:]), sampled[b[0] + shift]) for b in boxes))
            kind = "vehicle actor" if "vehicle" in json.loads(r["actor_types"] or "[]") else "no vehicle actor"
            for k in ("all", kind):
                local_pos[k].append(pos)
                local_neg[k] += neg
    local_rows = {k: {"n": len(local_pos[k]), "auc": auc(local_pos[k], local_neg[k]),
                      "median_event": statistics.median(local_pos[k]), "median_random": statistics.median(local_neg[k])}
                  for k in local_pos if local_pos[k]}

    # One time axis over the back-to-back clips for the plot.
    offset, timeline, plot_events = 0, [], []
    for v in clips:
        timeline.append(per_clip[v.video_id])
        plot_events += [{**e, "t0": e["t0"] + offset, "t1": e["t1"] + offset} for e in events
                        if e["video_id"] == v.video_id]
        offset += lengths[v.video_id]
    fig = REPO_ROOT / "results" / "figures" / f"motion_3.1_{args.camera}.png"
    plot(np.concatenate(timeline), plot_events, f"{args.camera} {clips[0].clip[:19]} +{args.minutes:g} min (tuning)",
         fig)

    lines = [
        "# Motion pass check (step 3.1)", "",
        f"Generated by `uv run python -m scripts.check_motion --camera {args.camera} --minutes {args.minutes:g}`. "
        f"Tuning clips: {', '.join(v.clip for v in clips)}. MOG2 settings: config `motion` "
        f"({json.dumps(cfg['motion'])}). Decode + MOG2: {elapsed:.0f} s for {video_s / 60:.1f} min of video "
        f"({video_s / elapsed:.0f}x realtime, one process).", "",
        f"Kitware events: {len(events)}; seconds covered by >= 1 event: {seconds_covered:.0%}. Each event vs "
        f"{check['random_windows_per_event']} random windows of the same length (seed {check['seed']}).", "",
        "| events | n | AUC (event > random) | median activity, events | median activity, random windows |",
        "|---|---|---|---|---|",
        *[f"| {name} | {r['n']} | {r['auc']:.3f} | {r['median_event']:.4f} | {r['median_random']:.4f} |"
          for name, r in rows.items()], "",
    ]
    if covered and uncovered:
        lines += [f"Per second: covered by an event {len(covered)} s, median {statistics.median(covered):.4f}; "
                  f"not covered {len(uncovered)} s, median {statistics.median(uncovered):.4f}; "
                  f"AUC {auc(covered, uncovered):.3f}.", ""]
    lines += ["Local measure: box coverage of the annotated actor boxes at the 5 fps samples, event vs the same boxes "
              f"at {check['random_windows_per_event']} random non-overlapping times in the same clip "
              f"({n_skipped} events without boxes or room skipped).", "",
              "| events | n | AUC (event > random) | median coverage, events | median coverage, random |",
              "|---|---|---|---|---|",
              *[f"| {name} | {r['n']} | {r['auc']:.3f} | {r['median_event']:.3f} | {r['median_random']:.3f} |"
                for name, r in local_rows.items()], ""]
    lines += [f"Plot: `{display_path(fig)}`.", ""]
    args.report.write_text("\n".join(lines))
    print("\n".join(lines))
    if not args.no_log:
        metrics = {"auc_all": rows["all"]["auc"], "median_event": rows["all"]["median_event"],
                   "median_random": rows["all"]["median_random"], "n_events": len(events),
                   "event_coverage": seconds_covered, "x_realtime": video_s / elapsed,
                   "local_auc_all": local_rows["all"]["auc"]}
        log_experiment("tuning", metrics, f"step 3.1 motion check {args.camera} {clips[0].clip} +{args.minutes:g} min")
    print(f"report: {display_path(args.report)}")


if __name__ == "__main__":
    main()
