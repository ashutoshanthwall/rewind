"""Step 6.1: tune the vehicle_stops / vehicle_starts rules on tuning; check them held out on dev.

Pre-registered in PROGRESS.md ("Phase 6 — PRE-REGISTRATION"). Ground truth: Kitware vehicle_starts / vehicle_stops
(pieces joined by group_id) in the split's clips that carry Kitware labels (RGB; empty_37 clips included, their
detections are false positives). Detected events: rewind/track/tracks.clip_events on the tracks of those clips
(scripts/track_pass.py). Matching: one-to-one, same camera and type, annotated span widened by
metrics.event_tolerance_s (rewind/eval/metrics.match_count_us = match_events on integer microseconds).

Default mode (tuning): every setting of events.sweep (speed_unit x stop_speed x min_stopped_s x min_moving_s x
smooth_s); selection = highest mean over the two types of pooled F1, ties -> first in grid order; a numeric choice at
a grid edge extends that parameter's grid one step past the edge, once, and the selection is rerun. At the selected
setting: per camera and per type P / R / F1, Gate 6.2 part 1 per type (P >= min_p and R >= min_r), the time-shift
control (every detected span moved by shift_control_s, circular within its clip), and IR (G474) recall against the
incomplete contrib labels (information only). Writes results/reports/events_6.1.md + events_6.1_grid.csv, one
experiments row.
--heldout dev: the same report at the setting frozen in config `events` (results/reports/events_6.1_dev.md, one row).
The test split is refused (scripts read it only at step 8.2).

Run from the repo root:
  uv run python -m scripts.event_sweep                    (tuning sweep)
  uv run python -m scripts.event_sweep --heldout dev      (after the selected setting is in config.yaml)
"""

import argparse
import csv
import itertools
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.eval.metrics import match_count_us
from rewind.eval.splits import split_video_ids
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect, load_videos
from rewind.motion import store as motion_store
from rewind.paths import REPO_ROOT, display_path
from rewind.track import store
from rewind.track.tracks import EVENT_TYPES, RuleParams, best_box, detect_events, track_speed
from scripts.track_pass import settings_hash

EPOCH = datetime(2018, 1, 1)
US = timedelta(microseconds=1)
REPORTS = REPO_ROOT / "results" / "reports"
PARAMS = ("speed_unit", "stop_speed", "min_stopped_s", "min_moving_s", "smooth_s")


def to_us(t: datetime) -> int:
    return (t - EPOCH) // US


def clip_ids_by_source(conn, video_ids: list[int]) -> dict[str, set[int]]:
    """'kitware' / 'contrib' -> the given clips whose labels come from that source."""
    out = {"kitware": set(), "contrib": set()}
    for vid, src in conn.execute("SELECT video_id, label_source FROM clip_labels"):
        if vid in video_ids:
            if src.startswith("kitware"):
                out["kitware"].add(vid)
            elif src == "contrib":
                out["contrib"].add(vid)
    return out


def annotated(conn, clip_ids: set[int], kitware: bool) -> dict[tuple[str, str], np.ndarray]:
    """(camera, type) -> (n, 2) int64 microsecond spans of the labelled events in these clips; pieces cut at clip
    boundaries joined by group_id (a group touching a clip outside the set is kept whole)."""
    src = "label_source != 'contrib'" if kitware else "label_source = 'contrib'"
    groups = defaultdict(list)
    for gid, vid, cam, typ, a, b in conn.execute(
            f"SELECT group_id, video_id, camera_id, activity_type, start_time, end_time FROM events WHERE {src} "
            f"AND activity_type IN ({','.join('?' * len(EVENT_TYPES))})", EVENT_TYPES):
        groups[gid].append((vid, cam, typ, a, b))
    out = defaultdict(list)
    for pieces in groups.values():
        if not any(p[0] in clip_ids for p in pieces):
            continue
        if len({(p[1], p[2]) for p in pieces}) != 1:
            raise ValueError(f"an event group spans cameras or types: {pieces}")
        out[(pieces[0][1], pieces[0][2])].append(
            (min(to_us(datetime.fromisoformat(p[3])) for p in pieces),
             max(to_us(datetime.fromisoformat(p[4])) for p in pieces)))
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in out.items()}


class ClipTracks:
    """One clip's tracks with speeds cached per (unit, smooth_s)."""

    def __init__(self, v, tracks: pd.DataFrame):
        self.v = v
        self.base_us = to_us(v.start_time)
        self.duration = v.n_frames / v.fps
        self.tracks = [(tr["t_s"].to_numpy(float), tr[["x1", "y1", "x2", "y2"]].to_numpy(float))
                       for _, tr in tracks.sort_values(["track", "t_s"]).groupby("track", sort=True)]
        self._speed = {}

    def speeds(self, unit: str, smooth: float) -> list[np.ndarray]:
        key = (unit, smooth)
        if key not in self._speed:
            self._speed[key] = [track_speed(t, b, smooth, unit) for t, b in self.tracks]
        return self._speed[key]

    def events(self, p: RuleParams, shift_s: float = 0.0) -> dict[str, np.ndarray]:
        """type -> (n, 2) int64 microsecond spans; shift_s moves each span, circular within the clip."""
        out = {t: [] for t in EVENT_TYPES}
        for (t, _), sp in zip(self.tracks, self.speeds(p.speed_unit, p.smooth_s)):
            for etype, s0, s1 in detect_events(t, sp, p):
                if shift_s:
                    s0, s1 = s0 + shift_s, s1 + shift_s
                    if s0 >= self.duration:
                        s0, s1 = s0 - self.duration, s1 - self.duration
                out[etype].append((self.base_us + round(s0 * 1e6), self.base_us + round(s1 * 1e6)))
        return {k: np.array(v, dtype=np.int64).reshape(-1, 2) for k, v in out.items()}


def counts(clips: list[ClipTracks], ann: dict, p: RuleParams, tol_us: int, shift_s: float = 0.0) -> dict:
    """(camera, type) -> [detected, annotated, matched]. Matching per camera over all its clips at once."""
    det = defaultdict(list)
    for c in clips:
        for typ, spans in c.events(p, shift_s).items():
            det[(c.v.camera_id, typ)].append(spans)
    cams = {c.v.camera_id for c in clips}
    out = {}
    for cam in sorted(cams):
        for typ in EVENT_TYPES:
            d = np.concatenate(det[(cam, typ)]) if det[(cam, typ)] else np.empty((0, 2), np.int64)
            a = ann.get((cam, typ), np.empty((0, 2), np.int64))
            out[(cam, typ)] = [len(d), len(a), match_count_us(d, a, tol_us)]
    return out


def prf(n_det: int, n_ann: int, n_match: int) -> tuple[float, float, float]:
    """(P, R, F1); P is NaN with no detections and R NaN with no labels (0/0 has no honest value); F1 is 0 then."""
    p = n_match / n_det if n_det else float("nan")
    r = n_match / n_ann if n_ann else float("nan")
    f1 = 2 * p * r / (p + r) if n_det and n_ann and p + r > 0 else 0.0
    return p, r, f1


def pooled(c: dict) -> dict[str, tuple[int, int, int]]:
    out = {}
    for typ in EVENT_TYPES:
        out[typ] = tuple(int(sum(v[i] for (cam, t), v in c.items() if t == typ)) for i in range(3))
    return out


def grid(sweep: dict) -> list[RuleParams]:
    out = []
    for unit in sweep["speed_unit"]:
        for spd, ms, mm, sm in itertools.product(sweep["stop_speed"][unit], sweep["min_stopped_s"],
                                                 sweep["min_moving_s"], sweep["smooth_s"]):
            out.append(RuleParams(unit, float(spd), float(ms), float(mm), float(sm)))
    return out


def extend_at_edges(sweep: dict, best: RuleParams) -> tuple[dict, list[str]]:
    """The pre-registered one-time extension: a numeric choice at its grid edge gets one more value past that edge."""
    new, notes = dict(sweep), []
    new["stop_speed"] = {u: list(v) for u, v in sweep["stop_speed"].items()}
    speeds = sorted(sweep["stop_speed"][best.speed_unit])
    if best.stop_speed == speeds[0]:
        new["stop_speed"][best.speed_unit] = [speeds[0] / 2] + speeds
        notes.append(f"stop_speed at low edge -> added {speeds[0] / 2}")
    elif best.stop_speed == speeds[-1]:
        new["stop_speed"][best.speed_unit] = speeds + [speeds[-1] * 2]
        notes.append(f"stop_speed at high edge -> added {speeds[-1] * 2}")
    for key, low, high in (("min_stopped_s", -0.2, 1.0), ("min_moving_s", -0.2, 1.0), ("smooth_s", -0.2, 0.6)):
        vals = sorted(float(x) for x in sweep[key])
        val = getattr(best, key)
        if val == vals[0] and round(vals[0] + low, 6) > 0:
            new[key] = [round(vals[0] + low, 6)] + vals
            notes.append(f"{key} at low edge -> added {round(vals[0] + low, 6)}")
        elif val == vals[-1]:
            new[key] = vals + [round(vals[-1] + high, 6)]
            notes.append(f"{key} at high edge -> added {round(vals[-1] + high, 6)}")
    return new, notes


def score(c: dict) -> float:
    return float(np.mean([prf(*v)[2] for v in pooled(c).values()]))


def load_clips(conn, cfg: dict, video_ids: set[int]) -> list[ClipTracks]:
    art = REPO_ROOT / "artifacts"
    videos = [v for v in load_videos(conn) if v.video_id in video_ids]
    det_settings = {motion_store.load_marker(art, v)["settings"] for v in videos}
    if len(det_settings) != 1:
        raise ValueError(f"detections with {len(det_settings)} settings")
    settings = settings_hash(cfg, det_settings.pop())
    return [ClipTracks(v, store.load_tracks(art, v, settings)) for v in videos]


def fmt(x: float, nd: int = 3) -> str:
    return "–" if x != x else f"{x:.{nd}f}"


def camera_table(c: dict, title: str) -> list[str]:
    lines = [f"**{title}**", "", "| camera | type | detected | labelled | matched | P | R | F1 |", "|---|---|---|---|---|---|---|---|"]
    for (cam, typ), v in sorted(c.items()):
        p, r, f = prf(*v)
        lines.append(f"| {cam} | {typ} | {v[0]} | {v[1]} | {v[2]} | {fmt(p)} | {fmt(r)} | {fmt(f)} |")
    for typ, v in pooled(c).items():
        p, r, f = prf(*v)
        lines.append(f"| **pooled** | {typ} | {v[0]} | {v[1]} | {v[2]} | **{fmt(p)}** | **{fmt(r)}** | {fmt(f)} |")
    return lines + [""]


def evaluate_at(conn, cfg: dict, split: str, p: RuleParams, clips: list[ClipTracks], ids: dict) -> dict:
    tol_us = round(cfg["metrics"]["event_tolerance_s"] * 1e6)
    kit = [c for c in clips if c.v.video_id in ids["kitware"]]
    ann_k = annotated(conn, ids["kitware"], kitware=True)
    res = {"main": counts(kit, ann_k, p, tol_us),
           "shift": counts(kit, ann_k, p, tol_us, shift_s=cfg["events"]["sweep"]["shift_control_s"])}
    ir = [c for c in clips if c.v.video_id in ids["contrib"]]
    if ir:
        res["ir"] = counts(ir, annotated(conn, ids["contrib"], kitware=False), p, tol_us)
    res["hours"] = {cam: sum(c.duration for c in kit if c.v.camera_id == cam) / 3600
                    for cam in sorted({c.v.camera_id for c in kit})}
    return res


def report_body(res: dict, p: RuleParams, cfg: dict) -> list[str]:
    sw = cfg["events"]["sweep"]
    lines = [f"Setting: `{p}`", ""]
    lines += camera_table(res["main"], "Per camera (Kitware RGB labels)")
    lines += ["Footage scored (h): " + ", ".join(f"{k} {v:.2f}" for k, v in res["hours"].items()), ""]
    lines += camera_table(res["shift"], f"Time-shift control: every detected span moved +{sw['shift_control_s']} s "
                                         "(circular within its clip); chance level for these dense labels")
    if "ir" in res:
        lines += ["**IR (G474) vs contrib labels** (incomplete labels: recall only, information only)", "",
                  "| type | detected | labelled | matched | R |", "|---|---|---|---|---|"]
        for typ, v in pooled(res["ir"]).items():
            lines.append(f"| {typ} | {v[0]} | {v[1]} | {v[2]} | {fmt(prf(*v)[1])} |")
        lines.append("")
    return lines


def finite(metrics: dict, notes: str) -> tuple[dict, str]:
    """experiments.csv takes finite numbers only: undefined values (NaN, e.g. P with no detections) are left out of
    the metrics and named in the notes."""
    nan = sorted(k for k, v in metrics.items() if v != v)
    return {k: v for k, v in metrics.items() if v == v}, notes + (f"; undefined (0/0), not logged: {nan}" if nan else "")


def gate_part1(res: dict, cfg: dict) -> dict[str, bool]:
    sw = cfg["events"]["sweep"]
    out = {}
    for typ, v in pooled(res["main"]).items():
        p, r, _ = prf(*v)
        out[typ] = bool(p == p and r == r and p >= sw["min_p"] and r >= sw["min_r"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    ap.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    ap.add_argument("--heldout", choices=["dev"], help="evaluate the setting frozen in config on this split")
    args = ap.parse_args()
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    conn = connect(args.db)
    split = args.heldout or "tuning"
    vids = split_video_ids(conn, split)          # refuses the test split
    ids = clip_ids_by_source(conn, vids)
    clips = load_clips(conn, cfg, set(vids))
    tol_us = round(cfg["metrics"]["event_tolerance_s"] * 1e6)

    if args.heldout:
        p = RuleParams.from_config(cfg)
        res = evaluate_at(conn, cfg, split, p, clips, ids)
        lines = [f"# Step 6.1 — vehicle stop/start rules, held-out check on {split}", "",
                 "Frozen setting from config (selected on tuning). Information only: not part of Gate 6.2.", ""]
        lines += report_body(res, p, cfg)
        out = REPORTS / f"events_6.1_{split}.md"
        out.write_text("\n".join(lines) + "\n")
        metrics = {}
        for typ, v in pooled(res["main"]).items():
            pp, r, f = prf(*v)
            metrics |= {f"{typ}_P": pp, f"{typ}_R": r, f"{typ}_F1": f, f"{typ}_n_det": v[0], f"{typ}_n_ann": v[1]}
        log_experiment(split, *finite(metrics, f"step 6.1 event rules held-out ({p}); report {display_path(out)}"),
                       stamp=stamp)
        print("\n".join(lines))
        return

    sweep = cfg["events"]["sweep"]
    kit = [c for c in clips if c.v.video_id in ids["kitware"]]
    ann = annotated(conn, ids["kitware"], kitware=True)
    print(f"tuning: {len(kit)} Kitware clips, labels " +
          ", ".join(f"{t} {sum(len(v) for (c, tt), v in ann.items() if tt == t)}" for t in EVENT_TYPES), flush=True)
    rows, done = [], {}

    def run(settings: list[RuleParams], round_name: str):
        for p in settings:
            if p in done:
                continue
            c = counts(kit, ann, p, tol_us)
            done[p] = c
            row = {"round": round_name, **{k: getattr(p, k) for k in PARAMS}, "mean_F1": score(c)}
            for typ, v in pooled(c).items():
                pp, r, f = prf(*v)
                row |= {f"{typ}_det": v[0], f"{typ}_ann": v[1], f"{typ}_match": v[2], f"{typ}_P": pp,
                        f"{typ}_R": r, f"{typ}_F1": f}
            rows.append(row)
        print(f"{round_name}: {len(done)} settings evaluated", flush=True)

    g1 = grid(sweep)
    run(g1, "grid")
    best = max(g1, key=lambda p: (score(done[p]), -g1.index(p)))
    ext_sweep, notes = extend_at_edges(sweep, best)
    final_grid = g1
    if notes:
        g2 = grid(ext_sweep)
        run(g2, "edge extension")
        final_grid = g2
        best = max(g2, key=lambda p: (score(done[p]), -g2.index(p)))   # ties: first in the extended grid's order
    res = evaluate_at(conn, cfg, split, best, clips, ids)
    passed = gate_part1(res, cfg)

    REPORTS.mkdir(parents=True, exist_ok=True)
    grid_csv = REPORTS / "events_6.1_grid.csv"
    with grid_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    top = sorted(rows, key=lambda r: -r["mean_F1"])[:10]
    lines = ["# Step 6.1 — vehicle stop/start rules tuned on tuning (Kitware RGB labels)", "",
             f"Pre-registered in PROGRESS.md (\"Phase 6 — PRE-REGISTRATION\"). Tolerance "
             f"{cfg['metrics']['event_tolerance_s']} s. {len(rows)} settings; full grid: {display_path(grid_csv)}.", "",
             f"Edge extension: {'; '.join(notes) if notes else 'none needed'}.", "",
             "## Top 10 settings (mean of the two types' pooled F1)", "",
             "| unit | stop_speed | min_stopped_s | min_moving_s | smooth_s | stops P/R | starts P/R | mean F1 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in top:
        lines.append(f"| {r['speed_unit']} | {r['stop_speed']} | {r['min_stopped_s']} | {r['min_moving_s']} | "
                     f"{r['smooth_s']} | {fmt(r['vehicle_stops_P'])} / {fmt(r['vehicle_stops_R'])} | "
                     f"{fmt(r['vehicle_starts_P'])} / {fmt(r['vehicle_starts_R'])} | {r['mean_F1']:.3f} |")
    lines += ["", "## Selected setting", ""] + report_body(res, best, cfg)
    lines += ["## Gate 6.2 part 1 (per type: pooled tuning P >= "
              f"{sweep['min_p']} and R >= {sweep['min_r']})", ""]
    lines += [f"- {typ}: **{'PASS' if ok else 'FAIL'}**" for typ, ok in passed.items()]
    lines += ["", "Caveat: measured on the split the setting was chosen on (optimistic); dev held-out check: "
                  "events_6.1_dev.md."]
    out = REPORTS / "events_6.1.md"
    out.write_text("\n".join(lines) + "\n")
    metrics = {"settings": len(rows), "mean_F1": score(res["main"])}
    for typ, v in pooled(res["main"]).items():
        pp, r, f = prf(*v)
        metrics |= {f"{typ}_P": pp, f"{typ}_R": r, f"{typ}_F1": f}
    log_experiment("tuning", *finite(metrics, f"step 6.1 event-rule sweep, selected {best}; gate part 1 {passed}; "
                                              f"report {display_path(out)}"), stamp=stamp)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
