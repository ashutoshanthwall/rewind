"""Step 3.3: detector smoke test — recall vs annotated boxes on tuning footage, speed, projected full-pass time.

Measure (replaces the 50-frame eyeball check, CLAUDE.md rule 16), tuning split only (split_video_ids refuses test):
  - Frames: the 5 fps sample grid of the full pass (frame % keep_step == 0). Per tuning clip, up to
    `frames_per_clip` grid frames that carry >= 1 ground-truth box, picked with a fixed seed.
  - Ground truth: RGB = Kitware actor boxes, typed from events.actor_types: 'person' -> person; 'vehicle' -> vehicle
    (other actor types, e.g. bag/bicycle/other, are not scored). IR = third-party (contrib) boxes, which have no
    usable type (all 'Person'), so IR is scored class-agnostic: any detection of a configured person/vehicle class
    may match. Boxes are clipped to the frame; within one frame and group, a box with IoU >= gt_dup_iou to an
    already kept box is the same object annotated in two activities and counts once.
  - Match: per frame and group, greedy one-to-one by descending IoU; a GT box is found if matched at IoU >=
    match_iou (0.5). Detections: config `detection` conf_thresh, nms_iou, classes; vehicle = events.vehicle_classes.
  - Recall per cell: RGB person, RGB vehicle, IR (any). Pre-registered (before any run): **overall recall = mean of
    the three cell recalls**; pooled box-level recall is reported too. If the two disagree on the Gate 3.3 outcome,
    the rule does not clearly decide.
  - Not measurable: precision (MEVA boxes only cover actors of annotated activities; unlabelled people and parked
    cars are everywhere).
  - Caveats: IR contrib boxes can be out of time with the object (docs/meva_notes.md) and are looser, so IR recall
    at IoU 0.5 is a lower bound; IoU 0.3 and "centre": a detection centre inside the GT box (one-to-one) are
    reported as well, and IR per label source.
  - Speed: decode, MOG2 and detector wall time are measured separately (detector: per batch of
    detection.batch_size frames incl. pre/post-processing and the GPU->CPU copy, first batch excluded as warm-up);
    projected full-pass time = video hours x measured decode+MOG2 cost + all 5 fps frames / detector frames per s.
Writes results/reports/detector_smoke_<name>.json, ~n_sample_images annotated frames to
results/figures/detector_smoke/<name>/ (GT green, detections red; gitignored), and one experiments.csv row.

Run from the repo root, in an environment that has the detector packages (see PROGRESS 3.3). All detectors given
run on the same decoded frames (one decode at the largest input size; each detector gets the frame scaled to its
own imgsz):
  <python> -m scripts.detector_smoke --detectors yolov8s_640=ultralytics:artifacts/weights/yolov8s.pt:640 \
      yolox_s_640=yolox:artifacts/weights/yolox_s.pth:640:yolox-s
  uv run python -m scripts.detector_smoke --report yolov8s_640 yolox_s_640     (markdown report from the JSONs)
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from rewind.detect.detectors import COCO_NAMES, class_ids, make_detector
from rewind.eval.splits import split_video_ids
from rewind.experiment_log import log_experiment
from rewind.ingest.manifest import connect, load_videos
from rewind.motion.mog2 import MotionConfig, MotionScorer
from rewind.paths import REPO_ROOT, display_path
from rewind.video import keep_step, sampled_frames, scaled_size

CELLS = ("rgb_person", "rgb_vehicle", "ir_any")
REPORTS = REPO_ROOT / "results" / "reports"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (n, 4) with every box in b (m, 4), x1 y1 x2 y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def greedy_match(gt: np.ndarray, det: np.ndarray, min_iou: float) -> np.ndarray:
    """One-to-one greedy matching by descending IoU; returns a bool per GT box (matched at IoU >= min_iou)."""
    found = np.zeros(len(gt), bool)
    iou = iou_matrix(gt, det)
    used = np.zeros(len(det), bool)
    for flat in np.argsort(-iou, axis=None):
        g, d = np.unravel_index(flat, iou.shape)
        if iou[g, d] < min_iou:
            break
        if not found[g] and not used[d]:
            found[g], used[d] = True, True
    return found


def size_bucket(box: np.ndarray, frame_h: int, edges: list[float]) -> str:
    """Box height as a fraction of the frame height, bucketed by `edges` (e.g. [0.05, 0.15])."""
    frac = (box[3] - box[1]) / frame_h
    for i, e in enumerate(edges):
        if frac < e:
            return f"<{e:g}" if i == 0 else f"{edges[i - 1]:g}-{e:g}"
    return f">={edges[-1]:g}"


def centre_match(gt: np.ndarray, det: np.ndarray) -> np.ndarray:
    """One-to-one: a GT box is found if an unused detection's centre lies inside it (smallest GT boxes first).
    For loose boxes (IR contrib), where IoU with a tight detection stays low even when the object is found."""
    found = np.zeros(len(gt), bool)
    if len(det) == 0:
        return found
    cx, cy = (det[:, 0] + det[:, 2]) / 2, (det[:, 1] + det[:, 3]) / 2
    used = np.zeros(len(det), bool)
    for g in np.argsort((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])):
        inside = ~used & (cx >= gt[g, 0]) & (cx <= gt[g, 2]) & (cy >= gt[g, 1]) & (cy <= gt[g, 3])
        if inside.any():
            used[np.flatnonzero(inside)[0]] = True
            found[g] = True
    return found


def load_ground_truth(conn, videos, step: int) -> dict:
    """{video_id: {frame: {kind: [(box, tag), ...]}}} on the sample grid (raw boxes). kind: person / vehicle (RGB)
    or any (IR); tag: the contrib source(s) for IR boxes, '' for RGB."""
    gt = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_id = {v.video_id: v for v in videos}
    rows = conn.execute(
        f"SELECT e.video_id, e.label_source, e.sources, e.actor_types, b.actor, b.frame, b.x1, b.y1, b.x2, b.y2 "
        f"FROM event_boxes b JOIN events e USING (event_id) "
        f"WHERE e.video_id IN ({','.join('?' * len(by_id))}) AND b.frame % ? = 0", (*by_id, step))
    for r in rows:
        v = by_id[r["video_id"]]
        box = (r["x1"], r["y1"], r["x2"], r["y2"])
        if v.modality == "ir":
            if r["label_source"] != "contrib":
                raise ValueError(f"{v.clip}: IR box from {r['label_source']}, expected contrib only (Gate 0.3)")
            gt[v.video_id][r["frame"]]["any"].append((box, "+".join(json.loads(r["sources"]))))
        else:
            if r["label_source"] == "contrib":
                raise ValueError(f"{v.clip}: contrib box on an RGB clip (Gate 0.3 forbids)")
            kind = json.loads(r["actor_types"])[r["actor"]]
            if kind in ("person", "vehicle"):
                gt[v.video_id][r["frame"]][kind].append((box, ""))
    return gt


def dedup_tagged(items: list, w: int, h: int, dup_iou: float) -> tuple[np.ndarray, list[str]]:
    """Clip to the frame, drop empty boxes, then drop near-duplicates (dedup); tags follow their boxes."""
    boxes = np.array([b for b, _ in items], np.float32)
    tags = [t for _, t in items]
    clipped = boxes.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, w)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, h)
    ok = [i for i in range(len(clipped)) if clipped[i, 2] > clipped[i, 0] and clipped[i, 3] > clipped[i, 1]]
    kept = []
    for i in ok:
        if not kept or iou_matrix(clipped[i:i + 1], clipped[kept]).max() < dup_iou:
            kept.append(i)
    return clipped[kept], [tags[i] for i in kept]


def draw(frame: np.ndarray, gt: dict, det, scale: float) -> np.ndarray:
    img = frame.copy()
    for boxes in gt.values():
        for b in boxes:
            cv2.rectangle(img, tuple(int(x / scale) for x in b[:2]), tuple(int(x / scale) for x in b[2:]), (0, 200, 0), 2)
    for b, s, c in zip(det.boxes, det.scores, det.classes):
        p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
        cv2.rectangle(img, p1, p2, (0, 0, 255), 1)
        cv2.putText(img, f"{COCO_NAMES[c]} {s:.2f}", (p1[0], max(10, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 0, 255), 1)
    return img


def parse_spec(spec: str) -> dict:
    """'name=backend:weights:imgsz[:yolox_exp]', e.g. 'yolov8s_640=ultralytics:artifacts/weights/yolov8s.pt:640'."""
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"bad detector spec {spec!r}")
    return {"name": name, "backend": parts[0], "weights": parts[1], "imgsz": int(parts[2]),
            "yolox_exp": parts[3] if len(parts) == 4 else None}


def run(args, cfg) -> None:
    dcfg, ccfg, mcfg = cfg["detection"], cfg["detection_check"], MotionConfig.from_config(cfg["motion"])
    if ccfg["match_iou"] not in ccfg["report_ious"]:
        raise ValueError("detection_check.match_iou must be one of report_ious")
    vehicle_ids = set(class_ids(cfg["events"]["vehicle_classes"]))
    person_id = class_ids(["person"])[0]
    dets = []
    for spec in map(parse_spec, args.detectors):
        d = {**spec, "found": {iou: defaultdict(lambda: [0, 0]) for iou in ccfg["report_ious"]},
             "time": defaultdict(float), "frames": defaultdict(int), "warm": True,
             "fig_dir": REPO_ROOT / "results" / "figures" / "detector_smoke" / spec["name"],
             "model": make_detector(spec["backend"], str(REPO_ROOT / spec["weights"]), spec["imgsz"],
                                    dcfg["conf_thresh"], dcfg["nms_iou"], dcfg["classes"], spec["yolox_exp"],
                                    args.legacy)}
        d["fig_dir"].mkdir(parents=True, exist_ok=True)
        dets.append(d)

    conn = connect(args.db)
    tuning = set(split_video_ids(conn, "tuning"))
    videos = [v for v in load_videos(conn) if v.video_id in tuning]
    if args.max_clips:
        videos = [v for m in ("rgb", "ir") for v in [x for x in videos if x.modality == m][-args.max_clips:]]
    step = keep_step(videos[0].fps, dcfg["fps"])
    if keep_step(videos[0].fps, mcfg.fps) != step:
        raise ValueError("motion.fps and detection.fps differ; the full pass runs both on the same frames")
    gt = load_ground_truth(conn, videos, step)
    rng = random.Random(ccfg["seed"])
    picked = {v.video_id: set(rng.sample(sorted(gt[v.video_id]), min(ccfg["frames_per_clip"], len(gt[v.video_id]))))
              for v in videos if gt[v.video_id]}
    # ~n_sample_images frames to draw, at least a quarter IR (if there are IR frames).
    modality = {v.video_id: v.modality for v in videos}
    all_ids = sorted((vid, f) for vid, fs in picked.items() for f in fs)
    ir_ids = [x for x in all_ids if modality[x[0]] == "ir"]
    rgb_ids = [x for x in all_ids if modality[x[0]] == "rgb"]
    n_ir = min(len(ir_ids), ccfg["n_sample_images"] // 4)
    image_ids = set(rng.sample(ir_ids, n_ir)) | set(rng.sample(rgb_ids, min(len(rgb_ids), ccfg["n_sample_images"] - n_ir)))

    timing, video_s = defaultdict(float), defaultdict(float)
    size_of = {r["video_id"]: (r["width"], r["height"]) for r in conn.execute("SELECT video_id, width, height FROM videos")}
    decode_width = max([mcfg.downscale_width] + [d["imgsz"] for d in dets])   # decode once at the largest size
    for n_clip, v in enumerate(videos, 1):
        if v.video_id not in picked:
            continue
        w, h = size_of[v.video_id]
        decoded_size = scaled_size(w, h, decode_width)
        motion_size = scaled_size(w, h, mcfg.downscale_width)
        scorer, keep = MotionScorer(mcfg, (w, h)), []
        t = time.perf_counter()
        frames_iter = sampled_frames(REPO_ROOT / v.path, v.n_frames, step, decode_width)
        while True:
            t0 = time.perf_counter()
            item = next(frames_iter, None)
            timing[f"decode_{v.modality}"] += time.perf_counter() - t0
            if item is None:
                break
            i, frame = item
            t0 = time.perf_counter()
            # MOG2 for timing only (warm-up order does not change the cost)
            scorer.score(i, frame if decoded_size == motion_size else
                         cv2.resize(frame, motion_size, interpolation=cv2.INTER_AREA))
            timing[f"mog2_{v.modality}"] += time.perf_counter() - t0
            if i in picked[v.video_id]:
                keep.append((i, frame))
        video_s[v.modality] += v.n_frames / v.fps
        for d in dets:
            size = scaled_size(w, h, d["imgsz"])
            scale = w / size[0]
            frames = [(i, f if size == decoded_size else cv2.resize(f, size, interpolation=cv2.INTER_AREA))
                      for i, f in keep]
            for b in range(0, len(frames), dcfg["batch_size"]):
                batch = frames[b:b + dcfg["batch_size"]]
                t0 = time.perf_counter()
                out = d["model"]([f for _, f in batch])
                dt = time.perf_counter() - t0
                if d["warm"]:
                    d["warm"] = False      # first batch: CUDA/cudnn warm-up, not timed
                else:
                    d["time"][v.modality] += dt
                    d["frames"][v.modality] += len(batch)
                for (i, frame), det in zip(batch, out):
                    det_boxes = det.boxes * scale
                    groups = {"person": det_boxes[det.classes == person_id],
                              "vehicle": det_boxes[np.isin(det.classes, list(vehicle_ids))],
                              "any": det_boxes[np.isin(det.classes, list(vehicle_ids | {person_id}))]}
                    drawn = {}
                    for kind, items in gt[v.video_id][i].items():
                        g, tags = dedup_tagged(items, w, h, ccfg["gt_dup_iou"])
                        drawn[kind] = g
                        cell = f"{v.modality}_{kind}"
                        hits = {iou: greedy_match(g, groups[kind], iou) for iou in ccfg["report_ious"]}
                        hits["centre"] = centre_match(g, groups[kind])
                        for key_iou, hit in hits.items():
                            counts = d["found"].setdefault(key_iou, defaultdict(lambda: [0, 0]))
                            for gb, tag, ok in zip(g, tags, hit):
                                keys = [cell, f"{cell}|{size_bucket(gb, h, ccfg['size_edges'])}", f"{cell}|{v.camera_id}"]
                                if tag:
                                    keys.append(f"{cell}|src={tag}")
                                for key in keys:
                                    counts[key][0] += int(ok)
                                    counts[key][1] += 1
                    if (v.video_id, i) in image_ids:
                        cv2.imwrite(str(d["fig_dir"] / f"{v.clip}_f{i:05d}.jpg"), draw(frame, drawn, det, scale),
                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"[{n_clip}/{len(videos)}] {v.clip}: {len(keep)} frames, {time.perf_counter() - t:.1f} s", flush=True)

    total_s = dict(conn.execute("SELECT modality, SUM(duration_s) FROM videos GROUP BY modality").fetchall())
    match_iou = ccfg["match_iou"]
    for d in dets:
        found = d["found"][match_iou]
        cells = {c: found[c][0] / found[c][1] for c in CELLS if found[c][1]}
        if len(cells) != len(CELLS):
            raise ValueError(f"cells without ground truth: {set(CELLS) - set(cells)}")
        pooled = sum(found[c][0] for c in CELLS) / sum(found[c][1] for c in CELLS)
        # Projection over the whole selection (manifest), per modality. Decode + MOG2 measured at the shared decode
        # width; with a single detector the decode happens at max(motion width, its imgsz) (VERIFY: same as here
        # only when that equals decode_width).
        proj = {}
        for m in ("rgb", "ir"):
            cpu_per_video_s = (timing[f"decode_{m}"] + timing[f"mog2_{m}"]) / video_s[m]
            det_fps = d["frames"][m] / d["time"][m]
            n = total_s[m] * dcfg["fps"]
            proj[m] = {"cpu_h": total_s[m] * cpu_per_video_s / 3600, "detect_h": n / det_fps / 3600,
                       "detect_fps": det_fps, "decode_mog2_x_realtime": 1 / cpu_per_video_s, "frames": n}
        result = {"name": d["name"], "backend": d["backend"], "weights": d["weights"], "imgsz": d["imgsz"],
                  "yolox_exp": d["yolox_exp"], "legacy": args.legacy, "decode_width": decode_width,
                  "detection_cfg": dcfg, "check_cfg": ccfg, "cells": cells,
                  "overall_mean_of_cells": float(np.mean(list(cells.values()))), "pooled": pooled,
                  "counts": {str(k): dict(c) for k, c in d["found"].items()},
                  "n_frames": sum(map(len, picked.values())), "projection": proj,
                  "projected_total_h": sum(p["cpu_h"] + p["detect_h"] for p in proj.values())}
        out = REPORTS / f"detector_smoke_{d['name']}.json"
        out.write_text(json.dumps(result, indent=1))
        print(d["name"], json.dumps({k: result[k] for k in ("cells", "overall_mean_of_cells", "pooled",
                                                            "projected_total_h")}), flush=True)
        if not args.no_log:
            metrics = {**{f"recall_{c}": r for c, r in cells.items()}, "recall_overall": result["overall_mean_of_cells"],
                       "recall_pooled": pooled, "detect_fps_rgb": proj["rgb"]["detect_fps"],
                       "projected_total_h": result["projected_total_h"]}
            log_experiment("tuning", metrics, f"step 3.3 detector smoke {d['name']} ({d['backend']}, {d['weights']}, "
                                              f"imgsz {d['imgsz']})")
        print(f"wrote {display_path(out)} and {len(image_ids)} images in {display_path(d['fig_dir'])}")


def report(names: list[str], cfg: dict) -> None:
    res = [json.loads((REPORTS / f"detector_smoke_{n}.json").read_text()) for n in names]
    iou = str(cfg["detection_check"]["match_iou"])
    lines = ["# Detector smoke test (step 3.3)", "",
             "Generated by `scripts/detector_smoke.py` (method in its docstring). Tuning split only. Recall = share "
             "of annotated actor boxes (5 fps grid frames) matched one-to-one by a detection of the right class at "
             f"IoU >= {iou}. Overall = mean of the three cells (pre-registered); pooled = all boxes together.", "",
             "| detector | frames | RGB person | RGB vehicle | IR (any class) | **overall** | pooled | IR @ IoU 0.3 | "
             "IR centre-in-box | detector fps (RGB / IR) | projected full pass |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        c = r["cells"]
        ir03, irc = r["counts"]["0.3"]["ir_any"], r["counts"]["centre"]["ir_any"]
        lines.append(f"| {r['name']} | {r['n_frames']} | {c['rgb_person']:.3f} | {c['rgb_vehicle']:.3f} | "
                     f"{c['ir_any']:.3f} | **{r['overall_mean_of_cells']:.3f}** | {r['pooled']:.3f} | "
                     f"{ir03[0] / ir03[1]:.3f} | {irc[0] / irc[1]:.3f} | {r['projection']['rgb']['detect_fps']:.0f} / "
                     f"{r['projection']['ir']['detect_fps']:.0f} | {r['projected_total_h']:.1f} h |")
    lines += ["", "Box counts (found / total) per cell, size bucket (box height / frame height) and camera:", ""]
    keys = sorted({k for r in res for k in r["counts"][iou]})
    lines += ["| key | " + " | ".join(r["name"] for r in res) + " |", "|---|" + "---|" * len(res)]
    for k in keys:
        vals = []
        for r in res:
            f, n = r["counts"][iou].get(k, [0, 0])
            vals.append(f"{f / n:.3f} ({f}/{n})" if n else "–")
        lines.append(f"| {k} | " + " | ".join(vals) + " |")
    lines += ["", "Projection (whole selection): " + "; ".join(
        f"{r['name']}: RGB decode+MOG2 {r['projection']['rgb']['cpu_h']:.2f} h + detect "
        f"{r['projection']['rgb']['detect_h']:.2f} h, IR {r['projection']['ir']['cpu_h']:.2f} h + "
        f"{r['projection']['ir']['detect_h']:.2f} h" for r in res) + ". One process, stages run one after the "
        "other (no overlap of CPU decode and GPU detection), so this is an upper bound for that design.", ""]
    out = REPORTS / "detector_3.3.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--detectors", nargs="+", metavar="SPEC",
                   help="name=backend:weights:imgsz[:yolox_exp]; all run on the same decoded frames")
    p.add_argument("--legacy", action="store_true", help="YOLOX pre-0.2 input normalisation")
    p.add_argument("--report", nargs="+", metavar="NAME")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--max-clips", type=int, help="debug: only the first N tuning clips per modality (implies --no-log)")
    args = p.parse_args()
    args.no_log = args.no_log or args.max_clips is not None
    cfg = yaml.safe_load(args.config.read_text())
    if args.report:
        report(args.report, cfg)
    elif args.detectors:
        run(args, cfg)
    else:
        p.error("give --detectors or --report")


if __name__ == "__main__":
    main()
