"""Rewind API (step 7.3): FastAPI app around the C++ engine. One process; everything is loaded once at startup.

Startup: config, manifest (videos, proxies), the engine's per-camera indexes of every active branch
(rewind/search/engine.py), the text encoders (CLIP for appearance, LanguageBind for action) and the fusion weights
(search.weights; a TBD or negative weight stops the start-up).

GET /search?q=...&cameras=G330,G336&start=2018-03-07T10:00:00&end=2018-03-07T11:00:00&k=20
  q: the description; cameras: comma-separated (default all); start / end: wall-clock (camera clock, ISO, both or
  neither); k: clips returned (default api.default_clips, at most api.max_clips). Each active branch with weight > 0
  returns its top search.k_per_index vectors in the selection -> fusion and clips (rewind/search/fusion.py).
  Response: the clips (camera, video, proxy, span in the video and in wall-clock time, score, the branches that
  matched and each branch's best hit: time, cosine, z, and for appearance the box and crop type), coverage of the
  selection, and timings. No clip is an event claim: the response says "candidate clips; people judge them".
GET /coverage?cameras=...&start=...&end=...  hours of indexed footage in the selection, per camera.
  Every /search response carries the same coverage, so an empty result is never a bare "no results".
Step 9.1 (the UI, ui/index.html, served at GET /):
GET /footage   cameras (modality, hours, continuous spans) and the UI settings (config section `ui`).
GET /motion?cameras=...&start=...&end=...&k=...  the fallback when the event is not in the list: segments of the
  selection ranked by mean MOG2 activity (the step 3.2 motion baseline, same code), most movement first.
GET /thumb?video_id=...&t=...&box=x1,y1,x2,y2  JPEG of the proxy frame at t seconds (box in original pixels, drawn
  scaled to the proxy). Decoded on request and cached in memory only: no frame is written to disk.
GET /proxy/{video_id}  the playback copy (MP4, faststart; byte ranges for seeking). Originals are never served.

Threads: FastAPI runs these sync endpoints in a thread pool. Engine indexes have their own locks (C++); the GPU text
encoders are used under one lock here.

Run from the repo root:  uv run uvicorn rewind.api.app:app --port 8000
"""

import os
import threading
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import av
import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from rewind.embed.clip_model import Clip
from rewind.embed.languagebind import LanguageBind
from rewind.eval.baselines import motion_ranked, motion_scores, segments
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT
from rewind.search.core import SearchConfig
from rewind.search.engine import Engine, EngineConfig, active_branches, branch_model, wall_seconds
from rewind.search.fusion import check_weights, fuse

HIT_FIELDS = {"appearance": ["t_s", "score", "z", "type", "view", "crop_source", "class", "x1", "y1", "x2", "y2"],
              "action": ["t_s", "window_start", "score", "z"]}
NOTE = "Candidate clips ranked by similarity to the description; people judge them. Similarity is not proof."
UI_DIR = REPO_ROOT / "ui"


def parse_time(s: str | None, name: str) -> datetime | None:
    if s is None:
        return None
    try:
        t = datetime.fromisoformat(s)
    except ValueError:
        raise HTTPException(400, f"{name}: not an ISO date-time: {s!r}")
    if t.tzinfo is not None:
        raise HTTPException(400, f"{name}: give camera wall-clock time without a time zone")
    return t


def overlap_s(a0: datetime, a1: datetime, b0: datetime | None, b1: datetime | None) -> float:
    lo = a0 if b0 is None else max(a0, b0)
    hi = a1 if b1 is None else min(a1, b1)
    return max(0.0, (hi - lo).total_seconds())


def merged_spans(videos, join_gap_s: float) -> dict[str, list[list[datetime]]]:
    """camera -> continuous footage spans [start, end): videos of a camera within join_gap_s of each other merge."""
    out: dict[str, list[list[datetime]]] = {}
    for v in sorted(videos, key=lambda v: (v.camera_id, v.start_time)):
        rs = out.setdefault(v.camera_id, [])
        if rs and v.start_time <= rs[-1][1] + timedelta(seconds=join_gap_s):
            rs[-1][1] = max(rs[-1][1], v.end_time)
        else:
            rs.append([v.start_time, v.end_time])
    return out


def frame_at(path: Path, t: float) -> np.ndarray:
    """BGR frame of a proxy at t seconds: seek to the keyframe at or before t, decode forward to the first frame whose
    time reaches t (within half a frame). Raises if the clip ends first."""
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        c.seek(int(t / st.time_base) + (st.start_time or 0), stream=st, backward=True)
        half = 0.5 / float(st.average_rate)
        for fr in c.decode(st):
            if fr.time is not None and fr.time >= t - half:
                return fr.to_ndarray(format="bgr24")
    raise ValueError(f"{path.name}: no frame at {t:.2f} s")


def draw_thumb(img: np.ndarray, box: tuple | None, scale: float, width: int) -> bytes:
    """Draw box (original pixels; scale = proxy width / original width) and resize to width; JPEG bytes."""
    if box is not None:
        x1, y1, x2, y2 = (int(round(c * scale)) for c in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 5)          # dark outline, then a bright box: visible
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 210, 255), 3)      # on light and dark footage (IR included)
    h = int(round(img.shape[0] * width / img.shape[1]))
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA))
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def clean(v):
    """JSON-safe value (NumPy scalars -> Python; NaN / None -> None)."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


class Rewind:
    """Everything loaded once; search() and coverage() are what the endpoints call."""

    def __init__(self, config_path: Path, db_path: Path):
        self.cfg = yaml.safe_load(config_path.read_text())
        self.scfg = SearchConfig.from_config(self.cfg)
        self.branches = active_branches(self.cfg)
        self.weights = check_weights(self.cfg["search"]["weights"], self.branches)
        api = self.cfg["api"]
        self.default_clips, self.max_clips = api["default_clips"], api["max_clips"]
        conn = connect(db_path)
        self.videos = {v.video_id: v for v in load_videos(conn)}
        self.durations = {vid: v.n_frames / v.fps for vid, v in self.videos.items()}
        self.proxies = {r[0]: r[1] for r in conn.execute("SELECT video_id, path FROM proxies")}
        self.frame_w = dict(conn.execute("SELECT video_id, width FROM videos").fetchall())
        conn.close()
        self.ui = self.cfg["ui"]
        self.join_s = self.cfg["ingest"]["max_name_duration_diff_s"]
        seg_s = self.cfg["baselines"]["segment_s"]
        self.motion = motion_scores(list(self.videos.values()), seg_s, REPO_ROOT / "artifacts")
        self.seg_video = {seg: v.video_id for v in self.videos.values() for seg in segments([v], seg_s)}
        self.thumb = lru_cache(maxsize=self.ui["thumb_cache"])(self._thumb)
        self.cameras = sorted({v.camera_id for v in self.videos.values()})
        self.modality = {v.camera_id: v.modality for v in self.videos.values()}
        missing = sorted(set(self.cameras) - set(self.ui["cameras"]))
        if missing:                                         # checked at start-up, not per request
            raise ValueError(f"config ui.cameras has no place label for cameras {missing}")
        t = time.perf_counter()
        self.engine = Engine(EngineConfig.from_config(self.cfg), {b: branch_model(self.cfg, b) for b in self.branches},
                             self.branches)
        a = self.cfg["appearance"]
        self.encoders = {"appearance": Clip(a["model"], a["pretrained"])}
        if "action" in self.branches:
            self.encoders["action"] = LanguageBind(REPO_ROOT / self.cfg["action"]["languagebind"]["weights_dir"])
        self.gpu = threading.Lock()
        self.load_s = time.perf_counter() - t

    def selection(self, cameras: str | None, start: str | None, end: str | None):
        cams = self.cameras if not cameras else [c.strip() for c in cameras.split(",") if c.strip()]
        unknown = sorted(set(cams) - set(self.cameras))
        if unknown:
            raise HTTPException(400, f"unknown cameras {unknown}; known: {self.cameras}")
        t0, t1 = parse_time(start, "start"), parse_time(end, "end")
        if (t0 is None) != (t1 is None):
            raise HTTPException(400, "give both start and end, or neither")
        if t0 is not None and t1 <= t0:
            raise HTTPException(400, "end must be after start")
        return cams, t0, t1

    def coverage(self, cams: list[str], t0: datetime | None, t1: datetime | None) -> dict:
        per = {c: 0.0 for c in cams}
        for v in self.videos.values():
            if v.camera_id in per:
                per[v.camera_id] += overlap_s(v.start_time, v.end_time, t0, t1)
        hours = sum(per.values()) / 3600
        with_footage = [c for c, s in per.items() if s > 0]
        return {"hours": round(hours, 2), "cameras": len(with_footage),
                "per_camera_hours": {c: round(s / 3600, 2) for c, s in per.items()},
                "branches": [b for b in self.branches if self.weights[b] > 0],
                "line": f"Searched {hours:.1f} h of footage across {len(with_footage)} camera"
                        f"{'s' if len(with_footage) != 1 else ''}."}

    def footage(self) -> dict:
        spans = merged_spans(self.videos.values(), self.join_s)
        cams = [{"camera": c, "modality": self.modality[c], **self.ui["cameras"][c],
                 "hours": round(sum((b - a).total_seconds() for a, b in spans[c]) / 3600, 2),
                 "spans": [[a.isoformat(), b.isoformat()] for a, b in spans[c]]} for c in self.cameras]
        return {"cameras": cams, "start": min(v.start_time for v in self.videos.values()).isoformat(),
                "end": max(v.end_time for v in self.videos.values()).isoformat(),
                "ui": {**{k: self.ui[k] for k in ("window_minutes", "default_window_minutes")},
                       "default_clips": self.default_clips, "max_clips": self.max_clips},
                "branches": [b for b in self.branches if self.weights[b] > 0], "note": NOTE}

    def motion_list(self, cams: list[str], t0: datetime | None, t1: datetime | None, k: int) -> dict:
        """Fallback review list: the selection's segments by mean motion activity (step 3.2 baseline order)."""
        sel = {s: a for s, a in self.motion.items()
               if s.camera_id in cams and (t0 is None or (s.start < t1 and t0 < s.end))}
        out = []
        for s in motion_ranked(sel)[:k]:
            v = self.videos[self.seg_video[s]]
            out.append({"camera": s.camera_id, "modality": v.modality, "video_id": v.video_id,
                        "proxy": self.proxies.get(v.video_id),
                        "start_s": round((s.start - v.start_time).total_seconds(), 2),
                        "end_s": round((s.end - v.start_time).total_seconds(), 2),
                        "wall_start": s.start.isoformat(), "wall_end": s.end.isoformat(),
                        "activity": round(float(sel[s]), 5)})
        return {"segments": out, "of": len(sel), "coverage": self.coverage(cams, t0, t1),
                "note": "Segments with the most movement first (the motion-search baseline): a manual review aid, "
                        "not a match to the description."}

    def _thumb(self, video_id: int, t: float, box: tuple | None) -> bytes:
        img = frame_at(REPO_ROOT / self.proxies[video_id], t)
        return draw_thumb(img, box, img.shape[1] / self.frame_w[video_id], self.ui["thumb_width"])

    def search(self, q: str, cams: list[str], t0: datetime | None, t1: datetime | None, k: int) -> dict:
        t_start = time.perf_counter()
        text = " ".join(q.split())
        if not text:
            raise HTTPException(400, "empty query")
        live = [b for b in self.branches if self.weights[b] > 0]
        with self.gpu:
            z = {b: self.encoders[b].encode_text([text])[0] for b in live}
        t_enc = time.perf_counter()
        w0, w1 = (None, None) if t0 is None else (wall_seconds(t0), wall_seconds(t1))
        hits = {b: self.engine.search(b, z[b], cameras=cams, t0=w0, t1=w1) for b in live}
        t_search = time.perf_counter()
        clips = fuse(hits, self.weights, self.durations, self.scfg, top_n=k)
        out = []
        for rank, c in enumerate(clips, 1):
            v = self.videos[c.video_id]
            out.append({
                "rank": rank, "camera": v.camera_id, "modality": v.modality, "video_id": c.video_id, "clip": v.clip,
                "proxy": self.proxies.get(c.video_id), "start_s": round(c.start_s, 2), "end_s": round(c.end_s, 2),
                "wall_start": (v.start_time + timedelta(seconds=c.start_s)).isoformat(),
                "wall_end": (v.start_time + timedelta(seconds=c.end_s)).isoformat(),
                "score": round(float(c.score), 4), "matched_by": c.branches,
                "best": {b: {f: clean(h.get(f)) for f in HIT_FIELDS[b]} for b, h in c.best.items()}})
        t_end = time.perf_counter()
        return {"query": text, "note": NOTE, "coverage": self.coverage(cams, t0, t1), "clips": out,
                "methods": {b: sorted(set(h["method"])) if len(h) else [] for b, h in hits.items()},
                "timing_ms": {"encode": round((t_enc - t_start) * 1000, 2),
                              "search": round((t_search - t_enc) * 1000, 2),
                              "fuse": round((t_end - t_search) * 1000, 2), "total": round((t_end - t_start) * 1000, 2)}}


def create_app(config_path: Path = REPO_ROOT / "configs" / "config.yaml",
               db_path: Path = REPO_ROOT / "data" / "rewind.sqlite", rw: "Rewind | None" = None) -> FastAPI:
    """rw: an already built Rewind (tests pass one without models); default: built from config_path and db_path."""
    rw = Rewind(config_path, db_path) if rw is None else rw
    app = FastAPI(title="Rewind", description="Describe what you saw, get the clip.")
    app.state.rewind = rw

    @app.get("/search")
    def search(q: str, cameras: str | None = None, start: str | None = None, end: str | None = None,
               k: int | None = Query(None, ge=1)):
        k = rw.default_clips if k is None else k
        if k > rw.max_clips:
            raise HTTPException(400, f"k must be <= {rw.max_clips}")
        cams, t0, t1 = rw.selection(cameras, start, end)
        return rw.search(q, cams, t0, t1, k)

    @app.get("/coverage")
    def coverage(cameras: str | None = None, start: str | None = None, end: str | None = None):
        return rw.coverage(*rw.selection(cameras, start, end))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(UI_DIR / "index.html", media_type="text/html")

    @app.get("/footage")
    def footage():
        return rw.footage()

    @app.get("/motion")
    def motion(cameras: str | None = None, start: str | None = None, end: str | None = None,
               k: int | None = Query(None, ge=1)):
        k = rw.ui["fallback_motion_k"] if k is None else k
        if k > rw.max_clips:
            raise HTTPException(400, f"k must be <= {rw.max_clips}")
        return rw.motion_list(*rw.selection(cameras, start, end), k)

    def video(video_id: int):
        v = rw.videos.get(video_id)
        if v is None or video_id not in rw.proxies:
            raise HTTPException(404, f"no video / proxy {video_id}")
        return v

    @app.get("/thumb")
    def thumb(video_id: int, t: float, box: str | None = None):
        v = video(video_id)
        if not 0 <= t <= v.n_frames / v.fps:
            raise HTTPException(400, f"t must be within 0..{v.n_frames / v.fps:.2f} s")
        b = None
        if box:
            try:
                b = tuple(float(x) for x in box.split(","))
            except ValueError:
                raise HTTPException(400, "box: four numbers x1,y1,x2,y2")
            if len(b) != 4:
                raise HTTPException(400, "box: four numbers x1,y1,x2,y2")
        return Response(rw.thumb(video_id, round(t, 1), b), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    @app.get("/proxy/{video_id}")
    def proxy(video_id: int):
        video(video_id)
        return FileResponse(REPO_ROOT / rw.proxies[video_id], media_type="video/mp4")

    return app


def __getattr__(name: str):
    """`uvicorn rewind.api.app:app` builds the app on first access (importing this module stays cheap)."""
    if name == "app":
        global app
        app = create_app(Path(os.environ.get("REWIND_CONFIG", REPO_ROOT / "configs" / "config.yaml")))
        return app
    raise AttributeError(name)
