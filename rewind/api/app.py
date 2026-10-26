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

Threads: FastAPI runs these sync endpoints in a thread pool. Engine indexes have their own locks (C++); the GPU text
encoders are used under one lock here.

Run from the repo root:  uv run uvicorn rewind.api.app:app --port 8000
"""

import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Query

from rewind.embed.clip_model import Clip
from rewind.embed.languagebind import LanguageBind
from rewind.ingest.manifest import connect, load_videos
from rewind.paths import REPO_ROOT
from rewind.search.core import SearchConfig
from rewind.search.engine import Engine, EngineConfig, active_branches, branch_model, wall_seconds
from rewind.search.fusion import check_weights, fuse

HIT_FIELDS = {"appearance": ["t_s", "score", "z", "type", "view", "crop_source", "class", "x1", "y1", "x2", "y2"],
              "action": ["t_s", "window_start", "score", "z"]}
NOTE = "Candidate clips ranked by similarity to the description; people judge them. Similarity is not proof."


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
        conn.close()
        self.cameras = sorted({v.camera_id for v in self.videos.values()})
        self.modality = {v.camera_id: v.modality for v in self.videos.values()}
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
               db_path: Path = REPO_ROOT / "data" / "rewind.sqlite") -> FastAPI:
    rw = Rewind(config_path, db_path)
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

    return app


def __getattr__(name: str):
    """`uvicorn rewind.api.app:app` builds the app on first access (importing this module stays cheap)."""
    if name == "app":
        global app
        app = create_app(Path(os.environ.get("REWIND_CONFIG", REPO_ROOT / "configs" / "config.yaml")))
        return app
    raise AttributeError(name)
