"""The C++ HNSW engine from Python (steps 7.1-7.3): per-camera, per-branch named indexes on disk and in memory.

Layout (search.index_dir, one directory per index, name = "<branch>/<camera>"):
  <index_dir>/<branch>/<camera>/vectors.f32     float32, L2-normalised in float32, row-major (engine store)
                               /manifest.json   model, dim, rows, l2_normalized, dtype, sha256 (checked on load)
                               /graph.hnsw      HNSW graph (engine format v2; tied to the vectors by SHA-256)
                               /meta.parquet    row -> metadata (row = engine id; vector_id, video_id, t_s, wall_s, ...)
Ids inside the engine are row indices 0..N-1 of one index; the id -> metadata map lives here, on the Python side
(decision of 26 Oct, PROGRESS.md step 7.1). The vectors on disk are fp16 (steps 4.2 / 5.2): fp16 rounding leaves
|norm - 1| ~ 1e-3, above the engine's search.norm_tolerance, so they are converted to float32 and re-normalised.

Time filter: every row has wall_s = wall-clock seconds (video start_time + t_s, naive local time counted from
1970-01-01, see wall_seconds). A row is in [t0, t1] if t0 <= wall_s <= t1 (the engine's rule, collection.h).
For a window (action branch) t_s is the window centre.

Search over a branch (Engine.search): each selected camera's index returns its top k (filtered by time if asked),
then the cameras are merged by score and the best k overall are kept (ties: camera id, then row). With exact
per-camera searches this equals one brute-force top k over the selected footage, so the semantics of
search.k_per_index match steps 4.4 / 5.1 (top k vectors of the branch).
"""

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from rewind.paths import REPO_ROOT

ENGINE_PYTHON_DIR = REPO_ROOT / "engine" / "build" / "python"
BUILD_HINT = ("cmake -S engine -B engine/build -DREWIND_PYTHON=ON -Dpybind11_DIR=$(uv run python -m pybind11 "
              "--cmakedir) -DPython_EXECUTABLE=$(uv run python -c 'import sys; print(sys.executable)') && "
              "cmake --build engine/build -j8")
EPOCH = datetime(1970, 1, 1)
BRANCHES = ("appearance", "action")


def engine_module():
    """The compiled rewind_engine module from engine/build/python (fails loudly with the build command)."""
    if str(ENGINE_PYTHON_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_PYTHON_DIR))
    try:
        return importlib.import_module("rewind_engine")
    except ImportError as e:
        raise ImportError(f"rewind_engine is not built ({e}). Build it from the repo root:\n  {BUILD_HINT}") from e


def wall_seconds(dt: datetime) -> float:
    """Wall-clock seconds of a naive manifest datetime, counted from 1970-01-01 in the same (camera) clock."""
    if dt.tzinfo is not None:
        raise ValueError(f"expected a naive manifest datetime, got {dt!r}")
    return (dt - EPOCH).total_seconds()


def to_unit_float32(x: np.ndarray) -> np.ndarray:
    """float32 copy of x with every row re-normalised in float32 (fp16 storage leaves |norm - 1| ~ 1e-3)."""
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError(f"expected a 2-d array, got shape {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("vectors contain NaN or Inf")
    n = np.linalg.norm(y, axis=1, keepdims=True)
    if (n == 0).any():
        raise ValueError(f"{int((n == 0).sum())} zero vectors cannot be normalised")
    return np.ascontiguousarray(y / n, dtype=np.float32)


def branch_model(cfg: dict, branch: str) -> str:
    """Model name recorded in an index's manifest and graph: the engine refuses an index built for another one.
    Appearance includes the search variant (frame_mode / crop_source / centering), which changes the vectors.
    Per-camera centering keeps its step 7.1 name ("centered"); per camera and type is "centered_camera_type"."""
    if branch == "appearance":
        a = cfg["appearance"]
        center = "raw" if not a["mean_center"] else (
            "centered" if a["center_groups"] == "camera" else f"centered_{a['center_groups']}")
        return f"{a['model']}/{a['pretrained']}:{a['frame_mode']}/{a['crop_source']}/{center}"
    if branch == "action":
        return cfg["action"]["model"]
    raise ValueError(f"unknown branch {branch!r}; expected one of {BRANCHES}")


def active_branches(cfg: dict) -> tuple[str, ...]:
    """appearance always; action if action.enabled. (events: dropped at Gate 6.2, not an engine index.)"""
    act = cfg["action"]["enabled"]
    if act not in (True, False):
        raise ValueError(f"action.enabled must be true or false, got {act!r}")
    return ("appearance", "action") if act else ("appearance",)


def index_name(branch: str, camera: str) -> str:
    return f"{branch}/{camera}"


def index_path(root: Path, branch: str, camera: str) -> Path:
    return root / branch / camera


def camera_ranges(videos: list, join_gap_s: float) -> dict[str, list[tuple[float, float]]]:
    """camera -> sorted disjoint wall-clock ranges covering these videos (e.g. one split's). Videos of a camera that
    follow each other within join_gap_s become one range; a range ends 1 us before its last video's end (end_time is
    exclusive, and a next video of another split may start exactly there)."""
    out: dict[str, list[list[float]]] = {}
    for v in sorted(videos, key=lambda v: (v.camera_id, v.start_time)):
        a, b = wall_seconds(v.start_time), wall_seconds(v.end_time) - 1e-6
        rs = out.setdefault(v.camera_id, [])
        if rs and a <= rs[-1][1] + join_gap_s:
            rs[-1][1] = max(rs[-1][1], b)
        else:
            rs.append([a, b])
    return {c: [(a, b) for a, b in rs] for c, rs in out.items()}


def write_store(d: Path, x: np.ndarray, model: str) -> str:
    """Writes vectors.f32 + manifest.json (engine format) for float32 unit rows; returns the SHA-256."""
    if x.dtype != np.float32 or not x.flags.c_contiguous:
        raise ValueError("write_store needs a C-contiguous float32 array (use to_unit_float32)")
    d.mkdir(parents=True, exist_ok=True)
    data = x.tobytes()
    sha = hashlib.sha256(data).hexdigest()
    (d / "vectors.f32").write_bytes(data)
    (d / "manifest.json").write_text(json.dumps({"model": model, "dim": int(x.shape[1]), "rows": int(x.shape[0]),
                                                 "l2_normalized": True, "dtype": "float32", "sha256": sha}, indent=1))
    return sha


@dataclass(frozen=True)
class EngineConfig:
    index_dir: Path
    k: int
    ef_search: int
    oversample: int
    min_fraction: float
    norm_tolerance: float
    M: int
    ef_construction: int
    seed: int

    @classmethod
    def from_config(cls, cfg: dict) -> "EngineConfig":
        s = cfg["search"]
        if s["ef_search"] < s["k_per_index"]:
            raise ValueError(f"search.ef_search ({s['ef_search']}) must be >= search.k_per_index ({s['k_per_index']})")
        return cls(REPO_ROOT / s["index_dir"], s["k_per_index"], s["ef_search"], s["filter_oversample"],
                   s["filter_min_fraction"], s["norm_tolerance"], s["hnsw"]["M"], s["hnsw"]["ef_construction"],
                   s["hnsw"]["seed"])


class Engine:
    """Loads every per-camera index of the given branches once; search() is safe to call from several threads
    (each engine index has its own lock; the Python state here is read-only after __init__)."""

    def __init__(self, ecfg: EngineConfig, models: dict[str, str], branches=BRANCHES):
        """models: branch -> model name recorded in the manifests (stale indexes are refused by the engine)."""
        self.cfg = ecfg
        self.mod = engine_module()
        self.collection = self.mod.IndexCollection()
        self.meta: dict[str, pd.DataFrame] = {}
        self.cameras: dict[str, list[str]] = {}
        for branch in branches:
            d = ecfg.index_dir / branch
            cams = sorted(p.name for p in d.iterdir() if (p / "graph.hnsw").exists()) if d.exists() else []
            if not cams:
                raise FileNotFoundError(f"no {branch} indexes under {d} (run scripts.build_indexes)")
            self.cameras[branch] = cams
            for cam in cams:
                p = index_path(ecfg.index_dir, branch, cam)
                man = json.loads((p / "manifest.json").read_text())
                idx = self.collection.add(index_name(branch, cam), models[branch])
                idx.load_vectors(str(p / "vectors.f32"), str(p / "manifest.json"), man["dim"], ecfg.norm_tolerance)
                meta = pd.read_parquet(p / "meta.parquet")
                if len(meta) != idx.size or (meta["row"].to_numpy() != np.arange(len(meta))).any():
                    raise ValueError(f"{p}: meta.parquet rows do not match the {idx.size} engine rows")
                idx.set_times(meta["wall_s"].to_numpy(np.float64))
                idx.load(str(p / "graph.hnsw"))
                self.meta[index_name(branch, cam)] = meta

    def dim(self, branch: str) -> int:
        return self.collection.get(index_name(branch, self.cameras[branch][0])).dim

    def search(self, branch: str, q: np.ndarray, k: int | None = None, cameras=None, t0: float | None = None,
               t1: float | None = None, exact: bool = False, ranges: dict | None = None) -> pd.DataFrame:
        """Top k rows of a branch over the selected cameras (default: all) and wall-clock range [t0, t1] (both or
        neither). ranges: instead of cameras / t0 / t1, camera -> disjoint [(t0, t1), ...] (e.g. one split's footage,
        camera_ranges()): the top k over the union. Returns the metadata rows + score + index + method, best first."""
        k = self.cfg.k if k is None else k
        if ranges is not None:
            if cameras is not None or t0 is not None or t1 is not None:
                raise ValueError("give ranges, or cameras / t0 / t1, not both")
            for cam, rs in ranges.items():
                for (a0, a1), (b0, b1) in zip(rs, rs[1:]):
                    if not (a0 <= a1 < b0 <= b1):
                        raise ValueError(f"{cam}: ranges must be sorted and disjoint, got {(a0, a1)} then {(b0, b1)}")
            jobs = [(cam, r0, r1) for cam, rs in ranges.items() for r0, r1 in rs]
        else:
            jobs = [(cam, t0, t1) for cam in (self.cameras[branch] if cameras is None else list(cameras))]
        unknown = {cam for cam, _, _ in jobs} - set(self.cameras[branch])
        if unknown:
            raise ValueError(f"{branch}: no index for cameras {sorted(unknown)}")
        q = np.ascontiguousarray(q, dtype=np.float32)
        parts = []
        for cam, t0, t1 in jobs:
            name = index_name(branch, cam)
            idx = self.collection.get(name)
            if exact:
                ids, scores, rep = idx.exact(q, k, t0, t1)
            else:
                ids, scores, rep = idx.search(q, k, self.cfg.ef_search, t0, t1, self.cfg.oversample,
                                              self.cfg.min_fraction)
            if len(ids):
                part = self.meta[name].iloc[ids].copy()
                part["score"] = scores.astype(np.float64)
                part["index"] = name
                part["method"] = rep["method"]
                parts.append(part)
        if not parts:
            cols = self.meta[index_name(branch, self.cameras[branch][0])].columns
            return pd.DataFrame(columns=[*cols, "score", "index", "method"])
        out = pd.concat(parts, ignore_index=True)
        out = out.sort_values(["score", "camera_id", "row"], ascending=[False, True, True], kind="mergesort")
        return out.head(k).reset_index(drop=True)
