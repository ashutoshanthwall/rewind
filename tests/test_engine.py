"""C++ engine through Python (rewind/search/engine.py, scripts/build_indexes.py): conversions, time filter and camera
merge against NumPy brute force, save/load, concurrent searches. Needs the built module (engine/build/python)."""

import threading
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from rewind.ingest.manifest import Video
from rewind.search.engine import (Engine, EngineConfig, engine_module, to_unit_float32, wall_seconds)
from scripts.build_indexes import build_one, with_wall_seconds

STAMP = {"git_hash": "test", "config_hash": "test"}


def ecfg(tmp_path, k=20, oversample=4, min_fraction=1.0):
    return EngineConfig(tmp_path, k, 200, oversample, min_fraction, 1e-5, 16, 100, 42)


def unit(n, d, seed):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_wall_seconds_hand_computed():
    assert wall_seconds(datetime(1970, 1, 2, 0, 0, 1)) == 86401.0
    # 2018-03-07 10:55:06 = 17,597 days after 1970-01-01 (48 years incl. 12 leap days) + 39,306 s
    assert wall_seconds(datetime(2018, 3, 7, 10, 55, 6)) == 17597 * 86400 + 10 * 3600 + 55 * 60 + 6
    with pytest.raises(ValueError):
        wall_seconds(datetime.fromisoformat("2018-03-07T10:55:06+00:00"))


def test_with_wall_seconds_adds_t_s_to_video_start():
    v = Video(7, "c", "p", "G330", "rgb", datetime(2018, 3, 7, 10, 0, 0), 9000, 30.0)   # 300 s
    m = pd.DataFrame({"video_id": [7, 7], "t_s": [0.0, 299.5]})
    out = with_wall_seconds(m, [v])
    base = wall_seconds(datetime(2018, 3, 7, 10, 0, 0))
    assert out["wall_s"].tolist() == [base, base + 299.5]
    with pytest.raises(ValueError, match="outside its video"):
        with_wall_seconds(pd.DataFrame({"video_id": [7], "t_s": [301.0]}), [v])


def test_to_unit_float32_renormalises_fp16():
    x = unit(50, 512, 1).astype(np.float16)
    assert np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1).max() > 1e-5   # fp16 rounding: engine rejects
    y = to_unit_float32(x)
    assert y.dtype == np.float32 and y.flags.c_contiguous
    assert np.abs(np.linalg.norm(y.astype(np.float64), axis=1) - 1).max() < 1e-6
    with pytest.raises(ValueError, match="zero"):
        to_unit_float32(np.zeros((2, 4), np.float16))
    with pytest.raises(ValueError, match="NaN"):
        to_unit_float32(np.array([[np.nan, 1.0]], np.float32))


def make_engine(tmp_path, cams, n=1500, d=32, **kw):
    """Two-camera toy branch 'appearance': row i of camera c has time 1000 + i (one per second)."""
    cfg = ecfg(tmp_path, **kw)
    mod = engine_module()
    data = {}
    for j, cam in enumerate(cams):
        x = unit(n, d, 10 + j)
        meta = pd.DataFrame({"camera_id": cam, "video_id": j, "t_s": np.arange(n, dtype=float),
                             "wall_s": 1000.0 + np.arange(n)})
        build_one(mod, cfg, "appearance", cam, "toy", x, meta, STAMP)
        data[cam] = (x, meta)
    return Engine(cfg, {"appearance": "toy"}, branches=("appearance",)), data


def brute(data, cams, q, k, t0=None, t1=None):
    """NumPy reference: [(camera, row)] of the top k over the cameras / range (ties: camera, row)."""
    rows = []
    for cam in cams:
        x, meta = data[cam]
        s = x.astype(np.float64) @ q.astype(np.float64)
        t = meta["wall_s"].to_numpy()
        for i in range(len(x)):
            if t0 is None or t0 <= t[i] <= t1:
                rows.append((-s[i], cam, i))
    return [(c, i) for _, c, i in sorted(rows)[:k]]


@pytest.mark.parametrize("cams,t0,t1", [(None, None, None), (["G1"], None, None), (None, 1100.0, 1300.0),
                                        (["G2"], 1000.0, 1050.0), (None, 5000.0, 6000.0)])
def test_engine_matches_brute_force(tmp_path, cams, t0, t1):
    eng, data = make_engine(tmp_path, ["G1", "G2"])
    sel = cams or ["G1", "G2"]
    qs = unit(20, 32, 99)
    recall = []
    for q in qs:
        want = brute(data, sel, q, 20, t0, t1)
        ex = eng.search("appearance", q, cameras=cams, t0=t0, t1=t1, exact=True)
        assert list(zip(ex.camera_id, ex.row)) == want                         # exact path: identical, same order
        got = eng.search("appearance", q, cameras=cams, t0=t0, t1=t1)
        assert set(got.camera_id) <= set(sel)
        if t0 is not None:
            assert got.wall_s.between(t0, t1).all()
        assert len(got) == len(want)
        assert (np.diff(got.score.to_numpy()) <= 0).all()                     # merged best first
        hit = set(zip(got.camera_id, got.row)) & set(want)
        recall.append(len(hit) / len(want) if want else float(got.empty))         # empty range: empty is right
    assert np.mean(recall) >= 0.95


def test_engine_reload_and_meta_mapping(tmp_path):
    eng, data = make_engine(tmp_path, ["G1"])
    q = unit(1, 32, 5)[0]
    a = eng.search("appearance", q)
    eng2 = Engine(eng.cfg, {"appearance": "toy"}, branches=("appearance",))          # loads from disk again
    b = eng2.search("appearance", q)
    assert a[["row", "score"]].equals(b[["row", "score"]])
    x, _ = data["G1"]
    assert np.allclose(a.score, x[a.row] @ q, atol=1e-5)                             # row id -> the right vector
    with pytest.raises(Exception, match="stale|model"):
        Engine(eng.cfg, {"appearance": "another-model"}, branches=("appearance",))


def test_concurrent_searches_equal_serial(tmp_path):
    eng, _ = make_engine(tmp_path, ["G1", "G2"], n=3000)
    qs = unit(30, 32, 7)
    serial = [eng.search("appearance", q)[["camera_id", "row"]].values.tolist() for q in qs]
    out, errors = {}, []

    def work(w):
        try:
            for rep in range(5):
                for i, q in enumerate(qs):
                    r = eng.search("appearance", q, t0=None if w % 2 else 1000.0, t1=None if w % 2 else 9000.0)
                    out[(w, rep, i)] = r[["camera_id", "row"]].values.tolist()
        except Exception as e:      # collected and re-raised below: a thread's error must fail the test
            errors.append(e)

    ts = [threading.Thread(target=work, args=(w,)) for w in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    # odd workers: no range = the serial answer; even workers: a range holding every row (filtered HNSW path)
    assert all(v == serial[i] for (w, _, i), v in out.items() if w % 2)
    filt = [eng.search("appearance", q, t0=1000.0, t1=9000.0)[["camera_id", "row"]].values.tolist() for q in qs]
    assert all(v == filt[i] for (w, _, i), v in out.items() if not w % 2)
