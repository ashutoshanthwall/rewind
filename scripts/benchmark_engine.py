"""Step 7.2: scaling benchmark of the C++ HNSW engine vs exact brute force (recall@k and latency).

Parts (settings in config `benchmark`, k = search.k_per_index; every latency is single-threaded, one query at a time,
`warmup` queries thrown away, then `repetitions` passes over all queries, a query's latency = median of its passes):
  --real       the 12 built per-camera indexes (scripts/build_indexes.py) and, for comparison, one single index per
               branch over all cameras (built here, kept in memory only). Queries: text vectors of templated
               activity prompts ("a " + activity name with spaces, every activity type) and of every tuning + dev
               query (never test), with the branch's own text encoder (CLIP for appearance, LanguageBind for action).
               Per ef in ef_sweep: per index, per branch (6 camera indexes searched and merged, as the API does) and
               single all-camera index: recall@k vs the engine's exact search, median / p95 latency. Exact baselines:
               the engine's brute force (scalar loop, as upstream) and NumPy (float32 matmul + argpartition, what
               steps 4.4 / 5.1 used). Plus the time filter on real data: filter_windows_per_camera windows of
               filter_window_s per camera, each starting at the time of a random indexed row (so never empty) ->
               recall vs exact filtered search and the method used.
  --synthetic N  N random unit 512-d vectors (Gaussian, seeded; written to a float32 memmap under artifacts/bench/,
               removed afterwards), random unit queries. A hard case for graph search (no structure): its recall says
               little about real data (not a proven bound); its build time, memory and exact cost do. Build time, recall@k and latency per ef, engine and NumPy brute force.
  --report     results/reports/scaling.md + results/figures/scaling_7.2.png from the JSON results of the parts.
Results: results/reports/scaling_7.2_real.json, scaling_7.2_synthetic_<N>.json.

Run from the repo root:
  uv run python -m scripts.benchmark_engine --real
  nohup uv run python -m scripts.benchmark_engine --synthetic 1000000 2000000 > results/logs/bench_synthetic.log 2>&1 &
  uv run python -m scripts.benchmark_engine --report
"""

import argparse
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from rewind.experiment_log import log_experiment, run_stamp
from rewind.paths import REPO_ROOT, display_path
from rewind.search.engine import Engine, EngineConfig, active_branches, branch_model, engine_module, index_name

REPORTS = REPO_ROOT / "results" / "reports"


def timed(fn, queries: np.ndarray, warmup: int, reps: int) -> tuple[np.ndarray, list]:
    """Per-query latency in ms (median of reps; repetition-outer order) and the results of the last pass."""
    for q in queries[:warmup]:
        fn(q)
    ms = np.zeros((reps, len(queries)))
    out = [None] * len(queries)
    for r in range(reps):
        for i, q in enumerate(queries):
            t = time.perf_counter()
            out[i] = fn(q)
            ms[r, i] = (time.perf_counter() - t) * 1000
    return np.median(ms, axis=0), out


def lat(ms: np.ndarray) -> dict:
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)), "mean_ms": float(ms.mean())}


def recall(approx: list, exact: list, k: int) -> float:
    """Mean over queries of |approx top k ∩ exact top k| / min(k, |exact|)."""
    if any(len(e) == 0 for e in exact):
        raise ValueError("an exact result list is empty: recall is undefined")
    return float(np.mean([len(set(a[:k]) & set(e[:k])) / min(k, len(e)) for a, e in zip(approx, exact)]))


def numpy_topk(x: np.ndarray, k: int):
    def run(q):
        s = x @ q
        idx = np.argpartition(-s, k - 1)[:k]
        return idx[np.argsort(-s[idx], kind="stable")].tolist()
    return run


def system_info() -> dict:
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    mem = next(int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.startswith("MemTotal"))
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "ram_bytes": mem, "kernel": platform.release(),
            "numpy_threads": os.environ.get("OMP_NUM_THREADS", "default"), "load_avg": os.getloadavg()}


# ---------------------------------------------------------------------------------------------------- real sizes

def query_texts(conn) -> list[str]:
    from rewind.eval.queries import load_queries
    from scripts.run_baselines import QUERIES_DIR
    types = sorted({r[0] for r in conn.execute("SELECT DISTINCT activity_type FROM events")})
    texts = ["a " + t.replace("_", " ") for t in types]
    for split in ("tuning", "dev"):
        texts += [q["query"] for q in load_queries(QUERIES_DIR, split)]
    return texts


def encode(cfg: dict, branch: str, texts: list[str]) -> np.ndarray:
    import torch
    if branch == "appearance":
        from rewind.embed.clip_model import Clip
        enc = Clip(cfg["appearance"]["model"], cfg["appearance"]["pretrained"])
    else:
        from rewind.embed.languagebind import LanguageBind
        enc = LanguageBind(REPO_ROOT / cfg["action"]["languagebind"]["weights_dir"])
    z = enc.encode_text(texts)
    del enc
    torch.cuda.empty_cache()
    return np.ascontiguousarray(z, dtype=np.float32)


def real(cfg: dict, args) -> dict:
    from rewind.ingest.manifest import connect
    b = cfg["benchmark"]
    ecfg = EngineConfig.from_config(cfg)
    k, efs, W, R = ecfg.k, b["ef_sweep"], b["warmup"], b["repetitions"]
    branches = active_branches(cfg)
    texts = query_texts(connect(args.db))
    Q = {br: encode(cfg, br, texts) for br in branches}
    eng = Engine(ecfg, {br: branch_model(cfg, br) for br in branches}, branches)
    mod = engine_module()
    rng = np.random.default_rng(b["seed"])
    out = {"k": k, "n_queries": len(texts), "ef_sweep": efs, "system": system_info(), "branches": {}}
    for br in branches:
        q = Q[br]
        res = {"per_index": {}, "branch": {}, "single": {}, "filter": {}}
        exact_lists = {}
        for cam in eng.cameras[br]:
            idx = eng.collection.get(index_name(br, cam))
            ms_bf, ex = timed(lambda v: idx.exact(v, k)[0].tolist(), q, W, R)
            exact_lists[cam] = ex
            x = np.fromfile(ecfg.index_dir / br / cam / "vectors.f32", np.float32).reshape(idx.size, idx.dim)
            ms_np, ex_np = timed(numpy_topk(x, min(k, idx.size)), q, W, R)
            row = {"rows": idx.size, "dim": idx.dim, "exact_engine": lat(ms_bf), "exact_numpy": lat(ms_np),
                   "numpy_vs_engine_exact_recall": recall(ex_np, ex, k), "hnsw": {}}
            for ef in efs:
                ms, ap = timed(lambda v: idx.search(v, k, ef, None, None, ecfg.oversample, ecfg.min_fraction)[0]
                               .tolist(), q, W, R)
                row["hnsw"][ef] = {"recall": recall(ap, ex, k), **lat(ms)}
            res["per_index"][cam] = row
            print(f"{br}/{cam}: {idx.size:,} rows, exact {row['exact_engine']['median_ms']:.2f} ms (numpy "
                  f"{row['exact_numpy']['median_ms']:.2f}), " + ", ".join(
                      f"ef{ef} R {row['hnsw'][ef]['recall']:.3f} {row['hnsw'][ef]['median_ms']:.2f} ms" for ef in efs),
                  flush=True)

        # branch = 6 camera indexes searched and merged (Engine.search); exact = engine brute force per camera
        key = lambda df: list(zip(df.camera_id, df.row))
        ms_bf, ex = timed(lambda v: key(eng.search(br, v, k, exact=True)), q, W, R)
        res["branch"] = {"rows": sum(len(eng.meta[index_name(br, c)]) for c in eng.cameras[br]),
                         "exact_engine": lat(ms_bf), "hnsw": {}}
        ecfg0 = eng.cfg
        for ef in efs:
            eng.cfg = EngineConfig(**{**ecfg0.__dict__, "ef_search": ef})
            ms, ap = timed(lambda v: key(eng.search(br, v, k)), q, W, R)
            res["branch"]["hnsw"][ef] = {"recall": recall(ap, ex, k), **lat(ms)}
        eng.cfg = ecfg0
        allx = np.concatenate([np.fromfile(ecfg.index_dir / br / c / "vectors.f32", np.float32)
                               .reshape(-1, eng.dim(br)) for c in eng.cameras[br]])
        ms_np, _ = timed(numpy_topk(allx, k), q, W, R)
        res["branch"]["exact_numpy_all_rows"] = lat(ms_np)

        # single index over every camera (comparison only; same vectors, same HNSW settings)
        single = mod.NamedIndex(f"{br}/all", "bench")
        single.set_vectors(allx, ecfg.norm_tolerance)
        build_s = single.build(ecfg.M, ecfg.ef_construction, ecfg.ef_search, ecfg.seed)
        ms_bf, ex1 = timed(lambda v: single.exact(v, k)[0].tolist(), q, W, R)
        res["single"] = {"rows": single.size, "build_s": build_s, "exact_engine": lat(ms_bf),
                         "exact_numpy": lat(ms_np), "hnsw": {}}
        for ef in efs:
            ms, ap = timed(lambda v: single.search(v, k, ef, None, None, 1, 1.0)[0].tolist(), q, W, R)
            res["single"]["hnsw"][ef] = {"recall": recall(ap, ex1, k), **lat(ms)}
        del single, allx
        print(f"{br}: branch (merged) " + ", ".join(f"ef{ef} R {res['branch']['hnsw'][ef]['recall']:.3f} "
                                                   f"{res['branch']['hnsw'][ef]['median_ms']:.2f} ms" for ef in efs)
              + f"; exact {res['branch']['exact_engine']['median_ms']:.1f} ms, numpy {ms_np.mean():.1f} ms", flush=True)

        # time filter on real data, at the config's ef / oversample / min_fraction
        methods, recs, ms_f, ms_fx = {}, [], [], []
        for cam in eng.cameras[br]:
            idx = eng.collection.get(index_name(br, cam))
            t = eng.meta[index_name(br, cam)]["wall_s"].to_numpy()
            for _ in range(b["filter_windows_per_camera"]):
                t0 = float(t[rng.integers(len(t))])     # a window starting at an indexed row: never empty
                t1 = t0 + b["filter_window_s"]
                ms, ap = timed(lambda v: idx.search(v, k, ecfg.ef_search, t0, t1, ecfg.oversample,
                                                    ecfg.min_fraction), q, 0, 1)
                msx, exf = timed(lambda v: idx.exact(v, k, t0, t1)[0].tolist(), q, 0, 1)
                for a in ap:
                    methods[a[2]["method"]] = methods.get(a[2]["method"], 0) + 1
                recs.append(recall([a[0].tolist() for a in ap], exf, k))
                ms_f += ms.tolist()
                ms_fx += msx.tolist()
        res["filter"] = {"window_s": b["filter_window_s"], "windows": len(recs), "searches": int(sum(methods.values())),
                         "recall": float(np.mean(recs)), "min_window_recall": float(np.min(recs)), "methods": methods,
                         "hnsw_filtered": lat(np.array(ms_f)), "exact_filtered": lat(np.array(ms_fx))}
        print(f"{br}: time filter {res['filter']}", flush=True)
        out["branches"][br] = res
    return out


# ---------------------------------------------------------------------------------------------------- synthetic

def synthetic(cfg: dict, n: int, art: Path) -> dict:
    b = cfg["benchmark"]
    ecfg = EngineConfig.from_config(cfg)
    k, efs, W, R, d = ecfg.k, b["ef_sweep"], b["warmup"], b["repetitions"], b["synthetic_dim"]
    path = art / "bench" / f"synthetic_{n}x{d}.f32"
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(b["seed"])
    x = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, d))
    step = 100_000
    for i in range(0, n, step):
        c = rng.standard_normal((min(step, n - i), d)).astype(np.float32)
        x[i:i + len(c)] = c / np.linalg.norm(c, axis=1, keepdims=True)
    x.flush()
    q = rng.standard_normal((b["synthetic_queries"], d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    mod = engine_module()
    idx = mod.NamedIndex(f"synthetic/{n}", "bench")
    t = time.perf_counter()
    idx.set_vectors(x, ecfg.norm_tolerance)
    load_s = time.perf_counter() - t
    print(f"synthetic {n:,} x {d}: vectors in engine ({load_s:.0f} s); building...", flush=True)
    build_s = idx.build(ecfg.M, ecfg.ef_construction, ecfg.ef_search, ecfg.seed)
    print(f"  build {build_s:.0f} s", flush=True)
    ms_bf, ex = timed(lambda v: idx.exact(v, k)[0].tolist(), q, W, R)
    ms_np, ex_np = timed(numpy_topk(x, k), q, W, R)
    out = {"rows": n, "dim": d, "k": k, "n_queries": len(q), "build_s": build_s, "system": system_info(),
           "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           "exact_engine": lat(ms_bf), "exact_numpy": lat(ms_np),
           "numpy_vs_engine_exact_recall": recall(ex_np, ex, k), "hnsw": {}}
    for ef in efs:
        ms, ap = timed(lambda v: idx.search(v, k, ef, None, None, 1, 1.0)[0].tolist(), q, W, R)
        out["hnsw"][ef] = {"recall": recall(ap, ex, k), **lat(ms)}
        print(f"  ef {ef}: recall {out['hnsw'][ef]['recall']:.3f}, {out['hnsw'][ef]['median_ms']:.2f} ms "
              f"(exact {out['exact_engine']['median_ms']:.0f} ms, numpy {out['exact_numpy']['median_ms']:.0f} ms)",
              flush=True)
    del idx, x
    path.unlink()            # our own temporary benchmark file (2-4 GB), never footage or a result
    return out


# ---------------------------------------------------------------------------------------------------- report

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]       # dataviz reference palette, categorical slots 1, 2, 3, 7


def hex_bgr(h: str) -> tuple:
    return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))


def plot(points: list[dict], path: Path, ef: int) -> None:
    """Two panels sharing the x axis (rows, log scale): latency (log ms) and recall@k at the config ef.
    points: {label, group, rows, hnsw_ms, exact_ms, numpy_ms, recall}. Lines join points of the same group only
    (real CLIP vectors vs random vectors are different data, not one trend)."""
    W, H, ml, mr, mt, gap, ph = 1100, 760, 90, 230, 50, 70, 290
    img = np.full((H, W, 3), (251, 252, 252), np.uint8)
    ink, mute, grid = (11, 11, 11), (78, 81, 82), (225, 226, 226)
    xs = [p["rows"] for p in points]
    lx0, lx1 = np.log10(min(xs)) - 0.15, np.log10(max(xs)) + 0.15
    X = lambda r: int(ml + (np.log10(r) - lx0) / (lx1 - lx0) * (W - ml - mr))
    lat_vals = [v for p in points for v in (p["hnsw_ms"], p["exact_ms"], p["numpy_ms"])]
    ly0, ly1 = np.floor(np.log10(min(lat_vals))), np.ceil(np.log10(max(lat_vals)))
    Y1 = lambda ms: int(mt + ph - (np.log10(ms) - ly0) / (ly1 - ly0) * ph)
    top2 = mt + ph + gap
    rmin = min(0.5, np.floor(min(p["recall"] for p in points) * 10) / 10)
    Y2 = lambda r: int(top2 + ph - (r - rmin) / (1 - rmin) * ph)
    put = lambda s, xy, c=ink, sc=0.5: cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, c, 1, cv2.LINE_AA)
    for e in range(int(ly0), int(ly1) + 1):
        y = Y1(10 ** e)
        cv2.line(img, (ml, y), (W - mr, y), grid, 1)
        put(f"{10 ** e:g} ms", (10, y + 5), mute)
    for r in np.arange(rmin, 1.0001, 0.1):
        y = Y2(r)
        cv2.line(img, (ml, y), (W - mr, y), grid, 1)
        put(f"{r:.1f}", (40, y + 5), mute)
    for e in range(int(np.ceil(lx0)), int(np.floor(lx1)) + 1):
        for yy in (mt + ph, top2 + ph):
            cv2.line(img, (X(10 ** e), yy), (X(10 ** e), yy + 6), mute, 1)
        put(f"{10 ** e:,}", (X(10 ** e) - 30, top2 + ph + 24), mute)
    put("Engine latency per query vs vectors in the index (log-log); k = 200, single thread", (ml, 30), ink, 0.6)
    put(f"Recall@200 of HNSW (ef {ef}) vs exact; lines join real indexes and random sets separately",
        (ml, top2 - 12), ink, 0.6)
    put("vectors in the index (log scale)", (ml + 250, H - 12), mute)
    series = [("hnsw_ms", "HNSW", SERIES[0]), ("exact_ms", "exact, engine (scalar)", SERIES[1]),
              ("numpy_ms", "exact, NumPy (BLAS)", SERIES[2])]
    pts = sorted(points, key=lambda p: p["rows"])
    for key, name, col in series:
        c = hex_bgr(col)
        P = [(X(p["rows"]), Y1(p[key])) for p in pts]
        for a, b, pa, pb in zip(P, P[1:], pts, pts[1:]):
            if pa["group"] == pb["group"]:
                cv2.line(img, a, b, c, 2, cv2.LINE_AA)
        for p in P:
            cv2.circle(img, p, 5, c, -1, cv2.LINE_AA)
            cv2.circle(img, p, 5, (251, 252, 252), 2, cv2.LINE_AA)
            cv2.circle(img, p, 4, c, -1, cv2.LINE_AA)
        put(name, (W - mr + 15, P[-1][1] + 5), ink)
    c = hex_bgr(SERIES[0])
    P = [(X(p["rows"]), Y2(p["recall"])) for p in pts]
    for a, b, pa, pb in zip(P, P[1:], pts, pts[1:]):
        if pa["group"] == pb["group"]:
            cv2.line(img, a, b, c, 2, cv2.LINE_AA)
    for p, pt in zip(pts, P):
        cv2.circle(img, pt, 5, c, -1, cv2.LINE_AA)
    last_x = -10 ** 6
    for p, pt in zip(pts, P):
        if pt[0] - last_x < 70:               # crowded cameras around 35k rows: label the first only
            continue
        last_x = pt[0]
        put(p["label"], (pt[0] - 25, pt[1] + (18 if pt[1] < top2 + ph - 30 else -12)), mute, 0.4)
    put("HNSW", (W - mr + 15, P[-1][1] + 5), ink)
    cv2.imwrite(str(path), img)


def report(cfg: dict, stamp: dict, args) -> None:
    ef0 = cfg["search"]["ef_search"]
    real_res = json.loads((REPORTS / "scaling_7.2_real.json").read_text())
    syn = {}
    for f in sorted(REPORTS.glob("scaling_7.2_synthetic_*.json")):
        r = json.loads(f.read_text())
        syn[r["rows"]] = r
    f2 = lambda v: f"{v:.2f}"
    efs = [str(e) for e in real_res["ef_sweep"]]
    lines = ["# Step 7.2 — engine scaling benchmark (HNSW vs exact brute force)", "",
             f"Generated by `scripts/benchmark_engine.py --report` at git `{stamp['git_hash']}`, config "
             f"`{stamp['config_hash'][:12]}`. k = {real_res['k']} (search.k_per_index), recall@k = |HNSW top k ∩ exact "
             f"top k| / k, latency per query single-threaded (median of {cfg['benchmark']['repetitions']} passes after "
             f"{cfg['benchmark']['warmup']} warm-up queries). HNSW: M {cfg['search']['hnsw']['M']}, efConstruction "
             f"{cfg['search']['hnsw']['ef_construction']}; config ef_search = {ef0}. Exact baselines: the engine's "
             "brute force (scalar dot loop kept bit-identical, as upstream) and NumPy float32 matmul + argpartition "
             "(what steps 4.4 / 5.1 used; multi-threaded BLAS).",
             f"Machine: {real_res['system']['cpu']}, {real_res['system']['logical_cpus']} logical CPUs, "
             f"{real_res['system']['ram_bytes'] / 2 ** 30:.0f} GiB RAM, kernel {real_res['system']['kernel']}.", ""]
    for br, res in real_res["branches"].items():
        lines += [f"## Real sizes — {br} ({real_res['n_queries']} text queries: templated activity prompts + tuning/dev "
                  "query texts)", "",
                  "| index | rows | exact engine ms | exact NumPy ms | " + " | ".join(
                      f"ef {e}: recall / ms" for e in efs) + " |",
                  "|---|---|---|---|" + "---|" * len(efs)]
        for cam, r in res["per_index"].items():
            lines.append(f"| {br}/{cam} | {r['rows']:,} | {f2(r['exact_engine']['median_ms'])} | "
                         f"{f2(r['exact_numpy']['median_ms'])} | " + " | ".join(
                             f"{r['hnsw'][e]['recall']:.3f} / {f2(r['hnsw'][e]['median_ms'])}" for e in efs) + " |")
        b = res["branch"]
        lines.append(f"| **{br}, 6 cameras merged (as the API searches)** | {b['rows']:,} | "
                     f"{f2(b['exact_engine']['median_ms'])} | {f2(b['exact_numpy_all_rows']['median_ms'])} | " +
                     " | ".join(f"{b['hnsw'][e]['recall']:.3f} / {f2(b['hnsw'][e]['median_ms'])}" for e in efs) + " |")
        s = res["single"]
        lines.append(f"| {br}, one index over all cameras (comparison) | {s['rows']:,} | "
                     f"{f2(s['exact_engine']['median_ms'])} | {f2(s['exact_numpy']['median_ms'])} | " +
                     " | ".join(f"{s['hnsw'][e]['recall']:.3f} / {f2(s['hnsw'][e]['median_ms'])}" for e in efs) + " |")
        flt = res["filter"]
        lines += ["", f"p95 at ef {ef0}: merged branch HNSW {f2(b['hnsw'][str(ef0)]['p95_ms'])} ms; "
                  f"exact engine {f2(b['exact_engine']['p95_ms'])} ms.", "",
                  f"**Time filter on real data** (ef {ef0}, oversample {cfg['search']['filter_oversample']}, min "
                  f"fraction {cfg['search']['filter_min_fraction']}): {flt['windows']} random {flt['window_s'] / 60:.0f}-min "
                  f"windows ({cfg['benchmark']['filter_windows_per_camera']} per camera) x {real_res['n_queries']} "
                  f"queries = {flt['searches']} searches: recall@k vs exact filtered search **{flt['recall']:.3f}** "
                  f"(worst window {flt['min_window_recall']:.3f}); methods {flt['methods']}; latency filtered HNSW "
                  f"median {f2(flt['hnsw_filtered']['median_ms'])} / p95 {f2(flt['hnsw_filtered']['p95_ms'])} ms vs exact "
                  f"filtered {f2(flt['exact_filtered']['median_ms'])} / {f2(flt['exact_filtered']['p95_ms'])} ms.", ""]
    if syn:
        lines += ["## Synthetic sizes (random unit 512-d vectors and queries: worst case for graph search)", "",
                  "| rows | build | peak RSS | exact engine ms | exact NumPy ms | " +
                  " | ".join(f"ef {e}: recall / ms" for e in efs) + " |", "|---|---|---|---|---|" + "---|" * len(efs)]
        for n, r in sorted(syn.items()):
            lines.append(f"| {n:,} | {r['build_s'] / 60:.0f} min | {r['peak_rss_mib'] / 1024:.1f} GiB | "
                         f"{f2(r['exact_engine']['median_ms'])} | {f2(r['exact_numpy']['median_ms'])} | " +
                         " | ".join(f"{r['hnsw'][e]['recall']:.3f} / {f2(r['hnsw'][e]['median_ms'])}" for e in efs) + " |")
        lines.append("")
    else:
        lines += ["## Synthetic sizes", "", "Not run yet.", ""]
    points = []
    e0 = str(ef0)
    app = real_res["branches"]["appearance"]
    for label, r in [*((f"app/{cam}", r) for cam, r in app["per_index"].items()), ("app/all", app["single"])]:
        points.append({"label": label, "group": "real", "rows": r["rows"], "hnsw_ms": r["hnsw"][e0]["median_ms"],
                       "exact_ms": r["exact_engine"]["median_ms"], "numpy_ms": r["exact_numpy"]["median_ms"],
                       "recall": r["hnsw"][e0]["recall"]})
    for n, r in sorted(syn.items()):
        points.append({"label": f"random {n // 10 ** 6}M", "group": "random", "rows": n, "hnsw_ms": r["hnsw"][e0]["median_ms"],
                       "exact_ms": r["exact_engine"]["median_ms"], "numpy_ms": r["exact_numpy"]["median_ms"],
                       "recall": r["hnsw"][e0]["recall"]})
    fig = REPO_ROOT / "results" / "figures" / "scaling_7.2.png"
    plot(points, fig, ef0)
    lines += [f"Figure: `{display_path(fig)}` (appearance indexes at real sizes + synthetic, at ef {ef0}; 512-d only).",
              ""]
    lines += args.notes
    (REPORTS / "scaling.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--real", action="store_true")
    p.add_argument("--synthetic", nargs="+", type=int)
    p.add_argument("--report", action="store_true")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--notes", nargs="*", default=[], help="extra lines appended to the report (interpretation)")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    stamp = run_stamp(args.config)
    ef0 = cfg["search"]["ef_search"]
    if args.real:
        res = real(cfg, args)
        (REPORTS / "scaling_7.2_real.json").write_text(json.dumps(res, indent=1))
        if not args.no_log:
            for br, r in res["branches"].items():
                log_experiment("none", {"rows": r["branch"]["rows"], f"recall@k_ef{ef0}": r["branch"]["hnsw"][ef0]["recall"],
                                        "hnsw_median_ms": r["branch"]["hnsw"][ef0]["median_ms"],
                                        "exact_engine_median_ms": r["branch"]["exact_engine"]["median_ms"],
                                        "exact_numpy_median_ms": r["branch"]["exact_numpy_all_rows"]["median_ms"],
                                        "filter_recall": r["filter"]["recall"]},
                               notes=f"step 7.2 engine benchmark, {br}, 6 camera indexes merged, real sizes",
                               stamp=stamp)
    for n in args.synthetic or []:
        res = synthetic(cfg, n, args.art)
        (REPORTS / f"scaling_7.2_synthetic_{n}.json").write_text(json.dumps(res, indent=1))
        if not args.no_log:
            log_experiment("none", {"rows": n, f"recall@k_ef{ef0}": res["hnsw"][ef0]["recall"],
                                    "hnsw_median_ms": res["hnsw"][ef0]["median_ms"],
                                    "exact_engine_median_ms": res["exact_engine"]["median_ms"],
                                    "exact_numpy_median_ms": res["exact_numpy"]["median_ms"], "build_s": res["build_s"]},
                           notes=f"step 7.2 engine benchmark, synthetic random unit 512-d", stamp=stamp)
    if args.report:
        report(cfg, stamp, args)


if __name__ == "__main__":
    main()
