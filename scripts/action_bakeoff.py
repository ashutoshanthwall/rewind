"""Step 5.1: action-branch bake-off on dev and GATE 5.1 (pre-registered in PROGRESS.md before any result).

Candidates, on identical windows (rewind/embed/windows.py: action.window_s, stride action.stride_s, every dev clip):
  (a) mean-pooled CLIP, the simple variant: the step 4.2 appearance vectors of the config's model/tag, frame view
      'pad' (letterboxed whole frame, as appearance.frame_mode), raw (appearance.mean_center is false), 1 fps samples
      of each window averaged. No new embedding run.
  (b) a video-text dual encoder, the complex variant: window vectors and query text vectors written by
      scripts/embed_action.py (--split dev, then --encode-queries dev).
Search: the top search.k_per_index windows by cosine -> clips (rewind/search/core.py, a window's time = its centre).
Specific queries search every camera, category queries RGB windows only (as in step 4.4). Queries: dev, qtype in
QTYPES. Metrics and the gate: the code used for the baselines and Gate 4.4 (scripts/run_baselines.py,
rewind/eval/gate.py paired_decision as written; adopt (b) only if it adopts the complex variant).

Run from the repo root:  uv run python -m scripts.action_bakeoff --model <action model key>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from rewind.embed.clip_model import Clip
from rewind.embed.windows import pooled_windows, window_starts
from rewind.eval.gate import paired_decision
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect
from rewind.paths import REPO_ROOT, display_path
from rewind.search.core import SearchConfig, group_hits, top_k
from scripts.appearance_search import query_rows, split_videos
from scripts.collect_appearance import index_dir
from scripts.run_baselines import summarise
from scripts.select_footage import rows_table

QTYPES = ("action", "vehicle_event")
WINDOW_COLUMNS = ["video_id", "camera_id", "modality", "window_start", "t_s"]


class WindowIndex:
    """Window vectors of one candidate over one split's videos (brute force)."""

    def __init__(self, x: np.ndarray, meta: pd.DataFrame, videos: list):
        if len(x) != len(meta):
            raise ValueError(f"{len(x)} vectors for {len(meta)} metadata rows")
        self.x, self.meta = x.astype(np.float32), meta.reset_index(drop=True)
        self.by_id = {v.video_id: v for v in videos}
        self.durations = {v.video_id: v.n_frames / v.fps for v in videos}
        self.rgb = (self.meta["modality"] == "rgb").to_numpy()
        self.footage_s = sum(self.durations.values())

    def search(self, q: np.ndarray, cfg: SearchConfig, rgb_only: bool) -> list:
        rows = np.flatnonzero(self.rgb) if rgb_only else np.arange(len(self.x))
        idx, sc = top_k(self.x[rows], q, cfg.k)
        return group_hits(rows[idx], sc, self.meta["video_id"].to_numpy(), self.meta["t_s"].to_numpy(),
                          self.durations, cfg)

    def ranked(self, q: np.ndarray, cfg: SearchConfig, rgb_only: bool, query: dict | None = None) -> list:
        return [c.span(self.by_id[c.video_id]) for c in self.search(q, cfg, rgb_only)]


def window_meta(v, starts: np.ndarray, window_s: float) -> pd.DataFrame:
    return pd.DataFrame({"video_id": v.video_id, "camera_id": v.camera_id, "modality": v.modality,
                         "window_start": starts, "t_s": starts + window_s / 2})


def clip_pooled(cfg: dict, videos: list, art: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Candidate (a): mean-pooled 1 fps 'pad' frame vectors of the collected appearance index."""
    a, act = cfg["appearance"], cfg["action"]
    d = index_dir(art, a["model"], a["pretrained"])
    vecs, meta = np.load(d / "vectors.npy", mmap_mode="r"), pd.read_parquet(d / "meta.parquet")
    frames = meta[(meta["type"] == "frame") & (meta["view"] == "pad")]
    by_video = {vid: g.sort_values("t_s") for vid, g in frames.groupby("video_id")}
    xs, metas = [], []
    for v in videos:
        g = by_video.get(v.video_id)
        if g is None:
            raise ValueError(f"{v.clip}: no pad frame vectors in the appearance index")
        starts = window_starts(v.n_frames / v.fps, act["window_s"], act["stride_s"])
        xs.append(pooled_windows(g["t_s"].to_numpy(), np.asarray(vecs[g["vector_id"].to_numpy()]), starts,
                                 act["window_s"], cfg["sampling"]["appearance_fps"]))
        metas.append(window_meta(v, starts, act["window_s"]))
    return np.concatenate(xs), pd.concat(metas, ignore_index=True)


def action_dir(art: Path, model_key: str) -> Path:
    return art / "embeddings" / "action" / model_key


def load_action(art: Path, model_key: str, videos: list, cfg: dict) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Candidate (b): per-clip window vectors from scripts/embed_action.py; windows must equal window_starts()."""
    act = cfg["action"]
    xs, metas, settings, markers = [], [], set(), []
    for v in videos:
        d = action_dir(art, model_key) / v.camera_id
        done = d / f"{v.clip}.done.json"
        if not done.exists():
            raise FileNotFoundError(f"{v.clip}: no {model_key} window vectors ({done})")
        mk = json.loads(done.read_text())
        settings.add(mk["settings"])
        markers.append(mk)
        x, m = np.load(d / f"{v.clip}.npy"), pd.read_parquet(d / f"{v.clip}.parquet")
        starts = window_starts(v.n_frames / v.fps, act["window_s"], act["stride_s"])
        if len(x) != len(m) or not np.array_equal(m["window_start"].to_numpy(), starts) or (m.video_id != v.video_id).any():
            raise ValueError(f"{v.clip}: {model_key} windows differ from window_starts()")
        xs.append(x)
        metas.append(m[WINDOW_COLUMNS])
    if len(settings) != 1:
        raise ValueError(f"{model_key}: clips embedded with {len(settings)} different settings")
    return np.concatenate(xs), pd.concat(metas, ignore_index=True), {"settings": settings.pop(), "markers": markers}


class TextLookup:
    """encode_text() over query vectors precomputed by scripts/embed_action.py --encode-queries."""

    def __init__(self, path: Path, settings: str):
        z = np.load(path)
        if str(z["settings"]) != settings:
            raise ValueError(f"{path}: text vectors from settings {z['settings']}, window vectors from {settings}")
        self.vec = {t: z["vectors"][i] for i, t in enumerate(z["texts"].tolist())}

    def encode_text(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self.vec]
        if missing:
            raise KeyError(f"no precomputed text vector for {missing}")
        return np.stack([self.vec[t] for t in texts]).astype(np.float32)


def fmt(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.3f}" if isinstance(x, float) else str(x))


def family(dec: dict, key: str, f: str) -> str:
    x = dec["families"].get(key)
    if x is None:
        return "—"
    ci = "n<2" if x["ci"] is None else f"[{f.format(x['ci'][0])}, {f.format(x['ci'][1])}]"
    return f"{f.format(x['mean_diff'])} {ci} (n={x['n']})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    p.add_argument("--art", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--model", required=True, help="action model key (directory under artifacts/embeddings/action)")
    p.add_argument("--split", choices=("tuning", "dev"), default="dev", help="GATE 5.1 uses dev")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "action_5.1.md")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    a, act, m = cfg["appearance"], cfg["action"], cfg["metrics"]
    scfg = SearchConfig.from_config(cfg)
    ap_key = f"AP@{max(m['category_k'])}"
    conn = connect(args.db)
    videos = split_videos(conn, args.split)

    t0 = time.perf_counter()
    xa, ma = clip_pooled(cfg, videos, args.art)
    pool_s = time.perf_counter() - t0
    xb, mb, info = load_action(args.art, args.model, videos, cfg)
    if not (ma[["video_id", "window_start"]].equals(mb[["video_id", "window_start"]])):
        raise ValueError("the two candidates have different windows")
    names = {"a": f"mean-pooled CLIP {a['model']}/{a['pretrained']} (pad, 1 fps)", "b": args.model}
    clip = Clip(a["model"], a["pretrained"])
    rows = {"a": query_rows(conn, args.split, WindowIndex(xa, ma, videos), clip, cfg, scfg, QTYPES),
            "b": query_rows(conn, args.split, WindowIndex(xb, mb, videos),
                            TextLookup(action_dir(args.art, args.model) / f"text_vectors_{args.split}.npz", info["settings"]),
                            cfg, scfg,
                            QTYPES)}
    del clip
    torch.cuda.empty_cache()
    dec = paired_decision(rows["a"], rows["b"], ap_key, m["bootstrap_n"], m["bootstrap_seed"], m["ci_level"])
    summ = {k: summarise(r, cfg) for k, r in rows.items()}

    mk = info["markers"]
    n_win = sum(x["windows"] for x in mk)
    cost = {"b": {"peak_vram_mib": max(x["peak_vram_mib"] for x in mk),
                  "windows_per_s": n_win / sum(x["seconds"] for x in mk),
                  "model_windows_per_s": n_win / sum(x["model_seconds"] for x in mk)},
            "a": {"pool_windows_per_s": len(xa) / pool_s}}
    if not args.no_log:
        for k in ("a", "b"):
            log_experiment(args.split, summ[k] | {"windows": len(xa)} | cost[k],
                           notes=f"step 5.1 action bake-off ({'simple' if k == 'a' else 'complex'}) {names[k]}; "
                                 f"qtypes {'+'.join(QTYPES)}; windows {act['window_s']} s / stride {act['stride_s']} s; "
                                 f"NumPy brute force", stamp=stamp)

    ks = m["recall_k"]
    lines = [
        "# Step 5.1 — action-branch bake-off on dev, GATE 5.1", "",
        f"Generated by `scripts/action_bakeoff.py --model {args.model}` at git `{stamp['git_hash']}`, config "
        f"`{stamp['config_hash'][:12]}`. Split {args.split}: {len(videos)} clips, "
        f"{sum(v.n_frames / v.fps for v in videos) / 3600:.2f} h, {len(xa):,} windows of {act['window_s']} s at stride "
        f"{act['stride_s']} s (identical for both candidates). Queries: qtype {' / '.join(QTYPES)}. Search: top "
        f"{scfg.k} windows -> clips (merge {scfg.merge_gap_s} s, cap {scfg.max_clip_s} s, mean top-3, "
        f"±{scfg.context_s} s; window time = centre); category queries search RGB windows only. Decision rule and "
        "everything above pre-registered in PROGRESS.md (step 5.1) before this run.", "",
        "## Candidates", "",
        rows_table(["candidate", "specific median s", "mean s", *(f"R@{k}" for k in ks), "category mAP@10",
                    "mAP@50"],
                   [[f"({k}) {names[k]}", fmt(summ[k]["median_seconds"]), fmt(summ[k]["mean_seconds"]),
                     *(fmt(summ[k][f"R@{k_}"]) for k_ in ks), fmt(summ[k]["mAP@10"]), fmt(summ[k]["mAP@50"])]
                    for k in ("a", "b")]), "",
        "## Cost", "",
        f"- (b) {args.model}: peak VRAM {cost['b']['peak_vram_mib']:.0f} MiB; {cost['b']['windows_per_s']:.1f} "
        f"windows/s end to end (decode + preprocessing + model), {cost['b']['model_windows_per_s']:.1f} windows/s "
        f"model only; settings `{info['settings']}`.",
        f"- (a) mean-pooled CLIP: no new GPU run; pooling {cost['a']['pool_windows_per_s']:,.0f} windows/s on the CPU. "
        "Its frame vectors come from the step 4.2 appearance run (peak VRAM 2,097 MiB, 27x real time, all views "
        "and crops), which Rewind needs anyway.", "",
        f"## GATE 5.1 (paired rule, `rewind/eval/gate.py`; specific: seconds watched, category: {ap_key})", "",
        rows_table(["simple", "complex", "wins / ties / losses", "specific mean diff s [CI]",
                    f"category mean diff {ap_key} [CI]", "mean better", "adopt complex"],
                   [["(a)", "(b)", f"{dec['wins']} / {dec['ties']} / {dec['losses']}",
                     family(dec, "specific_seconds", "{:+.0f}"), family(dec, "category_" + ap_key, "{:+.3f}"),
                     dec["mean_better"], dec["adopt"]]]), "",
        f"**Decision by the pre-registered rule: {'ADOPT ' + args.model if dec['adopt'] else 'DROP the action branch'}**"
        " (positive differences favour (b)).", "",
        "## Per query", "",
        rows_table(["query", "kind", "qtype", "modality", "(a) seconds / " + ap_key, "(a) hit rank",
                    "(b) seconds / " + ap_key, "(b) hit rank", "better"],
                   [[ra["query_id"], ra["kind"], ra["qtype"], ra["modality"],
                     fmt(ra["seconds"] if ra["kind"] == "specific" else ra[ap_key]), ra.get("hit_rank", ""),
                     fmt(rb["seconds"] if rb["kind"] == "specific" else rb[ap_key]), rb.get("hit_rank", ""),
                     _better(ra, rb, ap_key)]
                    for ra, rb in zip(rows["a"], rows["b"])]), "",
        "(specific: seconds watched, lower is better, hit rank -1 = no hit in the list; a miss in both is a tie. "
        f"category: {ap_key}, higher is better)", "",
        "## By qtype (specific queries: median seconds, R@10; category: mean " + ap_key + "); descriptive, n is tiny", "",
    ]
    for k in ("a", "b"):
        df = pd.DataFrame(rows[k])
        trs = []
        for (kind, qt), g in df.groupby(["kind", "qtype"]):
            trs.append([f"({k})", kind, qt, len(g),
                        fmt(float(g.seconds.median())) if kind == "specific" else fmt(float(g[ap_key].mean())),
                        fmt(float(g["R@10"].mean())) if kind == "specific" else "—"])
        lines += [rows_table(["cand.", "kind", "qtype", "n", "median s / mean AP", "R@10"], trs), ""]
    args.report.write_text("\n".join(lines))
    print("\n".join(lines))
    print("report:", display_path(args.report))
    print("DECISION:", "ADOPT" if dec["adopt"] else "DROP")


def _better(ra: dict, rb: dict, ap_key: str) -> str:
    if ra["kind"] == "specific":
        d = 0.0 if not ra["hit"] and not rb["hit"] else ra["seconds"] - rb["seconds"]
    else:
        d = rb[ap_key] - ra[ap_key]
    return "(b)" if d > 0 else "(a)" if d < 0 else "tie"


if __name__ == "__main__":
    main()
