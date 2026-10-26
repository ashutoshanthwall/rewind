"""Step 6.2, GATE 6.2 part 2 (dev): appearance search with the event branch on vs off.

Pre-registered in PROGRESS.md ("Phase 6 — PRE-REGISTRATION"):
  OFF = appearance-only search at the config's Gate 4.4 variant (scripts/appearance_search.Index).
  ON  = for a query whose text has an intent keyword of an enabled type (events.types; rewind/search/events.intent),
        the detected events of those types in the split (RGB only for category queries), ranked by cosine(crop
        vector, query vector) and grouped into clips (rewind/search/events.event_clips), first; then the OFF list.
        Queries without intent get the OFF list (tie).
  Queries: the dev queries with qtype vehicle_event; decision: rewind/eval/gate.paired_decision (OFF simple, ON
  complex), as written. Keep the event branch iff Part 1 left >= 1 type (events.types non-empty) AND ON is adopted.
Also reported (outside the rule): intent matches over every dev query.
Writes results/reports/events_6.2.md and two experiments rows (OFF, ON). Dev only; test is refused.

Run from the repo root:
  uv run python -m scripts.event_gate
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rewind.embed.clip_model import Clip
from rewind.eval.gate import paired_decision
from rewind.eval.metrics import Span
from rewind.eval.queries import load_queries
from rewind.experiment_log import log_experiment, run_stamp
from rewind.ingest.manifest import connect
from rewind.paths import REPO_ROOT, display_path
from rewind.search.core import SearchConfig
from rewind.search.events import event_clips, intent
from scripts.appearance_search import Index, query_rows, split_videos
from scripts.collect_appearance import index_dir, load_means
from scripts.embed_events import load_event_vectors
from scripts.run_baselines import QUERIES_DIR, summarise

SPLIT = "dev"
QTYPES = ("vehicle_event",)


class EventIndex:
    """OFF ranker + the event branch in front of it for queries with intent (module docstring)."""

    def __init__(self, base: Index, vecs: np.ndarray, meta: pd.DataFrame, keywords: dict, enabled: list[str]):
        self.base, self.vecs, self.meta = base, vecs.astype(np.float32), meta.reset_index(drop=True)
        self.keywords, self.enabled = keywords, enabled
        self.footage_s = base.footage_s
        self.video_ids = set(base.by_id)
        self.n_event_clips = {}

    def ranked(self, q: np.ndarray, cfg: SearchConfig, rgb_only: bool, query: dict | None = None) -> list[Span]:
        if query is None:
            raise ValueError("the event branch needs the query text")
        off = self.base.ranked(q, cfg, rgb_only)
        types = intent(query["query"], self.keywords, self.enabled)
        if not types:
            self.n_event_clips[query["query_id"]] = 0
            return off
        clips = event_clips(q, self.vecs, self.meta, types, self.video_ids, rgb_only, self.base.durations, cfg)
        self.n_event_clips[query["query_id"]] = len(clips)
        return [c.span(self.base.by_id[c.video_id]) for c in clips] + off


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    ap.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    args = ap.parse_args()
    stamp = run_stamp(args.config)
    cfg = yaml.safe_load(args.config.read_text())
    a, m, e = cfg["appearance"], cfg["metrics"], cfg["events"]
    if not isinstance(e["types"], list):
        raise ValueError(f"events.types is {e['types']!r}: set it from Gate 6.2 part 1 first")
    scfg = SearchConfig.from_config(cfg)
    ap_key = f"AP@{max(m['category_k'])}"
    conn = connect(args.db)
    videos = split_videos(conn, SPLIT)
    model, tag = a["model"], a["pretrained"]
    d = index_dir(REPO_ROOT / "artifacts", model, tag)
    vecs, meta = np.load(d / "vectors.npy"), pd.read_parquet(d / "meta.parquet")
    means = load_means(REPO_ROOT / "artifacts", model, tag, a["frame_mode"], a["crop_source"]) if a["mean_center"] else None
    off = Index(vecs, meta, videos, a["frame_mode"], a["crop_source"], a["max_crops_per_frame"], a["mean_center"], means)
    ev_vecs, ev_meta = load_event_vectors(REPO_ROOT / "artifacts", cfg, conn, videos)
    on = EventIndex(off, ev_vecs, ev_meta, e["intent_keywords"], e["types"])
    clip = Clip(model, tag)

    rows_off = query_rows(conn, SPLIT, off, clip, cfg, scfg, QTYPES)
    rows_on = query_rows(conn, SPLIT, on, clip, cfg, scfg, QTYPES)
    dec = paired_decision(rows_off, rows_on, ap_key, m["bootstrap_n"], m["bootstrap_seed"], m["ci_level"])
    keep = bool(e["types"]) and dec["adopt"]
    s_off, s_on = summarise(rows_off, cfg), summarise(rows_on, cfg)
    for name, s in (("OFF (appearance only)", s_off), ("ON (+ event branch)", s_on)):
        log_experiment(SPLIT, {k: v for k, v in s.items() if isinstance(v, (int, float)) and v == v},
                       f"step 6.2 Gate 6.2 dev vehicle_event queries, {name}; types {e['types']}; "
                       f"undefined values not logged", stamp=stamp)

    all_q = load_queries(QUERIES_DIR, SPLIT)
    fmt = lambda x: "—" if x is None or (isinstance(x, float) and x != x) else (f"{x:.3f}" if isinstance(x, float) else str(x))
    lines = ["# Step 6.2 — GATE 6.2 part 2: event branch on vs off (dev, vehicle_event queries)", "",
             f"Generated by `scripts/event_gate.py` at git `{stamp['git_hash']}`, config `{stamp['config_hash'][:12]}`. "
             f"OFF = appearance only ({a['frame_mode']}/{a['crop_source']}/{'centered' if a['mean_center'] else 'raw'}); "
             f"ON = event clips first for queries with intent (enabled types {e['types']}), then OFF. "
             f"{len(ev_meta)} detected events in the dev footage.", "",
             "## Per query", "",
             "| query | kind | text | intent | event clips | OFF seconds / hit rank / AP@50 | ON seconds / hit rank / AP@50 |",
             "|---|---|---|---|---|---|---|"]
    by_on = {r["query_id"]: r for r in rows_on}
    text = {q["query_id"]: q["query"] for q in all_q}
    for r in rows_off:
        o, n = r, by_on[r["query_id"]]
        cell = lambda x: (f"{x['seconds']:.0f} / {x['hit_rank']}" if x["kind"] == "specific" else f"AP {fmt(x[ap_key])}")
        lines.append(f"| {r['query_id']} | {r['kind']} | {text[r['query_id']]} | "
                     f"{', '.join(intent(text[r['query_id']], e['intent_keywords'], e['types'])) or '—'} | "
                     f"{on.n_event_clips[r['query_id']]} | {cell(o)} | {cell(n)} |")
    ci = lambda c: "—" if c is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    fam = "; ".join(f"{k}: n {v['n']}, mean diff {v['mean_diff']:+.3f}, CI {ci(v['ci'])}"
                    for k, v in dec["families"].items())
    lines += ["", "## Decision (pre-registered paired rule, `rewind/eval/gate.py`)", "",
              f"- wins / ties / losses (ON vs OFF): **{dec['wins']} / {dec['ties']} / {dec['losses']}**",
              f"- families (positive = ON better): {fam}",
              f"- mean better: {dec['mean_better']}; adopt ON: **{dec['adopt']}**",
              f"- Part 1 enabled types: {e['types']}",
              f"- **GATE 6.2: {'KEEP' if keep else 'DROP'} the event branch**", "",
              "## Summary", "",
              "| variant | specific median s | mean s | R@1 | R@5 | R@10 | category mAP@10 | mAP@50 |",
              "|---|---|---|---|---|---|---|---|"]
    for name, s in (("OFF", s_off), ("ON", s_on)):
        lines.append(f"| {name} | {fmt(s.get('median_seconds'))} | {fmt(s.get('mean_seconds'))} | {fmt(s.get('R@1'))} | "
                     f"{fmt(s.get('R@5'))} | {fmt(s.get('R@10'))} | {fmt(s.get('mAP@10'))} | {fmt(s.get('mAP@50'))} |")
    lines += ["", "## Intent over all dev queries (outside the rule)", "", "| query | qtype | intent |", "|---|---|---|"]
    for q in all_q:
        lines.append(f"| {q['query_id']} | {q['qtype']} | "
                     f"{', '.join(intent(q['query'], e['intent_keywords'], e['types'])) or '—'} |")
    out = REPO_ROOT / "results" / "reports" / "events_6.2.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"report: {display_path(out)}")


if __name__ == "__main__":
    main()
