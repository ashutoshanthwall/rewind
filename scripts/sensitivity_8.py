"""Step 8.2 sensitivity analysis (Ashutosh, 27 Oct): the primary test results if some missed specific queries are
counted as found, because a returned clip shows the queried event itself on a camera where it is not labelled
("correct but unlabelled"). The primary 8.2 results and the test labels are NOT changed: this writes a separate,
labelled report.

Reads only the 8.2 outputs (results/reports/test_rows.csv, test_clips.json): no search, no labels, no query file.
A scenario counts query q as found at Rewind's clip rank r (a clip in the saved top 10) on a person's judgement:
the hit rule's conditions (same camera, time overlap with the label) are replaced by that judgement, and the report
prints the clip's time gap to the label. Cost: the frozen seconds-watched rule otherwise: the full duration of the
clips ranked before r, plus the time from clip r's start to the labelled event's start (0 if it already started;
cameras share one wall clock). R@k from r. Everything else (other queries, baselines) is the 8.2 result as committed.

Refuses to run without --confirmed-by: each counted clip must first be checked by eye on the proxy.

Run:  uv run python -m scripts.sensitivity_8 --confirmed-by "<who, when, what was seen>" \\
        --scenario "s19=tes-s19:3" --scenario "s12=tes-s12:3" --scenario "s04=tes-s04:1"
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from rewind.eval.metrics import bootstrap_ci
from rewind.paths import REPO_ROOT, display_path
from scripts.run_baselines import SYSTEMS, summarise
from scripts.select_footage import rows_table


def counted(row: dict, q: dict, clips: list, rank: int, ks: list) -> dict:
    if not 1 <= rank <= len(clips):
        raise ValueError(f"{q['query_id']}: rank {rank} is not among the {len(clips)} saved clips")
    if row["hit"]:
        raise ValueError(f"{q['query_id']} is already a hit in 8.2: nothing to count")
    c = clips[rank - 1]
    into = max(0.0, (datetime.fromisoformat(q["start_time"]) - datetime.fromisoformat(c["start"])).total_seconds())
    secs = sum(x["end_s"] - x["start_s"] for x in clips[:rank - 1]) + into
    return row | {"seconds": secs, "hit": True, "hit_rank": rank, **{f"R@{k}": int(rank <= k) for k in ks}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--in-dir", type=Path, default=REPO_ROOT / "results" / "reports")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "reports" / "sensitivity_8.2.md")
    p.add_argument("--scenario", action="append", required=True, help="name=query_id:rank[,query_id:rank]")
    p.add_argument("--confirmed-by", required=True, help="who checked the clips on the proxies, when, what was seen")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    m, ks = cfg["metrics"], cfg["metrics"]["recall_k"]
    rows = pd.read_csv(args.in_dir / "test_rows.csv").to_dict("records")
    saved = json.loads((args.in_dir / "test_clips.json").read_text())
    rewind = [r for r in rows if r["system"] == "rewind"]
    spec_base = {s: {r["query_id"]: r["seconds"] for r in rows if r["system"] == s and r["kind"] == "specific"}
                 for s in SYSTEMS}

    def table_row(name: str, rw: list) -> list:
        s = summarise(rw, cfg)
        hits = sum(bool(r["hit"]) for r in rw if r["kind"] == "specific")
        return [name, f"{hits}/{s['n_specific']}", f"{s['median_seconds']:,.0f} [{s['median_seconds_ci_lo']:,.0f}, "
                f"{s['median_seconds_ci_hi']:,.0f}]", f"{s['mean_seconds']:,.0f}",
                *(f"{s[f'R@{k}']:.3f}" for k in ks)]

    def paired(rw: list) -> list:
        a = {r["query_id"]: r["seconds"] for r in rw if r["kind"] == "specific"}
        out = []
        for s in SYSTEMS:
            d = [spec_base[s][q] - a[q] for q in sorted(a)]
            lo, hi = bootstrap_ci(d, statistics.median, m["bootstrap_n"], m["bootstrap_seed"], m["ci_level"])
            out.append(f"{s} {statistics.median(d):+,.0f} [{lo:+,.0f}, {hi:+,.0f}] "
                       f"({sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x < 0 for x in d)})")
        return out

    head = ["scenario", "found", "median s watched [CI]", "mean s", *(f"R@{k}" for k in ks)]
    trs, prs, notes = [table_row("8.2 primary as committed", rewind)], [["8.2 primary as committed", *paired(rewind)]], []
    for sc in args.scenario:
        name, spec = sc.split("=", 1)
        changes = {q: int(r) for q, r in (x.split(":") for x in spec.split(","))}
        rw = []
        for r in rewind:
            if r["query_id"] in changes:
                q = saved[r["query_id"]]["query"]
                new = counted(r, q, saved[r["query_id"]]["clips"], changes[r["query_id"]], ks)
                c = saved[r["query_id"]]["clips"][changes[r["query_id"]] - 1]
                t = lambda x: datetime.fromisoformat(x)
                gap = max(0.0, (t(c["start"]) - t(q["end_time"])).total_seconds(),
                          (t(q["start_time"]) - t(c["end"])).total_seconds())
                notes.append(f"- **{name}**: {r['query_id']} counted at rank {new['hit_rank']}: clip {c['camera_id']} "
                             f"{c['start']}–{c['end'][11:19]} (label: {q['camera_id']} {q['start_time']}–"
                             f"{q['end_time'][11:19]}; time gap {gap:.1f} s, "
                             f"{'same' if c['camera_id'] == q['camera_id'] else 'other'} camera); seconds "
                             f"{r['seconds']:,.0f} → {new['seconds']:,.0f}")
                rw.append(new)
            else:
                rw.append(r)
        missing = set(changes) - {r["query_id"] for r in rewind}
        if missing:
            raise ValueError(f"unknown queries {sorted(missing)}")
        trs.append(table_row(name, rw))
        prs.append([name, *paired(rw)])
    lines = [
        "# Step 8.2 — SENSITIVITY ANALYSIS (not the primary result)", "",
        f"Generated by `scripts/sensitivity_8.py` on {datetime.now():%Y-%m-%d %H:%M} from the committed 8.2 outputs in "
        f"`{display_path(args.in_dir)}`. **The primary 8.2 results (`test_results.md`) and the test labels are unchanged.** "
        "Each scenario counts the named miss as found at the given Rewind clip rank, because the clip was judged to "
        "show the queried event itself on a camera where it is not labelled. The hit rule's same-camera and time-overlap "
        "conditions are replaced by that judgement; the time gap of each clip to the label is shown. Seconds otherwise "
        "follow the frozen rule (module docstring).", "",
        f"**Checked on the proxies by:** {args.confirmed_by}", "",
        *notes, "",
        rows_table(head, trs), "",
        "Median paired saving vs each baseline, s [95% CI] (faster / ties / slower):", "",
        rows_table(["scenario", *SYSTEMS], prs), ""]
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    print("report:", display_path(args.out))


if __name__ == "__main__":
    main()
