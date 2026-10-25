"""Open-issue pass item 6 (PROGRESS.md): contrib-vs-Kitware start/end offsets table and the NIST-JSON frame-base evidence (0-based, inclusive end)."""

import json
import sqlite3
from collections import Counter

import pandas as pd
import yaml

from rewind.ingest.annotations import activity_names, load_kpf, parse_kitware_activities
from rewind.paths import REPO_ROOT

MEVA_REPO = REPO_ROOT / "data" / "annotations" / "meva-data-repo"
ANNOT = MEVA_REPO / "annotation" / "DIVA-phase-2" / "MEVA"
SOURCES = ("CMU-v2", "UMD-IBM")           # the two contrib sets used for IR ground truth (step 1.3)
VEHICLE_START_STOP = ("vehicle_starts", "vehicle_stops")
FPS = 30                                  # MEVA RGB clips (manifest); offsets are reported in seconds
WITHIN_S = 2.0                            # report column only: share of offsets within this many seconds


def kitware_spans(selected: set[str], known: set[str]) -> dict[str, list[tuple[str, int, int]]]:
    """{clip: [(type, start, end)]} for Kitware clips with activities, outside our selection."""
    out = {}
    for folder in ("kitware", "kitware-meva-training"):
        for path in (ANNOT / folder).rglob("*.activities.yml"):
            clip = path.name.removesuffix(".activities.yml")
            if clip in selected:
                continue
            acts, _ = parse_kitware_activities(load_kpf(path), known, str(path))
            if acts:
                out[clip] = [(a["name"], a["start"], a["end"]) for a in acts]
    return out


def contrib_spans(source: str, clips: set[str], known: set[str], rename: dict[str, str]) -> dict[str, list]:
    """{clip: [(type, start, end)]} from the contrib -activities.yml (tsr0 = the .json spans, step 1.3)."""
    out = {}
    for path in (ANNOT / "contrib" / source).rglob("*-activities.yml"):
        clip = path.name.removesuffix("-activities.yml")
        if clip not in clips:
            continue
        acts = []
        for rec in yaml.load(path.read_text(), Loader=yaml.CSafeLoader) or []:
            if "act" not in rec:
                continue
            (name,) = list(rec["act"]["act2"])
            name = rename.get(name, name)
            if name not in known:
                raise ValueError(f"{path}: unknown activity {name!r}")
            ((start, end),) = [ts["tsr0"] for ts in rec["act"]["timespan"]]
            if end > start:                       # zero-length records are dropped, as in step 1.3
                acts.append((name, start, end))
        out[clip] = acts
    return out


def tiou(a: tuple, b: tuple) -> float:
    """Temporal IoU of two inclusive frame spans (type, start, end)."""
    inter = min(a[2], b[2]) - max(a[1], b[1]) + 1
    return max(0, inter) / (max(a[2], b[2]) - min(a[1], b[1]) + 1)


def match(cacts: list, kacts: list) -> list[tuple[float, tuple, tuple]]:
    """One-to-one greedy matches of the same type by temporal IoU (> 0), largest first."""
    pairs = sorted(((tiou(x, y), i, j) for i, x in enumerate(cacts) for j, y in enumerate(kacts)
                    if x[0] == y[0] and tiou(x, y) > 0), reverse=True)
    used_c, used_k, out = set(), set(), []
    for t, i, j in pairs:
        if i not in used_c and j not in used_k:
            used_c.add(i)
            used_k.add(j)
            out.append((t, cacts[i], kacts[j]))
    return out


def summary(g: pd.DataFrame) -> pd.Series:
    out = {}
    for col in ("d_start", "d_end"):
        v = g[col] / FPS
        out[f"{col}_med_s"] = round(v.median(), 2)
        out[f"{col}_p10..p90_s"] = f"{v.quantile(0.1):.1f}..{v.quantile(0.9):.1f}"
        out[f"{col}_within_{WITHIN_S:g}s"] = f"{(v.abs() <= WITHIN_S).mean():.0%}"
    out["n"] = len(g)
    return pd.Series(out)


def json_format_evidence() -> Counter:
    """Frame-base evidence from the local contrib .json files (NIST spec p9: 1-based, empty box = track end)."""
    c = Counter()
    for source in SOURCES:
        for path in (ANNOT / "contrib" / source).rglob("*.json"):
            for act in json.loads(path.read_text())["activities"]:
                (loc,) = act["localization"].values()
                keys = {int(k): v for k, v in loc.items()}
                on = [k for k, v in keys.items() if v == 1]
                off = [k for k, v in keys.items() if v == 0]
                c[source, "records"] += 1
                c[source, "start_on_frame_0"] += 0 in on
                for obj in act["objects"]:
                    (boxes,) = obj["localization"].values()
                    boxes = {int(k): v for k, v in boxes.items()}
                    c[source, "objects"] += 1
                    c[source, "empty_terminator"] += any(v == {} for v in boxes.values())
                    if off:
                        c[source, "box_on_off_frame"] += bool(boxes.get(off[0]))
    return c


def main() -> None:
    selected = {r[0] for r in sqlite3.connect(REPO_ROOT / "data" / "rewind.sqlite").execute("SELECT clip FROM videos")}
    known, rename = activity_names(MEVA_REPO)
    kit = kitware_spans(selected, known)
    rows = []
    for source in SOURCES:
        con = contrib_spans(source, set(kit), known, rename)
        n_matched = 0
        for clip, cacts in con.items():
            for t, x, y in match(cacts, kit[clip]):
                rows.append(dict(source=source, act=x[0], tiou=t, d_start=x[1] - y[1], d_end=x[2] - y[2]))
                n_matched += 1
        n_c, n_k = sum(map(len, con.values())), sum(len(kit[k]) for k in con)
        print(f"{source}: {len(con)} shared RGB clips outside the selection; {n_c} contrib / {n_k} Kitware activities; "
              f"matched {n_matched} ({n_matched / n_c:.0%} of contrib, {n_matched / n_k:.0%} of Kitware)")
    d = pd.DataFrame(rows)
    vss = d.act.isin(VEHICLE_START_STOP)
    for name, sub in (("all, tIoU > 0", d), ("all, tIoU >= 0.3", d[d.tiou >= 0.3]),
                      ("vehicle start/stop, tIoU > 0", d[vss])):
        print(f"\n== offsets contrib - Kitware, {name}")
        print(sub.groupby("source").apply(summary, include_groups=False).to_string())
    print("\nexact start offsets of -2..+2 frames:",
          {s: Counter(g.d_start[g.d_start.abs() <= 2]).most_common() for s, g in d.groupby("source")})
    print("\nNIST-JSON format evidence:", dict(json_format_evidence()))


if __name__ == "__main__":
    main()
