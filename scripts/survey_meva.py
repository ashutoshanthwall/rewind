"""Step 0.3: survey MEVA videos + annotations to choose the ~24 h footage selection.

Joins three sources into one row per released video clip:
  - S3 listing of s3://mevadata-public-01/drops-123-r13/ (name, size)
  - meva-data-repo metadata/meva-clip-camera-and-time-table.txt (camera set; 'IR' = IR camera)
  - meva-data-repo KPF *.activities.yml (activity counts per clip)
  - for IR cameras only: third-party contrib labels (CMU-v2, UMD-IBM, IBM-*), in separate ir_* columns.
    Kitware has no real IR labels; contrib labels are never attached to RGB clips (Gate 0.3 decision).
Writes data/annotations/clip_survey.csv and prints summaries. Read-only on S3; downloads no video.
The meva-data-repo checkout it needs is described in docs/meva_notes.md.

Usage: uv run python scripts/survey_meva.py
"""

import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MEVA_REPO = REPO_ROOT / "data" / "annotations" / "meva-data-repo"
ANNOT_ROOT = MEVA_REPO / "annotation" / "DIVA-phase-2" / "MEVA"
# Dataset facts (verified in meva-data-repo, commit 421841a): annotation sources and their quality level.
ANNOT_SOURCES = {"kitware": "evaluation", "kitware-meva-training": "training"}
VEHICLE_START_STOP = ("vehicle_starts", "vehicle_stops")  # from documents/activity-names.txt
# Not an activity: a whole-clip record meaning "checked, none of the 37 activities present"
# (each such file also carries a meta line "<clip> was marked empty-37").
EMPTY_MARKER = "empty_37"
# Third-party labels (training scope, old activity names), used for IR footage only. Checked on shared RGB
# clips: CMU-v2 has 0.23x and UMD-IBM 0.45x as many records as Kitware, so they are incomplete.
CONTRIB_SOURCES = ("CMU-v2", "UMD-IBM", "IBM-person-vehicle", "IBM-person-person")
S3_PREFIX = "s3://mevadata-public-01/drops-123-r13/"

# date.start.end.site.camera, e.g. 2018-03-07.17-35-06.17-40-06.school.G339
CLIP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.([a-z]+)\.(G\d+)$")


def parse_clip_name(name: str) -> dict:
    m = CLIP_RE.match(name)
    if not m:
        raise ValueError(f"Clip name does not match date.start.end.site.camera: {name!r}")
    date, start, end, site, camera = m.groups()
    to_s = lambda t: sum(int(x) * f for x, f in zip(t.split("-"), (3600, 60, 1)))
    duration = to_s(end) - to_s(start)
    if duration < 0:  # e.g. 23-55-00.00-00-00: the clip ends after midnight
        duration += 24 * 3600
    # duration == 0 exists in the release (one 115 KB clip); kept, but never selectable.
    return {"clip": name, "date": date, "start": start.replace("-", ":"), "end": end.replace("-", ":"),
            "start_s": to_s(start), "duration_s": duration, "site": site, "camera": camera}


def s3_listing() -> pd.DataFrame:
    out = subprocess.run(["aws", "s3", "ls", "--no-sign-request", "--recursive", S3_PREFIX],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        _, _, size, key = line.split(maxsplit=3)
        if not key.endswith(".r13.avi"):
            raise ValueError(f"Unexpected object in video listing: {key}")
        rows.append({**parse_clip_name(Path(key).name.removesuffix(".r13.avi")),
                     "s3_key": key, "size_bytes": int(size)})
    return pd.DataFrame(rows)


def camera_sets() -> pd.DataFrame:
    rows = []
    for line in (MEVA_REPO / "metadata" / "meva-clip-camera-and-time-table.txt").read_text().splitlines():
        fields = line.split()
        if len(fields) != 7:
            raise ValueError(f"Clip table line does not have 7 fields: {line!r}")
        rows.append({"clip": fields[0], "camera_set": fields[3]})
    return pd.DataFrame(rows)


def parse_activities(items: list, known: set[str], where: str,
                     rename: dict[str, str] | None = None) -> tuple[Counter, bool]:
    """Count activity records in one parsed activities file; return (counts, marked_empty).

    rename maps old activity names to current ones (contrib files use the old namespace).
    act2 is a dict in Kitware files and a YAML !!set in contrib files; both iterate over the name.
    """
    counts = Counter()
    marked_empty = False
    for item in items:
        if "act" in item:
            (name,) = list(item["act"]["act2"])
            name = (rename or {}).get(name, name)
            if name == EMPTY_MARKER:
                marked_empty = True
            elif name in known:
                counts[name] += 1
            else:
                raise ValueError(f"Activity {name!r} in {where} is not in activity-names.txt")
        elif "meta" not in item:
            raise ValueError(f"Unexpected KPF record in {where}: {item}")
    if marked_empty and counts:
        raise ValueError(f"{where} is marked {EMPTY_MARKER} but also has activities {dict(counts)}")
    return counts, marked_empty


def ir_by_camera(sets: pd.DataFrame, videos: pd.DataFrame) -> pd.Series:
    """IR flag per camera: a camera is IR if its clips are in camera set 'IR'. Fails if a camera is mixed.

    Decided per camera, not per clip, so clips missing from the clip table can't silently become RGB.
    """
    flagged = sets.assign(camera=sets["clip"].str.rsplit(".", n=1).str[1],
                          is_ir=sets["camera_set"].eq("IR"))
    mixed = flagged.groupby("camera")["is_ir"].nunique()
    if (mixed > 1).any():
        raise ValueError(f"Cameras with both IR and non-IR clips in the clip table: {list(mixed[mixed > 1].index)}")
    ir = flagged.groupby("camera")["is_ir"].first()
    unknown = set(videos["camera"]) - set(ir.index)
    if unknown:
        raise ValueError(f"Cameras with no clip in the clip table, IR status unknown: {sorted(unknown)}")
    return ir


def activity_counts() -> pd.DataFrame:
    known = set((MEVA_REPO / "documents" / "activity-names.txt").read_text().split())
    rows = []
    for source, level in ANNOT_SOURCES.items():
        for path in sorted((ANNOT_ROOT / source).rglob("*.activities.yml")):
            items = yaml.load(path.read_text(), Loader=yaml.CSafeLoader) or []
            counts, marked_empty = parse_activities(items, known, str(path))
            rows.append({"clip": path.name.removesuffix(".activities.yml"), "annot_source": source,
                         "annot_level": level, "marked_empty": marked_empty, "n_events": sum(counts.values()),
                         **{f"n_{a}": counts[a] for a in VEHICLE_START_STOP}, "counts": dict(counts)})
    return pd.DataFrame(rows)


def contrib_ir_counts(ir_cameras: set[str]) -> pd.DataFrame:
    """One row per (contrib source, IR clip) with activity counts, old names mapped to current ones."""
    known = set((MEVA_REPO / "documents" / "activity-names.txt").read_text().split())
    mapping = pd.read_csv(MEVA_REPO / "documents" / "activity-name-mapping.csv")
    rename = dict(zip(mapping["old_name"], mapping["new_name"]))
    rows = []
    for source in CONTRIB_SOURCES:
        for path in sorted((ANNOT_ROOT / "contrib" / source).rglob("*-activities.yml")):
            clip = path.name.removesuffix("-activities.yml")
            if parse_clip_name(clip)["camera"] not in ir_cameras:
                continue  # contrib labels are used for IR only
            items = yaml.load(path.read_text(), Loader=yaml.CSafeLoader) or []
            counts, _ = parse_activities(items, known, str(path), rename)
            rows.append({"clip": clip, "source": source, "records": sum(counts.values()),
                         **{f"n_{a}": counts[a] for a in VEHICLE_START_STOP}})
    if not rows:
        raise ValueError(f"No contrib activity files found for IR cameras {sorted(ir_cameras)} under {ANNOT_ROOT / 'contrib'}")
    return pd.DataFrame(rows)


def combine_contrib(per_source: pd.DataFrame) -> pd.DataFrame:
    """One row per IR clip. Counts are the max over sources (not the sum: sources label the same events)."""
    return per_source.groupby("clip").agg(
        ir_label_sources=("source", lambda s: "+".join(sorted(set(s)))),
        ir_label_records=("records", "max"),
        **{f"ir_{a}": (f"n_{a}", "max") for a in VEHICLE_START_STOP},
    ).reset_index()


def attach_ir_labels(df: pd.DataFrame, ir_labels: pd.DataFrame) -> pd.DataFrame:
    """Join contrib IR labels onto the survey; fail if any would land on an RGB or unknown clip."""
    target = df.set_index("clip").reindex(ir_labels["clip"])
    bad = ir_labels.loc[~target["ir"].eq(True).to_numpy(), "clip"]
    if len(bad):
        raise ValueError(f"Contrib labels for clips that are not IR videos in the survey: {sorted(bad)[:5]}")
    out = df.merge(ir_labels, on="clip", how="left")
    out["ir_labelled"] = out["ir_label_sources"].notna()
    for col in ["ir_label_records", *(f"ir_{a}" for a in VEHICLE_START_STOP)]:
        out[col] = out[col].fillna(0).astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "annotations" / "clip_survey.csv")
    args = parser.parse_args()

    videos = s3_listing()
    sets = camera_sets()
    acts = activity_counts()

    dup = acts[acts.duplicated("clip", keep=False)]
    if len(dup):  # none in commit 421841a; decide explicitly if a new release has any
        raise ValueError(f"{dup['clip'].nunique()} clips annotated in more than one source: {sorted(dup['clip'])[:5]}")
    orphan = sorted(set(acts["clip"]) - set(videos["clip"]))
    print(f"videos on S3: {len(videos)}  annotated clips: {len(acts)}  annotated but no video: {len(orphan)} {orphan[:5]}")

    df = videos.merge(sets, on="clip", how="left").merge(acts, on="clip", how="left")
    df["camera_set"] = df["camera_set"].fillna("not-in-clip-table")
    df["ir"] = df["camera"].map(ir_by_camera(sets, videos)).astype(bool)
    df["annotated"] = df["annot_source"].notna()
    for col in ["n_events", *(f"n_{a}" for a in VEHICLE_START_STOP)]:
        df[col] = df[col].fillna(0).astype(int)

    ir_labels = combine_contrib(contrib_ir_counts(set(df.loc[df["ir"], "camera"])))
    df = attach_ir_labels(df, ir_labels)
    print(f"IR clips with contrib labels: {int(df['ir_labelled'].sum())}  "
          f"records (max over sources): {int(df['ir_label_records'].sum())}  "
          + "  ".join(f"{a}: {int(df[f'ir_{a}'].sum())}" for a in VEHICLE_START_STOP))

    # Activity-type totals over all annotated clips (exact names as they appear in the KPF files).
    totals = Counter()
    for c in df.loc[df["annotated"], "counts"]:
        totals.update(c)
    print("\nActivity instances over all annotated clips:")
    for name, n in sorted(totals.items()):
        print(f"  {name:40s} {n}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["counts"]).to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
