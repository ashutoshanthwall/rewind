"""Step 1.1: build the ingest manifest (data/rewind.sqlite, table `videos`) from the originals in data/raw/.

For every selected file: parse the name, read fps/frames/resolution from the container, check the name against
the container, the MEVA clip table and the annotation files, compute SHA-256, insert one row.
- All names are checked (clip table, annotations, one site per camera) before anything is hashed or written.
- Resumable: clips already in the manifest are skipped (their size is re-checked). One commit per file.
- Read-only on data/raw/. Every inconsistency raises; nothing is skipped silently.
- --verify re-hashes and re-probes every file and compares with the manifest, changing nothing.
- Writes a per-camera summary to results/reports/manifest_1.1.md.

Run from the repo root (so `rewind` is importable without installing the package):
  uv run python -m scripts.build_manifest            build / resume, then write the report
  uv run python -m scripts.build_manifest --verify   re-hash everything and compare (read-only)
"""

import argparse
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from rewind.ingest.manifest import VIDEO_SUFFIX, connect, parse_video_filename, probe_video, sha256_file
from scripts.download import parse_selection

REPO_ROOT = Path(__file__).resolve().parent.parent
MEVA_REPO = REPO_ROOT / "data" / "annotations" / "meva-data-repo"
CLIP_TABLE = MEVA_REPO / "metadata" / "meva-clip-camera-and-time-table.txt"
ANNOT_ROOT = MEVA_REPO / "annotation" / "DIVA-phase-2" / "MEVA"
KITWARE_SOURCES = ("kitware", "kitware-meva-training")  # evaluation-level, training-level (docs/meva_notes.md)


def camera_modality(clip_table: Path, cameras: set[str]) -> dict[str, str]:
    """'ir' if the camera's clips are in camera set 'IR' of the MEVA clip table, else 'rgb'.

    Decided per camera: 7 selected clips (session edges) are missing from the clip table.
    """
    sets: dict[str, set[str]] = {}
    for n, line in enumerate(clip_table.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 7:
            raise ValueError(f"{clip_table}:{n}: expected 7 fields, got {line!r}")
        sets.setdefault(fields[0].rsplit(".", 1)[1], set()).add(fields[3])
    out = {}
    for cam in sorted(cameras):
        if cam not in sets:
            raise ValueError(f"Camera {cam} has no clip in {clip_table}; modality unknown")
        is_ir = {s == "IR" for s in sets[cam]}
        if len(is_ir) > 1:
            raise ValueError(f"Camera {cam} is in both IR and non-IR camera sets: {sorted(sets[cam])}")
        out[cam] = "ir" if is_ir.pop() else "rgb"
    return out


def annotation_index(annot_root: Path) -> tuple[dict[str, str], set[str]]:
    """Return ({clip: kitware source}, {clips with third-party contrib activity files})."""
    kitware = {}
    for source in KITWARE_SOURCES:
        for p in (annot_root / source).rglob("*.activities.yml"):
            clip = p.name.removesuffix(".activities.yml")
            if clip in kitware:  # none in commit 421841a; decide explicitly if a new release has any
                raise ValueError(f"{clip} is annotated in both {kitware[clip]} and {source}")
            kitware[clip] = source
    contrib = {p.name.removesuffix("-activities.yml") for p in (annot_root / "contrib").rglob("*-activities.yml")}
    if not kitware or not contrib:
        raise FileNotFoundError(f"No annotation files under {annot_root}; see docs/meva_notes.md for the checkout")
    return kitware, contrib


def check_name_vs_container(filename: str, name_duration_s: float, duration_s: float, max_diff_s: float) -> None:
    diff = duration_s - name_duration_s
    if abs(diff) > max_diff_s:
        raise ValueError(f"{filename}: container duration {duration_s:.3f} s vs filename {name_duration_s:.0f} s "
                         f"(diff {diff:+.3f} s > ingest.max_name_duration_diff_s = {max_diff_s})")


def selected_files(raw: Path, selection: Path) -> list[Path]:
    """Files in raw/, which must be exactly the files listed in the selection."""
    wanted = {Path(key).name for key, _ in parse_selection(selection)}
    present = {p.name for p in raw.iterdir()}
    if wanted != present:
        raise ValueError(f"{raw} does not match {selection}: missing {sorted(wanted - present)[:5]}, "
                         f"unexpected {sorted(present - wanted)[:5]}")
    return sorted(raw / n for n in wanted)


def validate_names(files: list[Path], modality: dict[str, str], kitware: dict[str, str]) -> None:
    """Checks on names only, run before any hashing: one site per camera, every RGB clip has a Kitware file."""
    names = [parse_video_filename(p.name) for p in files]
    sites: dict[str, set[str]] = {}
    for n in names:
        sites.setdefault(n.camera_id, set()).add(n.site)
    multi = {cam: sorted(s) for cam, s in sites.items() if len(s) > 1}
    if multi:
        raise ValueError(f"Cameras with more than one site in their filenames: {multi}")
    missing = [n.clip for n in names if modality[n.camera_id] == "rgb" and n.clip not in kitware]
    if missing:  # the selection unit is annotated RGB clips (Gate 0.3)
        raise ValueError(f"{len(missing)} RGB clips without a Kitware annotation file: {missing[:5]}")


def build(conn, files: list[Path], root: Path, modality: dict[str, str], max_diff_s: float) -> None:
    """Insert one row per file not yet in the manifest. Paths are stored relative to root."""
    done = {r["clip"]: r["size_bytes"] for r in conn.execute("SELECT clip, size_bytes FROM videos")}
    t0, hashed = time.time(), 0
    for i, path in enumerate(files, start=1):
        name = parse_video_filename(path.name)
        size = path.stat().st_size
        if name.clip in done:
            if done[name.clip] != size:
                raise ValueError(f"{path.name}: size {size} differs from manifest {done[name.clip]}")
            continue
        info = probe_video(path)
        check_name_vs_container(path.name, (name.name_end_time - name.start_time).total_seconds(),
                                info["duration_s"], max_diff_s)
        digest = sha256_file(path)
        conn.execute(
            "INSERT INTO videos (clip, path, size_bytes, sha256, camera_id, site, modality, start_time, name_end_time,"
            " duration_s, n_frames, fps, width, height, codec, ingested_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name.clip, str(path.relative_to(root)), size, digest, name.camera_id, name.site,
             modality[name.camera_id], name.start_time.isoformat(), name.name_end_time.isoformat(),
             info["duration_s"], info["n_frames"], info["fps"], info["width"], info["height"], info["codec"],
             datetime.now(timezone.utc).isoformat(timespec="seconds")))
        conn.commit()
        hashed += 1
        print(f"[{i}/{len(files)}] {path.name}  {info['n_frames']} frames @ {info['fps']:g} fps  "
              f"{digest[:12]}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"Inserted {hashed}, already present {len(files) - hashed}.")


def verify(conn, files: list[Path], root: Path) -> bool:
    """Re-hash and re-probe every file; print each mismatch; return True only if everything matches."""
    rows = {r["clip"]: r for r in conn.execute("SELECT clip, path, size_bytes, sha256, n_frames, fps FROM videos")}
    clips = {p.name.removesuffix(VIDEO_SUFFIX) for p in files}
    ok = set(rows) == clips
    if not ok:
        print(f"MISMATCH: manifest vs files: only in manifest {sorted(set(rows) - clips)[:5]}, "
              f"only on disk {sorted(clips - set(rows))[:5]}")
    t0 = time.time()
    for i, clip in enumerate(sorted(set(rows) & clips), start=1):
        r = rows[clip]
        path = root / r["path"]
        size, digest, info = path.stat().st_size, sha256_file(path), probe_video(path)
        if size != r["size_bytes"] or digest != r["sha256"]:
            ok = False
            print(f"MISMATCH {clip}: size {size} vs {r['size_bytes']}, sha256 {digest} vs {r['sha256']}")
        if (info["n_frames"], info["fps"]) != (r["n_frames"], r["fps"]):
            ok = False
            print(f"MISMATCH {clip}: n_frames/fps {info['n_frames']}/{info['fps']} vs {r['n_frames']}/{r['fps']}")
        if i % 50 == 0:
            print(f"  re-hashed {i}/{len(rows)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"Re-hashed and re-probed {len(rows)} files in {time.time() - t0:.0f} s: {'ALL MATCH' if ok else 'MISMATCHES FOUND'}")
    return ok


def sessions(df: pd.DataFrame, max_gap_s: float) -> tuple[int, list[float]]:
    """Count runs of back-to-back clips of one camera; return (n_sessions, within-session gaps in seconds).

    gap = next start - (this start + container duration); negative means the clips overlap.
    """
    df = df.sort_values("start")
    gaps = (df["start"].shift(-1) - (df["start"] + pd.to_timedelta(df["duration_s"], unit="s"))).dt.total_seconds()
    within = gaps[gaps.abs() <= max_gap_s].tolist()
    return 1 + int((gaps.dropna().abs() > max_gap_s).sum()), within


def report(conn, max_diff_s: float, kitware: dict[str, str], contrib: set[str]) -> str:
    df = pd.read_sql("SELECT * FROM videos", conn)
    df["start"] = pd.to_datetime(df["start_time"])
    df["name_s"] = (pd.to_datetime(df["name_end_time"]) - df["start"]).dt.total_seconds()
    df["date"] = df["start"].dt.date.astype(str)

    lines = ["# Ingest manifest (step 1.1)", "",
             f"Generated by `uv run python -m scripts.build_manifest` from `data/rewind.sqlite` "
             f"({len(df)} videos). Times are local wall-clock from the filenames.", "",
             "| camera | site | modality | resolution | fps | clips | hours | GB | clips per date | sessions "
             "| frames min-max | container - name duration (s) | gap between consecutive clips (s) "
             "| annotation files |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cam, g in df.groupby("camera_id"):
        n_sessions, gaps = sessions(g, max_diff_s)
        diff = g["duration_s"] - g["name_s"]
        if g["modality"].iat[0] == "rgb":
            levels = Counter(kitware[c] for c in g["clip"])
            annot = f"evaluation {levels['kitware']}, training {levels['kitware-meva-training']}"
        else:
            annot = f"third-party {int(g['clip'].isin(contrib).sum())}, Kitware {int(g['clip'].isin(kitware).sum())}"
        per_date = ", ".join(f"{d[5:]}: {n}" for d, n in g["date"].value_counts().sort_index().items())
        gap_txt = f"{min(gaps):+.2f} .. {max(gaps):+.2f}" if gaps else "n/a"
        lines.append(
            f"| {cam} | {g['site'].iat[0]} | {g['modality'].iat[0]} | {g['width'].iat[0]}x{g['height'].iat[0]} "
            f"| {'/'.join(f'{f:g}' for f in sorted(g['fps'].unique()))} | {len(g)} "
            f"| {g['duration_s'].sum() / 3600:.2f} | {g['size_bytes'].sum() / 1e9:.2f} | {per_date} | {n_sessions} "
            f"| {g['n_frames'].min()}-{g['n_frames'].max()} | {diff.min():+.2f} .. {diff.max():+.2f} | {gap_txt} "
            f"| {annot} |")
    lines += ["", f"**Total:** {len(df)} clips, {df['duration_s'].sum() / 3600:.2f} h, "
              f"{df['size_bytes'].sum() / 1e9:.2f} GB "
              f"(RGB {df.loc[df['modality'] == 'rgb', 'duration_s'].sum() / 3600:.2f} h, "
              f"IR {df.loc[df['modality'] == 'ir', 'duration_s'].sum() / 3600:.2f} h).",
              f"Sessions: runs of clips whose gap is within +/- {max_diff_s} s (ingest.max_name_duration_diff_s). "
              "A negative gap means consecutive clips overlap by that much (by container duration)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="re-hash every file and compare; change nothing")
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--selection", type=Path, default=REPO_ROOT / "data" / "selection.txt")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "rewind.sqlite")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "manifest_1.1.md")
    args = parser.parse_args()

    max_diff_s = float(yaml.safe_load(args.config.read_text())["ingest"]["max_name_duration_diff_s"])
    files = selected_files(args.raw, args.selection)
    if args.verify:
        if not args.db.exists():
            sys.exit(f"{args.db} does not exist; build it first")
        sys.exit(0 if verify(connect(args.db), files, REPO_ROOT) else 1)

    modality = camera_modality(CLIP_TABLE, {parse_video_filename(p.name).camera_id for p in files})
    kitware, contrib = annotation_index(ANNOT_ROOT)
    validate_names(files, modality, kitware)
    conn = connect(args.db)
    build(conn, files, REPO_ROOT, modality, max_diff_s)
    text = report(conn, max_diff_s, kitware, contrib)
    args.report.write_text(text)
    print(text)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
