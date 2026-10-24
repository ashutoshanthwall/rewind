"""Step 0.3: turn a camera/date choice into data/selection.txt and a report.

Selects, from data/annotations/clip_survey.csv (written by scripts/survey_meva.py):
  - RGB: every *annotated* clip (incl. clips annotated as empty) of the chosen cameras on the chosen dates.
    Unannotated clips are left out so all selected RGB footage can be evaluated.
  - IR: clips of the chosen IR camera that overlap in time with a selected RGB clip on the same date
    (MEVA has no real activity labels on IR, so IR footage is selected by co-timing, not by annotation).

Usage:
  uv run python scripts/select_footage.py --cameras G436 G424 ... --dates 2018-03-07 ... --ir-camera G479
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SLOT_S = 300  # MEVA clips are nominally 5 min; used only to report co-timing per slot


def select(survey: pd.DataFrame, cameras: list[str], dates: list[str], ir_camera: str | None) -> pd.DataFrame:
    unknown = set(cameras) - set(survey.loc[~survey["ir"], "camera"])
    if unknown:
        raise ValueError(f"Not RGB cameras in the survey: {sorted(unknown)}")
    rgb = survey[survey["camera"].isin(cameras) & survey["date"].isin(dates)
                 & survey["annotated"] & (survey["duration_s"] > 0)]
    missing = {(c, d) for c in cameras for d in dates} - set(zip(rgb["camera"], rgb["date"]))
    if missing:
        raise ValueError(f"No annotated clips for camera/date pairs: {sorted(missing)}")
    parts = [rgb]
    if ir_camera:
        ir_all = survey[(survey["camera"] == ir_camera) & survey["ir"] & survey["date"].isin(dates)
                        & (survey["duration_s"] > 0)]
        if ir_all.empty:
            raise ValueError(f"No IR clips for {ir_camera} on {dates}")
        keep = []
        for _, clip in ir_all.iterrows():
            same_day = rgb[rgb["date"] == clip["date"]]
            overlaps = (same_day["start_s"] < clip["start_s"] + clip["duration_s"]) & \
                       (clip["start_s"] < same_day["start_s"] + same_day["duration_s"])
            keep.append(bool(overlaps.any()))
        parts.append(ir_all[keep])
    sel = pd.concat(parts).sort_values(["date", "start_s", "camera"])
    if sel["s3_key"].duplicated().any():
        raise ValueError("Duplicate S3 keys in selection")
    return sel


def summarise(sel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sel = sel.assign(modality=sel["ir"].map({True: "IR", False: "RGB"}),
                     vss=sel["n_vehicle_starts"] + sel["n_vehicle_stops"],
                     window=sel["date"] + " " + (sel["start_s"] // 3600).astype(str).str.zfill(2) + "h")
    agg = dict(hours=("duration_s", lambda s: s.sum() / 3600), clips=("clip", "size"),
               eval_level_clips=("annot_level", lambda s: int((s == "evaluation").sum())),
               activity_records=("n_events", "sum"), vehicle_starts=("n_vehicle_starts", "sum"),
               vehicle_stops=("n_vehicle_stops", "sum"),
               # IR only: third-party contrib labels (incomplete; recall-type metrics only)
               ir_labelled_clips=("ir_labelled", "sum"), ir_label_records=("ir_label_records", "sum"),
               ir_vehicle_starts=("ir_vehicle_starts", "sum"), ir_vehicle_stops=("ir_vehicle_stops", "sum"),
               GB=("size_bytes", lambda s: s.sum() / 1e9))
    by_window = sel.groupby(["camera", "site", "modality", "window"]).agg(**agg).round(2)
    by_camera = sel.groupby(["camera", "site", "modality"]).agg(**agg).round(2)
    return by_window, by_camera


def markdown_table(df: pd.DataFrame) -> str:
    flat = df.reset_index()
    lines = ["| " + " | ".join(map(str, flat.columns)) + " |", "|" + "---|" * len(flat.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in flat.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cameras", nargs="+", required=True, help="RGB camera ids")
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--ir-camera", default=None)
    p.add_argument("--survey", type=Path, default=REPO_ROOT / "data" / "annotations" / "clip_survey.csv")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "selection.txt")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "results" / "reports" / "selection_0.3.md")
    args = p.parse_args()

    survey = pd.read_csv(args.survey)
    sel = select(survey, args.cameras, args.dates, args.ir_camera)
    by_window, by_camera = summarise(sel)
    rgb = sel[~sel["ir"]]
    slot_cams = rgb.groupby(["date", rgb["start_s"] // SLOT_S])["camera"].nunique()
    totals = (f"{len(sel)} clips, {sel['duration_s'].sum() / 3600:.2f} h "
              f"(RGB {rgb['duration_s'].sum() / 3600:.2f} h, IR {sel.loc[sel['ir'], 'duration_s'].sum() / 3600:.2f} h), "
              f"{sel['size_bytes'].sum() / 1e9:.2f} GB, {int(sel['n_events'].sum())} activity records, "
              f"{int(sel['n_vehicle_starts'].sum())} vehicle_starts, {int(sel['n_vehicle_stops'].sum())} vehicle_stops; "
              f"RGB clips by annotation level: {rgb['annot_level'].value_counts().to_dict()}; "
              f"IR clips with contrib labels: {int(sel['ir_labelled'].sum())} "
              f"({sel.loc[sel['ir_labelled'], 'duration_s'].sum() / 3600:.2f} h, "
              f"{int(sel['ir_label_records'].sum())} records, {int(sel['ir_vehicle_starts'].sum())} vehicle_starts, "
              f"{int(sel['ir_vehicle_stops'].sum())} vehicle_stops)")
    caveat = ("Counts are activity records per clip file, not unique world events: an activity that spans a "
              "5-min clip boundary, or is seen by two cameras, is counted once per clip.")
    cotiming = (f"RGB 5-min slots: {len(slot_cams)}; slots with all {len(args.cameras)} RGB cameras: "
                f"{int((slot_cams == len(args.cameras)).sum())}; distribution {slot_cams.value_counts().sort_index().to_dict()}")
    command = ("uv run python scripts/select_footage.py --cameras " + " ".join(args.cameras)
               + " --dates " + " ".join(args.dates) + (f" --ir-camera {args.ir_camera}" if args.ir_camera else ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(f"# Rewind footage selection (step 0.3). Generated by:\n# {command}\n# {totals}\n")
        f.write("# Format: s3_key<TAB>size_bytes (bucket s3://mevadata-public-01/)\n")
        for key, size in zip(sel["s3_key"], sel["size_bytes"]):
            f.write(f"{key}\t{size}\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Footage selection (step 0.3)\n\n"
        f"Command: `{command}`\n\n**Totals:** {totals}\n\n**Note:** {caveat}\n\n**Co-timing:** {cotiming}\n\n"
        "## Per camera\n\n" + markdown_table(by_camera) + "\n\n## Per camera x time window (date, hour)\n\n"
        + markdown_table(by_window) + "\n")

    print(by_camera.to_string(), "\n")
    print("TOTAL:", totals)
    print("CO-TIMING:", cotiming)
    print(f"Wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
