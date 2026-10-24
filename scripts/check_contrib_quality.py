"""Gate 0.3: how complete are the third-party (contrib) labels? Compare them with Kitware on shared RGB clips.

This comparison only judges label quality. Contrib labels are never used as RGB ground truth;
Rewind uses them for IR footage only, with recall-type metrics only.

Usage: uv run python scripts/check_contrib_quality.py   (after scripts/survey_meva.py)
"""

from collections import Counter

import pandas as pd
import yaml

from survey_meva import ANNOT_ROOT, MEVA_REPO, REPO_ROOT, VEHICLE_START_STOP, parse_activities

SOURCES = ("CMU-v2", "UMD-IBM")  # the two contrib sets that claim all 37 activities


def main() -> None:
    known = set((MEVA_REPO / "documents" / "activity-names.txt").read_text().split())
    mapping = pd.read_csv(MEVA_REPO / "documents" / "activity-name-mapping.csv")
    rename = dict(zip(mapping["old_name"], mapping["new_name"]))
    survey = pd.read_csv(REPO_ROOT / "data" / "annotations" / "clip_survey.csv")
    kitware = survey[survey["annotated"] & ~survey["ir"] & ~survey["marked_empty"].fillna(False).astype(bool)]
    kitware = kitware.set_index("clip")
    for source in SOURCES:
        rows = []
        for path in sorted((ANNOT_ROOT / "contrib" / source).rglob("*-activities.yml")):
            clip = path.name.removesuffix("-activities.yml")
            if clip not in kitware.index:
                continue
            counts, _ = parse_activities(yaml.load(path.read_text(), Loader=yaml.CSafeLoader) or [],
                                         known, str(path), rename)
            rows.append({"clip": clip, "records": sum(counts.values()),
                         "vss": sum(counts[a] for a in VEHICLE_START_STOP)})
        if not rows:
            raise ValueError(f"No {source} activity files overlap Kitware RGB clips; is the contrib checkout present?")
        m = pd.DataFrame(rows).set_index("clip").join(kitware[["n_events", "n_vehicle_starts", "n_vehicle_stops"]])
        k_vss = m["n_vehicle_starts"] + m["n_vehicle_stops"]
        print(f"{source}: {len(m)} RGB clips also labelled by Kitware; records {m['records'].sum()}/{m['n_events'].sum()} "
              f"= {m['records'].sum() / m['n_events'].sum():.2f}x; vehicle starts+stops {m['vss'].sum()}/{k_vss.sum()} "
              f"= {m['vss'].sum() / k_vss.sum():.2f}x")


if __name__ == "__main__":
    main()
