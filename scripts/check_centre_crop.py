"""Open-issue pass item 4 (PROGRESS.md): share of tuning events lost to CLIP's centre-crop (28 % overall, per camera, vehicle start/stop, per type)."""

import argparse
import sqlite3

import open_clip
import pandas as pd
import yaml
from torchvision.transforms import CenterCrop, Resize

from rewind.eval.splits import split_event_ids
from rewind.paths import REPO_ROOT

# Check parameters, not pipeline tunables: an event's class from the mean share of its box area outside the square.
INSIDE_MAX, LOST_MIN = 0.10, 0.90   # inside < 10 %, partial 10-90 %, lost > 90 %


def check_centre_square(model: str, tags: list[str]) -> int:
    """Verify each tag's eval transform is Resize(short side) + CenterCrop(square); return the square side."""
    sides = set()
    for tag in tags:
        _, _, preprocess = open_clip.create_model_and_transforms(model, pretrained=tag, device="cpu")
        resize, crop = preprocess.transforms[:2]
        if not (isinstance(resize, Resize) and isinstance(resize.size, int) and isinstance(crop, CenterCrop)
                and crop.size[0] == crop.size[1] == resize.size):
            raise ValueError(f"{model}/{tag}: preprocessing is not short-side resize + square centre crop: {preprocess}")
        sides.add(resize.size)
        print(f"{model}/{tag}: {preprocess.transforms[:2]}")
    return sides.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="open_clip model name (default: appearance.model in config.yaml)")
    parser.add_argument("--pretrained", nargs="+", required=True,
                        help="open_clip tags whose preprocessing to verify, e.g. laion2b_s34b_b88k")
    args = parser.parse_args()
    model = args.model or yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())["appearance"]["model"]
    check_centre_square(model, args.pretrained)

    conn = sqlite3.connect(REPO_ROOT / "data" / "rewind.sqlite")
    conn.execute("CREATE TEMP TABLE ids (event_id INTEGER PRIMARY KEY)")
    conn.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in split_event_ids(conn, "tuning")])
    b = pd.read_sql("""SELECT e.event_id, e.camera_id, e.activity_type, v.modality, v.width AS W, v.height AS H,
                              x.x1, x.x2
                       FROM ids JOIN events e USING (event_id) JOIN event_boxes x USING (event_id)
                       JOIN videos v USING (video_id)""", conn)
    if (b.W < b.H).any():
        raise ValueError("portrait frames: the kept range below assumes landscape")
    # short side scaled to the square, centre crop: the kept x range is the central H pixels of the width
    lo = (b.W - b.H) / 2
    hi = lo + b.H
    x1, x2 = b.x1.clip(0, b.W), b.x2.clip(0, b.W)          # boxes are stored unclipped; clip to the frame first
    width = (x2 - x1).clip(lower=0)
    inside = (x2.clip(upper=hi) - x1.clip(lower=lo)).clip(lower=0)
    keep = width > 0
    b = b[keep].assign(out=1 - inside[keep] / width[keep])
    print(f"\ntuning: {b.event_id.nunique()} events, {len(b)} box-frames; kept width:",
          {m: round(g.H.iloc[0] / g.W.iloc[0], 3) for m, g in b.groupby("modality")})

    ev = b.groupby(["event_id", "camera_id", "modality", "activity_type"]).out.mean().reset_index()
    ev["cls"] = pd.cut(ev.out, [-0.01, INSIDE_MAX, LOST_MIN, 1.01], labels=["inside", "partial", "lost"])
    print(pd.crosstab([ev.modality, ev.camera_id], ev.cls, margins=True).to_string())
    print((pd.crosstab(ev.camera_id, ev.cls, normalize="index") * 100).round(1).to_string())
    print("overall %:", (ev.cls.value_counts(normalize=True) * 100).round(1).to_dict())
    vss = ev[ev.activity_type.isin(["vehicle_starts", "vehicle_stops"])]
    print("vehicle start/stop:", vss.cls.value_counts().to_dict())
    by_type = (pd.crosstab(ev.activity_type, ev.cls, normalize="index") * 100).round(0)
    print(by_type.sort_values("lost", ascending=False).to_string())


if __name__ == "__main__":
    main()
