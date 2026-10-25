"""Open-issue pass item 2 (PROGRESS.md): camera-view stability numbers (consecutive-clip and vs-first-clip shifts per camera, G340/G336 changes)."""

import argparse
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rewind.paths import REPO_ROOT, display_path

# Check parameters, not pipeline tunables (nothing downstream reads them); changing them changes the PROGRESS numbers.
FRAME_S = 60.0          # compare the proxy frame at this second of every clip
WIDTH = 640             # frames scaled to this width before matching; displacements are in px at this width
SIFT_FEATURES = 4000
RATIO = 0.75            # Lowe's ratio test
RANSAC_PX = 3.0
FLAG_DISP_PX = 5.0      # consecutive pair flagged if mean corner displacement exceeds this ...
FLAG_MIN_INLIERS = 15   # ... or the homography has fewer inliers than this


def grab(path: Path, fps: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(FRAME_S * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame at {FRAME_S} s from {path}")
    return cv2.resize(frame, (WIDTH, round(WIDTH * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)


def features(frame: np.ndarray, sift) -> tuple:
    # CLAHE evens out lighting between dates so matches follow structure, not brightness
    grey = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return sift.detectAndCompute(grey, None)


def compare(a: tuple, b: tuple, shape: tuple, matcher) -> tuple[int, float]:
    """(RANSAC inliers, mean displacement of the 4 image corners under the a->b homography; nan if none found)."""
    (ka, da), (kb, db) = a, b
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return 0, float("nan")
    good = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < RATIO * n.distance]
    if len(good) < 8:
        return 0, float("nan")
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, RANSAC_PX)
    if H is None:
        return 0, float("nan")
    h, w = shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return int(mask.sum()), float(np.linalg.norm(cv2.perspectiveTransform(corners, H) - corners, axis=2).mean())


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="folder for camera_views.csv and the contact sheets")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(REPO_ROOT / "data" / "rewind.sqlite")
    rows = conn.execute("SELECT v.camera_id, v.clip, v.start_time, p.path, p.fps FROM videos v "
                        "JOIN proxies p USING (video_id) ORDER BY v.camera_id, v.start_time").fetchall()
    if not rows:
        raise ValueError("no proxies in the manifest: has scripts.make_proxies been run?")
    sift, matcher = cv2.SIFT_create(SIFT_FEATURES), cv2.BFMatcher()

    records, sheet_rows, flagged = [], [], []
    for cam in sorted({r[0] for r in rows}):
        clips = [r for r in rows if r[0] == cam]
        ref = prev = None
        firsts = {}
        for _, clip, start, path, fps in clips:
            frame = grab(REPO_ROOT / path, fps)
            feats = features(frame, sift)
            ref = ref or (clip, frame, feats)
            ref_inl, ref_disp = compare(ref[2], feats, frame.shape, matcher)
            prev_inl, prev_disp = compare(prev[2], feats, frame.shape, matcher) if prev else (None, None)
            records.append(dict(camera=cam, clip=clip, date=start[:10], ref_inliers=ref_inl, ref_disp_px=ref_disp,
                                prev_inliers=prev_inl, prev_disp_px=prev_disp))
            if prev and (not prev_disp <= FLAG_DISP_PX or prev_inl < FLAG_MIN_INLIERS):  # nan counts as flagged
                flagged.append(np.hstack([label(prev[1], prev[0]), label(frame, clip)]))
            firsts.setdefault(start[:10], label(frame, f"{cam} {start}"))
            prev = (clip, frame, feats)
        sheet_rows.append(np.hstack([cv2.resize(f, (WIDTH // 2, f.shape[0] // 2)) for f in firsts.values()]))

    d = pd.DataFrame(records)
    d.to_csv(args.out / "camera_views.csv", index=False)
    widest = max(r.shape[1] for r in sheet_rows)
    sheet = np.vstack([np.pad(r, ((0, 0), (0, widest - r.shape[1]), (0, 0))) for r in sheet_rows])
    cv2.imwrite(str(args.out / "camera_views_sheet.jpg"), sheet)
    for i, pair in enumerate(flagged):
        cv2.imwrite(str(args.out / f"flagged_{i:02d}.jpg"), pair)

    print(f"{len(d)} clips, frame at {FRAME_S:.0f} s, displacements in px at {WIDTH} px wide\n")
    print("vs the camera's first clip:")
    print(d.groupby(["camera", "date"]).agg(n=("clip", "size"), inliers_min=("ref_inliers", "min"),
                                            disp_med=("ref_disp_px", "median"), disp_max=("ref_disp_px", "max"))
          .round(1).to_string())
    pairs = d.dropna(subset=["prev_inliers"])
    print("\nvs the previous clip of the same camera:")
    print(pairs.groupby("camera").agg(pairs=("clip", "size"), inliers_med=("prev_inliers", "median"),
                                      disp_med=("prev_disp_px", "median")).round(1).to_string())
    bad = pairs[~(pairs.prev_disp_px <= FLAG_DISP_PX) | (pairs.prev_inliers < FLAG_MIN_INLIERS)]
    print(f"\nflagged consecutive pairs (disp > {FLAG_DISP_PX} px or < {FLAG_MIN_INLIERS} inliers): {len(bad)}")
    print(bad[["camera", "clip", "prev_inliers", "prev_disp_px", "ref_disp_px"]].round(1).to_string(index=False))
    print(f"\nwrote {display_path(args.out)}/: camera_views.csv, camera_views_sheet.jpg, {len(flagged)} flagged_*.jpg")


if __name__ == "__main__":
    main()
