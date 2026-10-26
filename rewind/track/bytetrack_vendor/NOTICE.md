# Vendored third-party code: ByteTrack (tracker only)

- **Source:** https://github.com/FoundationVision/ByteTrack (formerly `ifzhang/ByteTrack`; GitHub redirects) at commit
  `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, files `yolox/tracker/{byte_tracker,basetrack,kalman_filter,matching}.py`.
  **Copyright** (c) 2021 Yifu Zhang. **License:** MIT (full text in `LICENSE`, byte-identical to upstream `LICENSE`
  at that commit; GitHub's license API also reports MIT). Paper: Zhang et al., *ByteTrack*, ECCV 2022.
- **Unchanged:** `kalman_filter.py` (byte-identical; needs scipy).
- **Changes by Rewind** (each file that changed starts with a provenance comment; changed lines say "Rewind"):
  1. `byte_tracker.py`: unused imports removed (`torch`, `os`, `copy`, `deque`); `np.float` -> `float` (removed in
     NumPy 2); `from yolox.tracker import matching` -> relative import; `update()` takes an (N, 5) array of x1, y1, x2,
     y2, score in original pixels (the YOLOX-output branch and the rescaling by `img_info` / `img_size` removed);
     track ids come from a per-tracker counter `BYTETracker.n_ids` passed to `STrack.activate` (was the global
     `BaseTrack._count`, shared by every tracker in the process); `re_activate(new_id=True)` raises (ByteTrack never
     calls it); each `STrack` keeps `det_tlbr`, the box of the detection last matched to it (Rewind stores detector
     boxes, not Kalman boxes). No association, Kalman or threshold logic changed.
  2. `basetrack.py`: the global `_count` and `next_id()` removed (see 1).
  3. `matching.py`: trimmed to the four functions `BYTETracker` uses (`linear_assignment`, `ious`, `iou_distance`,
     `fuse_score`); `lap.lapjv(extend_cost=True, cost_limit=thresh)` -> `scipy.optimize.linear_sum_assignment` on an
     extended matrix with the same objective (checked against lap 0.4.0's `_lapjv.pyx`; equal up to ties, see the
     comment in the code and `tests/test_bytetrack.py`); `cython_bbox.bbox_overlaps` -> the same formula in NumPy
     (+1 pixel convention kept).
  `__init__.py` and this file are new.
- **Why vendored:** the reference repo is not a package (its tracker lives inside its own YOLOX fork) and needs `lap`
  and `cython_bbox` (compiled); Ultralytics' copy is AGPL-3.0 (avoided at Gate 3.3).
- **Use in Rewind** (CLAUDE.md 1.2): one tracker per clip, within one camera; tracks are motion evidence only, never
  identities, never linked across clips or cameras.
