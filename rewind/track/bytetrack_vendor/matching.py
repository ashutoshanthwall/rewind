# Vendored from ByteTrack yolox/tracker/matching.py (MIT, see LICENSE). MODIFIED by Rewind (see NOTICE.md): only the
# four functions BYTETracker uses are kept; lap.lapjv is replaced by scipy's linear_sum_assignment on the same extended
# cost matrix lap builds for cost_limit, and cython_bbox.bbox_overlaps by a NumPy copy of its formula.
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    # Rewind: replaces lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh). lap 0.4.0 pads to 2N x 2N
    # (N = max(n, m)) with thresh for "unmatched" entries; in its optimum each matched pair changes the total by
    # (cost - thresh). Here: an (n+m) x (n+m) matrix whose dummy entries cost thresh / 2 (dummy-dummy 0), so leaving a
    # row and a column unmatched costs thresh: the same objective, minimise sum over matches of (cost - thresh), hence
    # the same optimum up to ties (tests/test_bytetrack.py checks it against brute force).
    n, m = cost_matrix.shape
    ext = np.full((n + m, n + m), thresh / 2.0)
    ext[n:, m:] = 0.0
    ext[:n, :m] = cost_matrix
    rows, cols = linear_sum_assignment(ext)
    x = np.full(n, -1, dtype=int)
    y = np.full(m, -1, dtype=int)
    for r, c in zip(rows, cols):
        if r < n and c < m:
            x[r], y[c] = c, r
    matches = np.asarray([[ix, mx] for ix, mx in enumerate(x) if mx >= 0], dtype=int).reshape(-1, 2)
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    # Rewind: cython_bbox.bbox_overlaps, same formula (+1 pixel convention: a box from x1 to x2 is x2 - x1 + 1 wide)
    a = np.ascontiguousarray(atlbrs, dtype=float)
    b = np.ascontiguousarray(btlbrs, dtype=float)
    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    ious = inter / (area_a[:, None] + area_b[None, :] - inter)

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
