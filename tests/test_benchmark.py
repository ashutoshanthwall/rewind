"""Step 7.2 benchmark metrics (scripts/benchmark_engine.py): recall@k and the NumPy exact top-k. Hand-computed."""

import numpy as np
import pytest

from scripts.benchmark_engine import numpy_topk, recall


def test_recall_hand_computed():
    # query 1: {1, 2, 3} vs {1, 2, 4} -> 2/3; query 2: {5, 6} vs {6, 5} -> 2/2 (order ignored); mean 5/6
    assert recall([[1, 2, 3], [5, 6]], [[1, 2, 4], [6, 5]], 3) == pytest.approx(5 / 6)
    # only the first k of each list count: top-2 {1, 9} vs {1, 2} -> 1/2
    assert recall([[1, 9, 2]], [[1, 2, 3]], 2) == pytest.approx(0.5)
    # fewer exact results than k (filtered search): divide by |exact| = 2, both found -> 1
    assert recall([[7, 8]], [[8, 7]], 200) == 1.0


def test_recall_refuses_empty_exact():
    with pytest.raises(ValueError, match="empty"):
        recall([[1]], [[]], 5)


def test_numpy_topk_hand_computed():
    # rows . q with q = (1, 0): scores 0.1, 0.9, -0.5, 0.7, 0.9 -> descending, ties by row: 1, 4, 3, 0, 2
    x = np.array([[0.1, 0], [0.9, 0], [-0.5, 0], [0.7, 0], [0.9, 0]], np.float32)
    run = numpy_topk(x, 3)
    top = run(np.array([1, 0], np.float32))
    assert top[2] == 3 and set(top[:2]) == {1, 4}
    assert numpy_topk(x, 5)(np.array([1, 0], np.float32))[2:] == [3, 0, 2]
