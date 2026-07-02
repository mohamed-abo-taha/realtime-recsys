import numpy as np
import pytest

from recsys.eval.recall import recall_at_k


def test_recall_counts_hits():
    retrieved = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    targets = np.array([2, 99, 9])
    assert recall_at_k(retrieved, targets) == pytest.approx(2 / 3)


def test_recall_length_mismatch_raises():
    with pytest.raises(ValueError):
        recall_at_k(np.zeros((2, 3)), np.zeros(3))
