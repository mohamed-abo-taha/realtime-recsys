import numpy as np
import pytest

from recsys.index.ann import AnnIndex


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_self_is_nearest_neighbour():
    vectors = unit_vectors(100, 16)
    index = AnnIndex(vectors, backend="numpy")
    scores, idx = index.search(vectors[:5], k=1)
    assert list(idx[:, 0]) == [0, 1, 2, 3, 4]
    assert np.allclose(scores[:, 0], 1.0, atol=1e-5)


def test_results_sorted_and_k_clamped():
    vectors = unit_vectors(10, 8)
    index = AnnIndex(vectors, backend="numpy")
    scores, idx = index.search(vectors[0], k=50)  # k > num_items
    assert idx.shape == (1, 10)
    assert np.all(np.diff(scores[0]) <= 1e-6)  # descending


def test_matches_exact_brute_force():
    vectors = unit_vectors(200, 12, seed=1)
    queries = unit_vectors(7, 12, seed=2)
    index = AnnIndex(vectors, backend="numpy")
    _, idx = index.search(queries, k=5)
    exact = np.argsort(-(queries @ vectors.T), axis=1)[:, :5]
    assert np.array_equal(idx, exact)


def test_save_load_roundtrip(tmp_path):
    vectors = unit_vectors(50, 8)
    AnnIndex(vectors, backend="numpy").save(tmp_path)
    loaded = AnnIndex.load(tmp_path, backend="numpy")
    assert np.array_equal(loaded.vectors, vectors)


def test_faiss_backend_requires_faiss_if_forced():
    pytest.importorskip("faiss", reason="faiss not installed")
    vectors = unit_vectors(100, 16)
    index = AnnIndex(vectors, backend="faiss")
    _, idx = index.search(vectors[:3], k=1)
    assert list(idx[:, 0]) == [0, 1, 2]
