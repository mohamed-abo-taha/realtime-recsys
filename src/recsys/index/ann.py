"""Approximate-nearest-neighbour index over L2-normalized item vectors.

Backend is pluggable: FAISS (HNSW) when installed, otherwise an exact numpy
inner-product search. Vectors are normalized, so inner product == cosine.
At MovieLens scale (~10^4-10^5 items) exact search is already millisecond-
fast; FAISS is what makes the story hold at millions of items.

Vectors are persisted as .npy and the index is rebuilt at load time — for
catalogues this size a rebuild takes seconds and avoids fragile serialized
index formats.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "faiss" if HAS_FAISS else "numpy"
    if backend == "faiss" and not HAS_FAISS:
        raise RuntimeError("faiss backend requested but faiss is not installed")
    return backend


class AnnIndex:
    def __init__(self, vectors: np.ndarray, backend: str = "auto", hnsw_m: int = 32):
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-D (num_items, dim)")
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.backend = _resolve_backend(backend)
        self._faiss_index = None
        if self.backend == "faiss":
            index = faiss.IndexHNSWFlat(self.vectors.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.add(self.vectors)
            self._faiss_index = index

    def __len__(self) -> int:
        return len(self.vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices), each (num_queries, k), best first."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        k = min(k, len(self.vectors))
        if self.backend == "faiss":
            return self._faiss_index.search(queries, k)
        scores = queries @ self.vectors.T
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        row = np.arange(len(queries))[:, None]
        order = np.argsort(-scores[row, top], axis=1)
        idx = top[row, order]
        return scores[row, idx], idx

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "item_vectors.npy", self.vectors)
        (path / "index_meta.json").write_text(
            json.dumps({"backend": self.backend, "num_items": len(self.vectors)})
        )

    @classmethod
    def load(cls, path: Path, backend: str = "auto") -> AnnIndex:
        vectors = np.load(path / "item_vectors.npy")
        return cls(vectors, backend=backend)
