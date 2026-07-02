"""Online path: user_id -> top-K, with per-stage timings.

Stages mirror the architecture diagram and are timed individually, because
the per-stage latency breakdown is the number this project exists to produce:

    feature_fetch   online feature store (Redis or in-process)
    encode          user-tower forward pass (CPU)
    ann             ANN candidate retrieval
    rank            LightGBM scoring of the candidate set (when trained)

If no ranker artifact exists the engine serves retrieval-only (Phase 1
behaviour). Unknown users get the popularity fallback instead of a 500.
An optional cache keeps hot users' vectors+features in memory — user
embeddings are static between offline runs, so this is safe and is the
before/after comparison for the load test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.features.store import LocalOnlineStore, open_online_store
from recsys.index.ann import AnnIndex
from recsys.serving.artifacts import load_model

STAGES = ("feature_fetch", "encode", "ann", "rank")


@dataclass
class Recommendation:
    item_id: int
    score: float
    title: str | None = None


@dataclass
class RetrievalResult:
    user_id: int
    cold_start: bool
    items: list[Recommendation]
    timings_ms: dict[str, float] = field(default_factory=dict)


def _timings(**kwargs: float) -> dict[str, float]:
    out = {stage: round(kwargs.get(stage, 0.0), 3) for stage in STAGES}
    out["total"] = round(kwargs.get("total", 0.0), 3)
    return out


class RetrievalEngine:
    def __init__(
        self,
        artifacts_dir: Path,
        index_backend: str = "auto",
        redis_url: str | None = None,
        retrieve_k: int = 200,
        enable_cache: bool = True,
    ):
        self.model = load_model(artifacts_dir)
        self.index = AnnIndex(np.load(artifacts_dir / "item_vectors.npy"), backend=index_backend)
        self.retrieve_k = retrieve_k
        self.enable_cache = enable_cache
        self._user_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        user_map = pd.read_parquet(artifacts_dir / "user_map.parquet")
        item_map = pd.read_parquet(artifacts_dir / "item_map.parquet")
        popularity = pd.read_parquet(artifacts_dir / "popularity.parquet")

        self.user_to_idx: dict[int, int] = dict(
            zip(user_map["user_id"].astype(int), user_map["user_idx"].astype(int), strict=True)
        )
        self.idx_to_item: np.ndarray = item_map.sort_values("item_idx")["item_id"].to_numpy()
        self.titles: np.ndarray | None = (
            item_map.sort_values("item_idx")["title"].to_numpy()
            if "title" in item_map.columns
            else None
        )
        self.popular_idx: np.ndarray = popularity["item_idx"].to_numpy()

        # Phase 2/3 artifacts are optional: engine degrades to retrieval-only.
        self.store: LocalOnlineStore | None = None
        self.ranker = None
        if (artifacts_dir / "user_features.parquet").exists():
            self.store = open_online_store(artifacts_dir, redis_url)
        if (artifacts_dir / "ranker.txt").exists() and self.store is not None:
            import lightgbm as lgb

            self.ranker = lgb.Booster(model_file=str(artifacts_dir / "ranker.txt"))

    @property
    def num_items(self) -> int:
        return len(self.index)

    @property
    def two_stage(self) -> bool:
        return self.ranker is not None

    def _to_recommendations(self, idx: np.ndarray, scores: np.ndarray) -> list[Recommendation]:
        return [
            Recommendation(
                item_id=int(self.idx_to_item[i]),
                score=round(float(s), 6),
                title=str(self.titles[i]) if self.titles is not None else None,
            )
            for i, s in zip(idx, scores, strict=True)
        ]

    def _user_state(self, user_idx: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Returns (user_features, user_vector, feature_fetch_ms, encode_ms)."""
        if self.enable_cache and user_idx in self._user_cache:
            feats, vec = self._user_cache[user_idx]
            return feats, vec, 0.0, 0.0
        t0 = time.perf_counter()
        feats = (
            self.store.get_user_features(user_idx)
            if self.store is not None
            else np.empty(0, dtype=np.float32)
        )
        t1 = time.perf_counter()
        with torch.no_grad():
            vec = self.model.encode_users(torch.tensor([user_idx])).numpy()
        t2 = time.perf_counter()
        if self.enable_cache:
            self._user_cache[user_idx] = (feats, vec)
        return feats, vec, (t1 - t0) * 1000, (t2 - t1) * 1000

    def recommend(self, user_id: int, k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        user_idx = self.user_to_idx.get(user_id)

        if user_idx is None:  # cold start: popularity fallback, no model in the path
            idx = self.popular_idx[:k]
            scores = np.linspace(1.0, 0.0, num=len(idx), endpoint=False)
            items = self._to_recommendations(idx, scores)
            total = (time.perf_counter() - t0) * 1000
            return RetrievalResult(user_id, True, items, _timings(total=total))

        feats, user_vec, fetch_ms, encode_ms = self._user_state(user_idx)

        n = max(self.retrieve_k, k) if self.two_stage else k
        t1 = time.perf_counter()
        ann_scores, cand = self.index.search(user_vec, n)
        t2 = time.perf_counter()

        if self.two_stage:
            item_feats = self.store.get_item_features(cand[0])
            user_block = np.broadcast_to(feats, (len(cand[0]), len(feats)))
            X = np.hstack([user_block, item_feats, ann_scores[0].reshape(-1, 1)])
            pred = self.ranker.predict(X, num_threads=1)
            order = np.argsort(-pred)[:k]
            idx, scores = cand[0][order], pred[order]
        else:
            idx, scores = cand[0][:k], ann_scores[0][:k]
        t3 = time.perf_counter()

        items = self._to_recommendations(idx, scores)
        return RetrievalResult(
            user_id,
            False,
            items,
            _timings(
                feature_fetch=fetch_ms,
                encode=encode_ms,
                ann=(t2 - t1) * 1000,
                rank=(t3 - t2) * 1000 if self.two_stage else 0.0,
                total=(time.perf_counter() - t0) * 1000,
            ),
        )
