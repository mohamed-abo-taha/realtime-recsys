"""Feature store: one feature definition, an offline build, an online store.

The point of this module is train/serve consistency. Features are defined
once (`USER_FEATURES` / `ITEM_FEATURES` / `PAIR_FEATURES`) and computed once
(`build_features`, offline, from the processed interactions). The ranker
trains on the offline Parquet output; serving reads the *same rows* through
an online store — Redis when available (`REDIS_URL`), in-process otherwise.
The materialize step copies offline -> online, exactly the Feast
offline/online mental model, kept dependency-light. Swapping this module for
Feast is a mechanical change, discussed in the README.

Feature definitions (order matters — the ranker consumes this order):
    user:  user_activity_log    log1p(#train interactions)
    item:  item_popularity_log  log1p(#train interactions)
           item_year            release year parsed from the title (NaN-safe)
    pair:  two_tower_score      retrieval dot product, computed at request time
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

USER_FEATURES = ["user_activity_log"]
ITEM_FEATURES = ["item_popularity_log", "item_year"]
PAIR_FEATURES = ["two_tower_score"]
ALL_FEATURES = USER_FEATURES + ITEM_FEATURES + PAIR_FEATURES


def build_features(processed_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Offline build: processed interactions -> user/item feature tables."""
    train = pd.read_parquet(processed_dir / "train.parquet")
    user_map = pd.read_parquet(processed_dir / "user_map.parquet")
    item_map = pd.read_parquet(processed_dir / "item_map.parquet")

    user_counts = train["user_idx"].value_counts()
    user_features = pd.DataFrame({"user_idx": user_map["user_idx"]})
    user_features["user_activity_log"] = np.log1p(
        user_features["user_idx"].map(user_counts).fillna(0)
    ).astype(np.float32)

    item_counts = train["item_idx"].value_counts()
    item_features = pd.DataFrame({"item_idx": item_map["item_idx"]})
    item_features["item_popularity_log"] = np.log1p(
        item_features["item_idx"].map(item_counts).fillna(0)
    ).astype(np.float32)
    if "title" in item_map.columns:
        year = item_map["title"].astype(str).str.extract(r"\((\d{4})\)")[0]
        item_features["item_year"] = pd.to_numeric(year, errors="coerce").astype(np.float32)
    else:
        item_features["item_year"] = np.float32(np.nan)

    out_dir.mkdir(parents=True, exist_ok=True)
    user_features.to_parquet(out_dir / "user_features.parquet", index=False)
    item_features.to_parquet(out_dir / "item_features.parquet", index=False)
    (out_dir / "feature_meta.json").write_text(
        json.dumps(
            {"user": USER_FEATURES, "item": ITEM_FEATURES, "pair": PAIR_FEATURES}, indent=2
        )
    )
    return user_features, item_features


class LocalOnlineStore:
    """In-process online store: feature tables held as dense numpy arrays.

    Item features stay in memory in every deployment (the catalogue is static
    between offline runs); user features go to Redis when it is configured.
    """

    def __init__(self, artifacts_dir: Path):
        user_features = pd.read_parquet(artifacts_dir / "user_features.parquet")
        item_features = pd.read_parquet(artifacts_dir / "item_features.parquet")
        self.user_matrix = (
            user_features.sort_values("user_idx")[USER_FEATURES].to_numpy(np.float32)
        )
        self.item_matrix = (
            item_features.sort_values("item_idx")[ITEM_FEATURES].to_numpy(np.float32)
        )

    def get_user_features(self, user_idx: int) -> np.ndarray:
        return self.user_matrix[user_idx]

    def get_item_features(self, item_idx: np.ndarray) -> np.ndarray:
        return self.item_matrix[item_idx]


class RedisOnlineStore(LocalOnlineStore):
    """User features served from Redis; item features from process memory.

    Keys: ``user_feat:{user_idx}`` -> float32 bytes in USER_FEATURES order.
    """

    def __init__(self, artifacts_dir: Path, url: str):
        super().__init__(artifacts_dir)
        import redis

        self.client = redis.Redis.from_url(url)
        self.client.ping()

    def get_user_features(self, user_idx: int) -> np.ndarray:
        raw = self.client.get(f"user_feat:{user_idx}")
        if raw is None:  # not materialized -> behave like a fresh user
            return np.zeros(len(USER_FEATURES), dtype=np.float32)
        return np.frombuffer(raw, dtype=np.float32)

    def materialize(self, batch_size: int = 10_000) -> int:
        """Copy the offline user features into Redis."""
        pipe = self.client.pipeline(transaction=False)
        for user_idx, row in enumerate(self.user_matrix):
            pipe.set(f"user_feat:{user_idx}", row.tobytes())
            if (user_idx + 1) % batch_size == 0:
                pipe.execute()
        pipe.execute()
        return len(self.user_matrix)


def open_online_store(artifacts_dir: Path, redis_url: str | None) -> LocalOnlineStore:
    if redis_url:
        return RedisOnlineStore(artifacts_dir, redis_url)
    return LocalOnlineStore(artifacts_dir)
