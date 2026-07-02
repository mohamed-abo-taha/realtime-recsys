from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from recsys.data.synthetic import generate
from recsys.features.store import ALL_FEATURES, ITEM_FEATURES, USER_FEATURES
from recsys.models.two_tower import TwoTower
from recsys.serving.artifacts import save_artifacts

NUM_USERS = 40
NUM_ITEMS = 25


def build_artifact_bundle(out: Path, with_ranker: bool) -> None:
    """A small untrained-but-valid artifact bundle for engine/API tests."""
    model = TwoTower(NUM_USERS, NUM_ITEMS, embed_dim=8, hidden_dim=16, out_dim=8)
    item_vectors = model.encode_items().numpy()
    user_map = pd.DataFrame(
        {"user_id": np.arange(1, NUM_USERS + 1), "user_idx": np.arange(NUM_USERS)}
    )
    item_map = pd.DataFrame(
        {
            "item_id": np.arange(100, 100 + NUM_ITEMS),
            "item_idx": np.arange(NUM_ITEMS),
            "title": [f"Item {i}" for i in range(NUM_ITEMS)],
        }
    )
    popularity = pd.DataFrame(
        {"item_idx": np.arange(NUM_ITEMS), "count": np.arange(NUM_ITEMS, 0, -1)}
    )
    save_artifacts(out, model, item_vectors, user_map, item_map, popularity)

    if not with_ranker:
        return
    rng = np.random.default_rng(0)
    user_features = pd.DataFrame({"user_idx": np.arange(NUM_USERS)})
    for name in USER_FEATURES:
        user_features[name] = rng.random(NUM_USERS).astype(np.float32)
    item_features = pd.DataFrame({"item_idx": np.arange(NUM_ITEMS)})
    for name in ITEM_FEATURES:
        item_features[name] = rng.random(NUM_ITEMS).astype(np.float32)
    user_features.to_parquet(out / "user_features.parquet", index=False)
    item_features.to_parquet(out / "item_features.parquet", index=False)

    groups, candidates = 30, 10
    X = rng.random((groups * candidates, len(ALL_FEATURES)))
    y = np.tile(np.eye(1, candidates, 0, dtype=np.int8).ravel(), groups)
    ranker = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=5, num_leaves=7, min_child_samples=1, verbosity=-1
    )
    ranker.fit(X, y, group=np.full(groups, candidates), feature_name=ALL_FEATURES)
    ranker.booster_.save_model(str(out / "ranker.txt"))


@pytest.fixture(scope="session")
def synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate(num_users=60, num_items=40, interactions_per_user=15, seed=7)


@pytest.fixture(scope="session")
def artifacts_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("artifacts_two_stage")
    build_artifact_bundle(out, with_ranker=True)
    return out


@pytest.fixture(scope="session")
def retrieval_only_artifacts_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("artifacts_retrieval")
    build_artifact_bundle(out, with_ranker=False)
    return out
