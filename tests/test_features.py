import numpy as np
import pytest

from recsys.config import PrepareConfig
from recsys.data.prepare import prepare
from recsys.eval.ranking import ndcg_at_k
from recsys.features.store import ITEM_FEATURES, USER_FEATURES, LocalOnlineStore, build_features


@pytest.fixture()
def processed_dir(synthetic_frames, tmp_path):
    ratings, items = synthetic_frames
    cfg = PrepareConfig(min_user_interactions=5, min_item_interactions=2)
    frames = prepare(ratings, cfg, items)
    for name, frame in frames.items():
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    return tmp_path


def test_build_features_shapes_and_columns(processed_dir, tmp_path):
    out = tmp_path / "feats"
    user_features, item_features = build_features(processed_dir, out)
    assert list(user_features.columns) == ["user_idx"] + USER_FEATURES
    assert list(item_features.columns) == ["item_idx"] + ITEM_FEATURES
    assert (user_features["user_activity_log"] >= 0).all()
    assert (out / "feature_meta.json").exists()


def test_online_store_matches_offline_tables(processed_dir, tmp_path):
    out = tmp_path / "feats"
    user_features, item_features = build_features(processed_dir, out)
    store = LocalOnlineStore(out)
    # Same rows through the online read path as in the offline table.
    row = user_features.sort_values("user_idx").iloc[3][USER_FEATURES].to_numpy(np.float32)
    assert np.allclose(store.get_user_features(3), row)
    items = store.get_item_features(np.array([0, 2]))
    assert items.shape == (2, len(ITEM_FEATURES))


def test_ndcg_at_k():
    ranked = np.array([[5, 1, 2], [9, 8, 7], [3, 4, 6]])
    targets = np.array([5, 7, 99])
    # hits at rank 1 (gain 1.0), rank 3 (1/log2(4)=0.5), and a miss
    assert ndcg_at_k(ranked, targets, k=3) == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert ndcg_at_k(ranked, targets, k=2) == pytest.approx(1.0 / 3)


def test_ndcg_length_mismatch_raises():
    with pytest.raises(ValueError):
        ndcg_at_k(np.zeros((2, 3)), np.zeros(3), k=2)
