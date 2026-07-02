import pandas as pd
import pytest

from recsys.config import PrepareConfig
from recsys.data.prepare import prepare
from recsys.data.synthetic import generate


def test_prepare_on_synthetic_data():
    ratings, items = generate(num_users=80, num_items=50, interactions_per_user=12, seed=3)
    cfg = PrepareConfig(positive_threshold=4.0, min_user_interactions=5, min_item_interactions=3)
    frames = prepare(ratings, cfg, items)

    train, tune, val = frames["train"], frames["tune"], frames["val"]
    # Exactly one val and one tune row per user, and indices are contiguous.
    assert val["user_idx"].is_unique and tune["user_idx"].is_unique
    assert set(val["user_idx"]) == set(range(len(frames["user_map"])))
    assert frames["item_map"]["item_idx"].tolist() == list(range(len(frames["item_map"])))
    assert "title" in frames["item_map"].columns
    assert len(train) > 0


def test_held_out_items_are_chronologically_last():
    ratings = pd.DataFrame(
        {
            "user_id": [1] * 6 + [2] * 6,
            "item_id": [10, 11, 12, 13, 14, 15] * 2,
            "rating": [5.0] * 12,
            "timestamp": list(range(6)) + list(range(10, 16)),
        }
    )
    cfg = PrepareConfig(min_user_interactions=2, min_item_interactions=1)
    frames = prepare(ratings, cfg)
    item_map = frames["item_map"].set_index("item_idx")["item_id"]
    assert frames["val"]["item_idx"].map(item_map).tolist() == [15, 15]  # last
    assert frames["tune"]["item_idx"].map(item_map).tolist() == [14, 14]  # second-to-last


def test_low_ratings_are_dropped():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "item_id": [10, 11, 12, 13],
            "rating": [5.0, 5.0, 5.0, 1.0],
            "timestamp": [1, 2, 3, 4],
        }
    )
    cfg = PrepareConfig(min_user_interactions=1, min_item_interactions=1)
    frames = prepare(ratings, cfg)
    assert len(frames["item_map"]) == 3  # the 1.0-rated item never appears


def test_everything_filtered_raises():
    ratings = pd.DataFrame(
        {"user_id": [1], "item_id": [10], "rating": [5.0], "timestamp": [1]}
    )
    with pytest.raises(ValueError, match="no interactions left"):
        prepare(ratings, PrepareConfig(min_user_interactions=99))
