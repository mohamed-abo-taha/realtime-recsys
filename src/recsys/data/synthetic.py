"""Synthetic clustered interactions for offline smoke tests.

Users and items are assigned to latent groups; users mostly interact with
items from their own group, so a working retrieval model beats the popularity
baseline. Lets the full pipeline (prepare -> train -> index -> serve) run end
to end with zero downloads. Same schema as the MovieLens raw data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import data_dir


def generate(
    num_users: int = 500,
    num_items: int = 300,
    num_groups: int = 5,
    interactions_per_user: int = 30,
    in_group_prob: float = 0.85,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    user_group = rng.integers(0, num_groups, num_users)
    item_group = rng.integers(0, num_groups, num_items)
    items_by_group = [np.flatnonzero(item_group == g) for g in range(num_groups)]

    rows = []
    ts = 0
    for u in range(num_users):
        own = items_by_group[user_group[u]]
        for _ in range(interactions_per_user):
            if len(own) and rng.random() < in_group_prob:
                item = int(rng.choice(own))
            else:
                item = int(rng.integers(0, num_items))
            ts += 1
            rows.append((u + 1, item + 1, 5.0, ts))

    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    ratings = ratings.drop_duplicates(["user_id", "item_id"], keep="last")
    items = pd.DataFrame(
        {
            "item_id": np.arange(1, num_items + 1),
            "title": [f"Item {i} (group {g})" for i, g in enumerate(item_group, start=1)],
        }
    )
    return ratings, items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=data_dir() / "raw")
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--items", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratings, items = generate(num_users=args.users, num_items=args.items, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(args.out / "ratings.parquet", index=False)
    items.to_parquet(args.out / "items.parquet", index=False)
    print(f"wrote {len(ratings)} synthetic interactions -> {args.out}")


if __name__ == "__main__":
    main()
