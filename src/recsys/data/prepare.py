"""Turn raw ratings into implicit-feedback training data.

Steps: threshold ratings into positives, drop sparse users/items, map raw ids
to contiguous indices, then hold out each user's last two interactions (by
timestamp): the last for final validation, the second-to-last for training
the ranking stage. The three-way split keeps ranker labels temporally
disjoint from the interactions the towers were trained on.

Outputs under data/processed/:
    train.parquet         user_idx, item_idx — trains the two-tower model
    tune.parquet          user_idx, item_idx — labels for the ranking stage
    val.parquet           user_idx, item_idx — final evaluation only
    user_map.parquet      user_id -> user_idx
    item_map.parquet      item_id -> item_idx (+ title if metadata available)
    popularity.parquet    item_idx ordered by train interaction count
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recsys.config import PrepareConfig, data_dir

RAW_COLUMNS = {"user_id", "item_id", "rating", "timestamp"}


def prepare(
    ratings: pd.DataFrame,
    cfg: PrepareConfig = PrepareConfig(),
    items: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    missing = RAW_COLUMNS - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings frame missing columns: {sorted(missing)}")

    pos = ratings[ratings["rating"] >= cfg.positive_threshold].copy()

    # Iteratively drop sparse users/items until both constraints hold.
    while True:
        user_counts = pos["user_id"].value_counts()
        item_counts = pos["item_id"].value_counts()
        keep = pos["user_id"].map(user_counts).ge(cfg.min_user_interactions) & pos[
            "item_id"
        ].map(item_counts).ge(cfg.min_item_interactions)
        if keep.all():
            break
        pos = pos[keep]
    if pos.empty:
        raise ValueError("no interactions left after filtering — lower the thresholds")

    user_ids = pos["user_id"].drop_duplicates().sort_values().reset_index(drop=True)
    item_ids = pos["item_id"].drop_duplicates().sort_values().reset_index(drop=True)
    user_map = pd.DataFrame({"user_id": user_ids, "user_idx": range(len(user_ids))})
    item_map = pd.DataFrame({"item_id": item_ids, "item_idx": range(len(item_ids))})
    if items is not None and "title" in items.columns:
        item_map = item_map.merge(items[["item_id", "title"]], on="item_id", how="left")

    pos = pos.merge(user_map, on="user_id").merge(item_map, on="item_id")

    # Leave-last-out, twice: last positive -> val, second-to-last -> tune.
    pos = pos.sort_values(["user_idx", "timestamp"], kind="stable")
    last = pos.groupby("user_idx").tail(1)
    rest = pos.drop(last.index)
    second_last = rest.groupby("user_idx").tail(1)
    train = rest.drop(second_last.index)[["user_idx", "item_idx"]].reset_index(drop=True)
    tune = second_last[["user_idx", "item_idx"]].reset_index(drop=True)
    val = last[["user_idx", "item_idx"]].reset_index(drop=True)

    popularity = (
        train["item_idx"].value_counts().rename_axis("item_idx").reset_index(name="count")
    )
    return {
        "train": train,
        "tune": tune,
        "val": val,
        "user_map": user_map,
        "item_map": item_map,
        "popularity": popularity,
    }


def run(raw_path: Path, out_dir: Path, cfg: PrepareConfig, items_path: Path | None = None) -> None:
    ratings = pd.read_parquet(raw_path) if raw_path.suffix == ".parquet" else pd.read_csv(raw_path)
    items = None
    if items_path is not None and items_path.exists():
        items = (
            pd.read_parquet(items_path)
            if items_path.suffix == ".parquet"
            else pd.read_csv(items_path)
        )
    frames = prepare(ratings, cfg, items)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
    print(
        f"prepared: {len(frames['train'])} train / {len(frames['tune'])} tune / "
        f"{len(frames['val'])} val interactions, "
        f"{len(frames['user_map'])} users, {len(frames['item_map'])} items -> {out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=data_dir() / "raw" / "ratings.parquet")
    parser.add_argument("--items", type=Path, default=data_dir() / "raw" / "items.parquet")
    parser.add_argument("--out", type=Path, default=data_dir() / "processed")
    parser.add_argument("--positive-threshold", type=float, default=PrepareConfig.positive_threshold)
    parser.add_argument("--min-user", type=int, default=PrepareConfig.min_user_interactions)
    parser.add_argument("--min-item", type=int, default=PrepareConfig.min_item_interactions)
    args = parser.parse_args()
    cfg = PrepareConfig(args.positive_threshold, args.min_user, args.min_item)
    run(args.raw, args.out, cfg, args.items)


if __name__ == "__main__":
    main()
