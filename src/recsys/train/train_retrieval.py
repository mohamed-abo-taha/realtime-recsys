"""Train the two-tower retrieval model and write serving artifacts.

Offline path, step 1+2 of the architecture diagram: interactions -> trained
towers -> precomputed item embeddings. The ANN index itself is rebuilt from
item_vectors.npy at serve time (see recsys.index.ann).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.config import TrainConfig, artifacts_dir, data_dir
from recsys.eval.recall import recall_at_k
from recsys.index.ann import AnnIndex
from recsys.models.two_tower import TwoTower
from recsys.serving.artifacts import save_artifacts


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train(
    train_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    cfg: TrainConfig,
) -> TwoTower:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = TwoTower(
        num_users, num_items, cfg.embed_dim, cfg.hidden_dim, cfg.out_dim, cfg.temperature
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    users = torch.as_tensor(train_df["user_idx"].to_numpy(), dtype=torch.long)
    items = torch.as_tensor(train_df["item_idx"].to_numpy(), dtype=torch.long)
    n = len(users)

    # logQ correction: log sampling probability of each item under the
    # empirical interaction distribution (what in-batch sampling draws from).
    counts = torch.bincount(items, minlength=num_items).float()
    log_q = torch.log(counts.clamp(min=1.0) / n).to(device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total, batches = 0.0, 0
        start = time.perf_counter()
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            if len(idx) < 2:  # in-batch negatives need at least 2 rows
                continue
            batch_items = items[idx].to(device)
            loss = model.in_batch_loss(
                users[idx].to(device), batch_items, item_log_q=log_q[batch_items]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        print(
            f"epoch {epoch}/{cfg.epochs} loss={total / max(batches, 1):.4f} "
            f"({time.perf_counter() - start:.1f}s, device={device})"
        )
    return model.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=data_dir() / "processed")
    parser.add_argument("--out", type=Path, default=artifacts_dir())
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--dim", type=int, default=TrainConfig.out_dim)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--eval-k", type=int, default=100)
    args = parser.parse_args()

    train_df = pd.read_parquet(args.data / "train.parquet")
    val_df = pd.read_parquet(args.data / "val.parquet")
    user_map = pd.read_parquet(args.data / "user_map.parquet")
    item_map = pd.read_parquet(args.data / "item_map.parquet")
    popularity = pd.read_parquet(args.data / "popularity.parquet")

    cfg = TrainConfig(
        out_dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    model = train(train_df, len(user_map), len(item_map), cfg)
    item_vectors = model.encode_items().numpy()

    # Offline retrieval quality vs the popularity baseline.
    index = AnnIndex(item_vectors)
    user_vecs = model.encode_users(
        torch.as_tensor(val_df["user_idx"].to_numpy(), dtype=torch.long)
    ).numpy()
    _, retrieved = index.search(user_vecs, args.eval_k)
    model_recall = recall_at_k(retrieved, val_df["item_idx"].to_numpy())
    pop_top = popularity["item_idx"].to_numpy()[: args.eval_k]
    pop_recall = recall_at_k(
        np.broadcast_to(pop_top, (len(val_df), len(pop_top))), val_df["item_idx"].to_numpy()
    )
    metrics = {
        "recall_at_k": model_recall,
        "popularity_recall_at_k": pop_recall,
        "k": args.eval_k,
        "num_val_users": len(val_df),
        "epochs": cfg.epochs,
    }
    print(f"recall@{args.eval_k}: model={model_recall:.4f} popularity={pop_recall:.4f}")

    save_artifacts(args.out, model, item_vectors, user_map, item_map, popularity, metrics)
    print(f"artifacts -> {args.out}")


if __name__ == "__main__":
    main()
