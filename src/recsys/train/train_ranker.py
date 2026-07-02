"""Train the ranking stage: a LightGBM LambdaRank model over ANN candidates.

For each user in the `tune` split (temporally after the tower-training data,
before the val item), the retrieval stage proposes N candidates; the ranker
learns to push the user's actual next interaction to the top using feature-
store features + the retrieval score. Users whose tune item the retrieval
stage missed carry no ordering signal and are dropped from ranker training.

Evaluates on `val` and reports the two stages separately:
  retrieval  recall@N (candidate ceiling)
  ranking    NDCG@K / recall@K vs (a) retrieval-score order, (b) popularity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from recsys.config import artifacts_dir, data_dir
from recsys.eval.ranking import ndcg_at_k
from recsys.eval.recall import recall_at_k
from recsys.features.store import ALL_FEATURES, LocalOnlineStore, build_features
from recsys.index.ann import AnnIndex
from recsys.models.two_tower import TwoTower
from recsys.serving.artifacts import load_model


def encode_users_batched(model: TwoTower, user_idx: np.ndarray, batch: int = 32768) -> np.ndarray:
    out = []
    for start in range(0, len(user_idx), batch):
        ids = torch.as_tensor(user_idx[start : start + batch], dtype=torch.long)
        out.append(model.encode_users(ids).numpy())
    return np.concatenate(out)


def retrieve_candidates(
    model: TwoTower, index: AnnIndex, user_idx: np.ndarray, n: int, batch: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scores, candidate_idx), each (num_users, n)."""
    scores, cands = [], []
    for start in range(0, len(user_idx), batch):
        vecs = encode_users_batched(model, user_idx[start : start + batch])
        s, c = index.search(vecs, n)
        scores.append(s)
        cands.append(c)
    return np.concatenate(scores), np.concatenate(cands)


def assemble_features(
    store: LocalOnlineStore, user_idx: np.ndarray, cands: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    """(num_users, n) candidates -> (num_users * n, num_features), ALL_FEATURES order."""
    num_users, n = cands.shape
    user_feats = store.user_matrix[user_idx]  # (num_users, U)
    user_block = np.repeat(user_feats, n, axis=0)
    item_block = store.item_matrix[cands.ravel()]
    pair_block = scores.reshape(-1, 1).astype(np.float32)
    return np.hstack([user_block, item_block, pair_block])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=data_dir() / "processed")
    parser.add_argument("--artifacts", type=Path, default=artifacts_dir())
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--max-train-users", type=int, default=40_000)
    parser.add_argument("--max-eval-users", type=int, default=30_000)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = load_model(args.artifacts)
    index = AnnIndex(np.load(args.artifacts / "item_vectors.npy"))
    build_features(args.data, args.artifacts)
    store = LocalOnlineStore(args.artifacts)

    # ---- Ranker training set from the tune split -------------------------
    tune = pd.read_parquet(args.data / "tune.parquet")
    if len(tune) > args.max_train_users:
        tune = tune.sample(args.max_train_users, random_state=args.seed)
    user_idx = tune["user_idx"].to_numpy()
    target = tune["item_idx"].to_numpy()

    scores, cands = retrieve_candidates(model, index, user_idx, args.candidates)
    hit_mask = (cands == target[:, None]).any(axis=1)
    print(
        f"tune users: {len(user_idx)}, retrieval hit@{args.candidates}: {hit_mask.mean():.4f} "
        f"({hit_mask.sum()} usable for ranker training)"
    )
    u, c, s, t = user_idx[hit_mask], cands[hit_mask], scores[hit_mask], target[hit_mask]
    X = assemble_features(store, u, c, s)
    y = (c == t[:, None]).ravel().astype(np.int8)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=args.estimators,
        learning_rate=0.05,
        num_leaves=63,
        random_state=args.seed,
        verbosity=-1,
    )
    ranker.fit(X, y, group=np.full(len(u), args.candidates), feature_name=ALL_FEATURES)
    ranker.booster_.save_model(str(args.artifacts / "ranker.txt"))

    # ---- Evaluation on the val split, stages reported separately ---------
    val = pd.read_parquet(args.data / "val.parquet")
    if len(val) > args.max_eval_users:
        val = val.sample(args.max_eval_users, random_state=args.seed)
    v_users = val["user_idx"].to_numpy()
    v_target = val["item_idx"].to_numpy()

    v_scores, v_cands = retrieve_candidates(model, index, v_users, args.candidates)
    Xv = assemble_features(store, v_users, v_cands, v_scores)
    pred = ranker.predict(Xv).reshape(len(v_users), args.candidates)

    order_ranked = np.take_along_axis(v_cands, np.argsort(-pred, axis=1), axis=1)
    order_retrieval = v_cands  # ANN output is already score-ordered
    popularity = pd.read_parquet(args.data / "popularity.parquet")["item_idx"].to_numpy()
    order_pop = np.broadcast_to(popularity[: args.candidates], v_cands.shape)

    k = args.eval_k
    metrics = {
        "retrieval_recall_at_candidates": recall_at_k(v_cands, v_target),
        "candidates": args.candidates,
        f"ranked_ndcg_at_{k}": ndcg_at_k(order_ranked, v_target, k),
        f"ranked_recall_at_{k}": recall_at_k(order_ranked[:, :k], v_target),
        f"retrieval_ndcg_at_{k}": ndcg_at_k(order_retrieval, v_target, k),
        f"retrieval_recall_at_{k}": recall_at_k(order_retrieval[:, :k], v_target),
        f"popularity_ndcg_at_{k}": ndcg_at_k(order_pop, v_target, k),
        f"popularity_recall_at_{k}": recall_at_k(order_pop[:, :k], v_target),
        "eval_users": len(v_users),
    }
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")

    metrics_path = args.artifacts / "metrics.json"
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    existing["ranking"] = metrics
    metrics_path.write_text(json.dumps(existing, indent=2))
    print(f"ranker + features -> {args.artifacts}")


if __name__ == "__main__":
    main()
