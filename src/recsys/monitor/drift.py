"""Embedding / retrieval drift between two offline runs.

As the catalogue and behaviour shift, a retrained model's embedding space
moves. Two legible signals, compared between an old and a new artifacts dir:

    item_cosine_mean   mean cosine similarity of matched item vectors
                       (same item_id in both catalogues) — embedding drift
    topk_jaccard_mean  mean Jaccard overlap of the top-K retrieved sets for a
                       shared user sample — what drift means for serving

Low cosine with high Jaccard = space rotated but recommendations held; both
low = the model genuinely changed what it serves — investigate before
promoting. Thresholds are a deployment policy, so this reports numbers and
takes a --fail-under gate rather than hardcoding one.

    python -m recsys.monitor.drift --old artifacts_v1 --new artifacts_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.index.ann import AnnIndex
from recsys.serving.artifacts import load_model


def _load(artifacts: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    vectors = np.load(artifacts / "item_vectors.npy")
    item_map = pd.read_parquet(artifacts / "item_map.parquet")
    user_map = pd.read_parquet(artifacts / "user_map.parquet")
    return vectors, item_map, user_map


def item_embedding_drift(old_dir: Path, new_dir: Path) -> dict:
    old_vec, old_items, _ = _load(old_dir)
    new_vec, new_items, _ = _load(new_dir)
    merged = old_items.merge(new_items, on="item_id", suffixes=("_old", "_new"))
    a = old_vec[merged["item_idx_old"].to_numpy()]
    b = new_vec[merged["item_idx_new"].to_numpy()]
    cos = (a * b).sum(axis=1)  # vectors are L2-normalized
    return {
        "matched_items": len(merged),
        "item_cosine_mean": round(float(cos.mean()), 4),
        "item_cosine_p05": round(float(np.percentile(cos, 5)), 4),
    }


def topk_retrieval_drift(old_dir: Path, new_dir: Path, k: int, sample: int, seed: int) -> dict:
    old_vec, _, old_users = _load(old_dir)
    new_vec, _, new_users = _load(new_dir)
    shared = old_users.merge(new_users, on="user_id", suffixes=("_old", "_new"))
    if len(shared) > sample:
        shared = shared.sample(sample, random_state=seed)

    overlaps = []
    for suffix, vectors, artifacts in (("_old", old_vec, old_dir), ("_new", new_vec, new_dir)):
        model = load_model(artifacts)
        idx = torch.as_tensor(shared[f"user_idx{suffix}"].to_numpy(), dtype=torch.long)
        vecs = model.encode_users(idx).numpy()
        _, top = AnnIndex(vectors).search(vecs, k)
        overlaps.append(top)
    old_top, new_top = overlaps

    # Jaccard needs raw item ids (indices differ between runs).
    def raw_ids(artifacts: Path) -> np.ndarray:
        item_map = pd.read_parquet(artifacts / "item_map.parquet")
        return item_map.sort_values("item_idx")["item_id"].to_numpy()

    old_ids, new_ids = raw_ids(old_dir), raw_ids(new_dir)
    jaccards = []
    for row_old, row_new in zip(old_ids[old_top], new_ids[new_top], strict=True):
        a, b = set(row_old.tolist()), set(row_new.tolist())
        jaccards.append(len(a & b) / len(a | b))
    return {
        "shared_users_sampled": len(shared),
        "k": k,
        "topk_jaccard_mean": round(float(np.mean(jaccards)), 4),
        "topk_jaccard_p05": round(float(np.percentile(jaccards, 5)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-under", type=float, help="exit 1 if topk_jaccard_mean falls below")
    args = parser.parse_args()

    report = {
        **item_embedding_drift(args.old, args.new),
        **topk_retrieval_drift(args.old, args.new, args.k, args.sample, args.seed),
    }
    print(json.dumps(report, indent=2))
    if args.fail_under is not None and report["topk_jaccard_mean"] < args.fail_under:
        print(f"DRIFT: topk_jaccard_mean < {args.fail_under}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
