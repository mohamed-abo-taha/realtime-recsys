"""Ranking quality under leave-last-out: NDCG@K with one relevant item.

With a single held-out item per user, NDCG@K reduces to 1/log2(rank+1) when
the item appears at `rank` (1-based) in the top K, else 0 — averaged over
users. Reported next to recall@K so retrieval coverage and ranking order can
be judged separately.
"""

from __future__ import annotations

import numpy as np


def ndcg_at_k(ranked: np.ndarray, target_items: np.ndarray, k: int) -> float:
    """ranked: (num_users, >=k) item indices in ranked order; target: (num_users,)."""
    if len(ranked) != len(target_items):
        raise ValueError("ranked and target_items must have the same length")
    top = ranked[:, :k]
    hits = top == np.asarray(target_items)[:, None]
    ranks = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0)
    gains = np.zeros(len(ranks))
    hit_mask = ranks > 0
    gains[hit_mask] = 1.0 / np.log2(ranks[hit_mask] + 1)
    return float(gains.mean())
