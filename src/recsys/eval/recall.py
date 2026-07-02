"""Retrieval quality: recall@K under leave-last-out.

For each validation user the model retrieves K candidates; recall@K is the
fraction of users whose held-out item is in that set. Ranking metrics
(NDCG@K) arrive with the ranking stage in Phase 2.
"""

from __future__ import annotations

import numpy as np


def recall_at_k(retrieved: np.ndarray, target_items: np.ndarray) -> float:
    """retrieved: (num_users, K) item indices; target_items: (num_users,)."""
    if len(retrieved) != len(target_items):
        raise ValueError("retrieved and target_items must have the same length")
    hits = (retrieved == np.asarray(target_items)[:, None]).any(axis=1)
    return float(hits.mean())
