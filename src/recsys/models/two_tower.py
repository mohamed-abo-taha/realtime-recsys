"""Two-tower retrieval model.

Separate user and item encoders map ids into a shared embedding space where
dot product ~ affinity. Item vectors are precomputed offline into an ANN
index; only the user tower runs at request time.

Training uses in-batch sampled softmax: every other item in the batch acts as
a negative. Popular items appear more often as in-batch negatives, which
biases against them; the standard logQ sampling correction subtracts each
item's log sampling probability from its logit (Yi et al., 2019).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Tower(nn.Module):
    """id -> L2-normalized embedding."""

    def __init__(self, num_ids: int, embed_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_ids, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.embedding.weight, std=0.05)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(self.embedding(ids)), dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 64,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.user_tower = Tower(num_users, embed_dim, hidden_dim, out_dim)
        self.item_tower = Tower(num_items, embed_dim, hidden_dim, out_dim)
        self.temperature = temperature
        self.hparams = {
            "num_users": num_users,
            "num_items": num_items,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
            "temperature": temperature,
        }

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_tower(user_ids), self.item_tower(item_ids)

    def in_batch_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_log_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """item_log_q: per-batch-item log sampling probability for the logQ correction."""
        u, v = self(user_ids, item_ids)
        logits = (u @ v.T) / self.temperature
        if item_log_q is not None:
            logits = logits - item_log_q[None, :]
        labels = torch.arange(len(u), device=u.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def encode_items(self, device: str = "cpu", batch_size: int = 8192) -> torch.Tensor:
        """Precompute all item embeddings (offline path)."""
        self.eval()
        num_items = self.hparams["num_items"]
        out = []
        for start in range(0, num_items, batch_size):
            ids = torch.arange(start, min(start + batch_size, num_items), device=device)
            out.append(self.item_tower(ids).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.user_tower(user_ids)
