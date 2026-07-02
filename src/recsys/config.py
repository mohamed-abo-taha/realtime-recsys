"""Central paths and hyperparameters.

Kept as plain dataclasses (no config framework) — override via CLI flags on the
entrypoint scripts or the ARTIFACTS_DIR / DATA_DIR environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))


def artifacts_dir() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))


@dataclass
class PrepareConfig:
    positive_threshold: float = 4.0  # rating >= threshold counts as an implicit positive
    min_user_interactions: int = 5
    min_item_interactions: int = 5


@dataclass
class TrainConfig:
    embed_dim: int = 64
    hidden_dim: int = 128
    out_dim: int = 64
    temperature: float = 0.05
    batch_size: int = 4096
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-6
    seed: int = 42
    device: str = "auto"  # auto -> cuda if available


@dataclass
class ServingConfig:
    default_k: int = 10  # final top-K returned to the caller
    max_k: int = 500
    retrieve_k: int = 200  # candidates pulled from the ANN index before ranking
    index_backend: str = "auto"  # auto -> faiss if importable, else exact numpy
    enable_cache: bool = os.environ.get("CACHE_USERS", "1") != "0"
    redis_url: str | None = os.environ.get("REDIS_URL") or None


@dataclass
class Config:
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
