"""Save/load the bundle of artifacts the online path needs.

One directory holds everything serving requires: the user tower weights,
precomputed item vectors, id maps, and the popularity fallback. Training
writes it; the API loads it. Keeping this a single explicit contract is what
makes offline and online agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.models.two_tower import TwoTower


def save_artifacts(
    out_dir: Path,
    model: TwoTower,
    item_vectors: np.ndarray,
    user_map: pd.DataFrame,
    item_map: pd.DataFrame,
    popularity: pd.DataFrame,
    metrics: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "two_tower.pt")
    (out_dir / "model_meta.json").write_text(json.dumps(model.hparams, indent=2))
    np.save(out_dir / "item_vectors.npy", np.ascontiguousarray(item_vectors, dtype=np.float32))
    user_map.to_parquet(out_dir / "user_map.parquet", index=False)
    item_map.to_parquet(out_dir / "item_map.parquet", index=False)
    popularity.to_parquet(out_dir / "popularity.parquet", index=False)
    if metrics is not None:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_model(artifacts_dir: Path) -> TwoTower:
    meta = json.loads((artifacts_dir / "model_meta.json").read_text())
    model = TwoTower(**meta)
    model.load_state_dict(torch.load(artifacts_dir / "two_tower.pt", map_location="cpu"))
    model.eval()
    return model
