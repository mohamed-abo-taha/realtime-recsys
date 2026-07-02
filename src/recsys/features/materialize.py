"""Materialize offline user features into the Redis online store.

    python -m recsys.features.materialize --redis-url redis://localhost:6379/0

Run after every offline build; serving then reads the same rows from Redis
that training read from Parquet.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from recsys.config import artifacts_dir
from recsys.features.store import RedisOnlineStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=artifacts_dir())
    parser.add_argument(
        "--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    )
    args = parser.parse_args()

    store = RedisOnlineStore(args.artifacts, args.redis_url)
    count = store.materialize()
    print(f"materialized {count} users -> {args.redis_url}")


if __name__ == "__main__":
    main()
