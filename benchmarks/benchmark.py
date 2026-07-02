"""Async load generator: p50/p95/p99 latency and throughput vs concurrency.

Self-contained (httpx only) so the headline numbers are reproducible without
extra installs; a Locust scenario for interactive load exploration lives next
to this file. Run against a live server:

    python benchmarks/benchmark.py --artifacts artifacts --requests 2000 \
        --concurrency 1 8 32 [--out benchmarks/results.json]

User ids are sampled (seeded) from the trained user map so requests hit the
real model path, with a slice of unknown users to exercise the cold-start
fallback like real traffic would.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


def percentile(sorted_ms: list[float], p: float) -> float:
    idx = min(int(len(sorted_ms) * p), len(sorted_ms) - 1)
    return sorted_ms[idx]


async def run_level(url: str, user_ids: list[int], k: int, concurrency: int) -> dict:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for uid in user_ids:
        queue.put_nowait(uid)
    latencies: list[float] = []
    errors = 0

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal errors
        while True:
            try:
                uid = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            t0 = time.perf_counter()
            try:
                resp = await client.get(f"{url}/recommend", params={"user_id": uid, "k": k})
                if resp.status_code != 200:
                    errors += 1
            except httpx.HTTPError:
                errors += 1
            latencies.append((time.perf_counter() - t0) * 1000)

    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
        start = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        wall = time.perf_counter() - start

    latencies.sort()
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "errors": errors,
        "rps": round(len(latencies) / wall, 1),
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--unknown-frac", type=float, default=0.02)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    user_map = pd.read_parquet(args.artifacts / "user_map.parquet")
    rng = np.random.default_rng(args.seed)
    known = rng.choice(user_map["user_id"].to_numpy(), size=args.requests, replace=True)
    unknown_mask = rng.random(args.requests) < args.unknown_frac
    ids = np.where(unknown_mask, 10**9 + np.arange(args.requests), known).tolist()

    results = []
    for c in args.concurrency:
        rng.shuffle(ids)
        level = await run_level(args.url, ids, args.k, c)
        results.append(level)
        print(
            f"c={level['concurrency']:>3}  rps={level['rps']:>8}  "
            f"p50={level['p50_ms']:>7}ms  p95={level['p95_ms']:>7}ms  "
            f"p99={level['p99_ms']:>7}ms  errors={level['errors']}"
        )

    if args.out:
        payload = {"label": args.label, "k": args.k, "levels": results}
        existing = json.loads(args.out.read_text()) if args.out.exists() else []
        existing.append(payload)
        args.out.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
