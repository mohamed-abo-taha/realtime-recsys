"""Locust scenario for interactive load exploration (requires `pip install locust`).

    locust -f benchmarks/locustfile.py --host http://127.0.0.1:8000

The scripted benchmark.py produces the README numbers; this exists for
poking at the throughput ceiling with a live dashboard.
"""

from __future__ import annotations

import random

from locust import HttpUser, task


class RecommendUser(HttpUser):
    @task(50)
    def recommend_known(self) -> None:
        self.client.get(
            "/recommend", params={"user_id": random.randint(1, 162_000), "k": 10}
        )

    @task(1)
    def recommend_cold_start(self) -> None:
        self.client.get(
            "/recommend", params={"user_id": random.randint(10**9, 2 * 10**9), "k": 10}
        )
