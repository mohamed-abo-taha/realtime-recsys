# Real-Time Recommendation Serving System

A two-stage retrieval-and-ranking recommender served behind a REST API under a **sub-100ms p99 latency budget** — measured **p99 = 10.1ms at 8 concurrent users (~1,500 req/s)** on MovieLens 25M, with training and serving reading the same feature definitions.

```
OFFLINE — training & indexing (batch)
  interactions ──> train two-tower ──> precompute item embeddings ──> build ANN index (FAISS)
                                  │
                          FEATURE STORE
        offline (Parquet, training) ──materialize──> online (Redis, serving)
                one feature definition · no train/serve skew
                                  │
ONLINE — request-time serving (measured p99 = 10.1ms)
  request(user_id) ─> fetch features ─> encode user ─> ANN retrieve 200 ─> rank ─> top-K
                          0.004ms          1.0ms           0.12ms          1.0ms
```

## The numbers (MovieLens 25M: 160,770 users · 19,935 items · 12.1M interactions)

**System** (single uvicorn worker, RTX-4070-trained / CPU-served, 2,000 reqs/level):

| concurrency | req/s | p50 | p95 | p99 |
|---|---|---|---|---|
| 1 | 488 | 1.9ms | 2.9ms | 3.3ms |
| 8 (no cache) | 1,151 | 6.7ms | 8.7ms | 10.8ms |
| **8 (hot-user cache)** | **1,506** | **5.0ms** | **7.0ms** | **10.1ms** |
| 32 | 584 | 33ms | 171ms | 279ms — single-worker ceiling; scale = more workers |
| 8 (Docker + Redis store) | 366 | 21.8ms | 29.2ms | 34.4ms |

Caching user vectors+features buys **+31% throughput** on repeat-user traffic. Per-stage breakdown (in every response): feature fetch 0.004ms in-process / **0.2ms from Redis** · user encode 1.0ms · ANN 0.12ms · LightGBM rank 1.0ms. The Docker row's client-side overhead is Docker Desktop's Windows VM networking — in-container stage totals stay ~1.5ms/request.

**Quality** (leave-last-out, 30k val users, each stage vs baselines):

| metric | two-stage | retrieval order | popularity |
|---|---|---|---|
| NDCG@10 | **0.028** | 0.011 | 0.017 |
| recall@10 | **0.060** | 0.028 | 0.035 |

Retrieval recall@100 = **0.345 vs 0.19 popularity** (+81%; the logQ sampled-softmax correction alone took it from 0.21). Candidate ceiling: recall@200 = 0.48.

**Honesty note:** offline metrics are a proxy. True recommender quality is only measurable with an online A/B test on live traffic, which a portfolio project cannot run. And at this catalogue size the latency concerns don't truly bite — the architecture is what scales, not this box.

## Quickstart

```bash
export PYTHONPATH=src            # Windows: $env:PYTHONPATH = "src"
python -m recsys.data.download small   # or: 25m (~250MB) / recsys.data.synthetic (offline)
python -m recsys.data.prepare --raw data/ml-small/raw/ratings.parquet \
    --items data/ml-small/raw/items.parquet --out data/ml-small/processed
python -m recsys.train.train_retrieval --data data/ml-small/processed --epochs 40 --batch-size 512
python -m recsys.train.train_ranker --data data/ml-small/processed
uvicorn recsys.serving.app:app --port 8000
curl "http://localhost:8000/recommend?user_id=100&k=10"

pytest                                  # 30 tests, no data/downloads needed
docker compose up --build               # api on :8080 + redis; --profile monitoring adds Prometheus/Grafana
python -m recsys.features.materialize   # push user features into the Redis online store
python benchmarks/benchmark.py          # reproduce the latency table
python -m recsys.monitor.drift --old artifacts_v1 --new artifacts_v2   # embedding drift gate
```

## Design decisions & tradeoffs

- **Two stages, not one model** — the latency math forbids scoring the whole catalogue per request; cheap ANN retrieval narrows ~20k (or millions) to 200, LightGBM LambdaRank refines 200 to top-K.
- **Only the user tower runs at request time** — item vectors are precomputed offline into FAISS (exact-numpy fallback when FAISS is absent); the online path is one CPU forward pass + one ANN lookup + one small ranker call.
- **Feature store** — features defined once ([features/store.py](src/recsys/features/store.py)), built offline to Parquet, materialized to Redis for serving (in-process fallback for dev). Same rows both sides = no train/serve skew. Feast would replace this module wholesale; hand-rolled here to keep the dependency surface auditable.
- **logQ correction** — in-batch softmax over-penalizes popular items as negatives; subtracting each item's log sampling probability nearly doubled retrieval recall.
- **Three-way temporal split** — train (towers) / tune (ranker labels) / val (final eval), so the ranker never trains on interactions the towers saw.
- **Cold-start** — unknown users get the popularity fallback, never a 500; flagged `cold_start` in the response and counted in Prometheus.
- **Observability** — `/metrics` exposes latency histograms, per-stage summaries, cold-start counts; Grafana/Prometheus ship in the compose monitoring profile.
- **Drift** — [monitor/drift.py](src/recsys/monitor/drift.py) compares item-embedding cosine and top-K Jaccard overlap between offline runs, with a `--fail-under` promotion gate.

**How I'd scale this:** more uvicorn workers behind a load balancer (the c=32 row is a one-process ceiling, not an architecture ceiling), Triton for the towers, Kubernetes for orchestration, streaming feature updates into Redis, and a real A/B framework — deliberately out of scope here.

## Build phases

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Retrieval slice: two-tower + FAISS, served + containerized | ✅ |
| 2 | Ranking stage: LightGBM LambdaRank, staged metrics vs baselines | ✅ |
| 3 | Feature store: Parquet offline / Redis online, materialize step | ✅ |
| 4 | Load test, hot-user cache (+31% rps), **p99 = 10.1ms @ c=8** | ✅ |
| 5 | Cold-start fallback, drift monitoring, Prometheus, CI | ✅ |
