---
title: Real-Time Recommender
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Two-stage retrieval+ranking recsys with per-request stage timings
---

# Real-Time Recommendation Serving System — live demo

Two-stage recommender (two-tower retrieval → FAISS ANN → LightGBM ranking) trained on
MovieLens 25M, served under a sub-100ms latency budget. Every response includes the
per-stage latency breakdown (feature fetch / user encode / ANN / rank).

- **Demo UI:** the Space root
- **API docs:** `/docs` · **Prometheus metrics:** `/metrics`
- **Source, benchmarks & design decisions:** https://github.com/mohamed-abo-taha/realtime-recsys

Headline numbers (local benchmark, single worker): p99 = 10.1ms @ 8 concurrent
(~1,500 req/s); retrieval recall@100 = 0.345 vs 0.19 popularity baseline; ranked
NDCG@10 = 0.028 vs 0.011 retrieval-order. This free-tier Space is slower than the
benchmark box — the architecture, not this hardware, is the point.
