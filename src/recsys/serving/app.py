"""FastAPI online path: GET /recommend returns the ranked top-K.

Run locally:
    uvicorn recsys.serving.app:app --port 8000

Environment:
    ARTIFACTS_DIR   training output (default ./artifacts)
    REDIS_URL       online feature store (in-process fallback when unset)
    CACHE_USERS=0   disable the hot-user cache (for before/after load tests)

Prometheus metrics at /metrics: request latency histogram, per-stage latency
summaries, cold-start counter.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Summary, generate_latest

from recsys import __version__
from recsys.config import ServingConfig, artifacts_dir
from recsys.serving.engine import STAGES, RetrievalEngine
from recsys.serving.schemas import HealthResponse, ItemOut, RecommendResponse

logger = logging.getLogger("recsys.serving")

REQUEST_LATENCY = Histogram(
    "recsys_request_latency_seconds",
    "End-to-end /recommend latency",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
STAGE_LATENCY = Summary(
    "recsys_stage_latency_ms", "Per-stage latency in ms", labelnames=("stage",)
)
COLD_STARTS = Counter("recsys_cold_start_total", "Requests served by the popularity fallback")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = ServingConfig()
    path = artifacts_dir()
    app.state.engine = RetrievalEngine(
        path,
        index_backend=cfg.index_backend,
        redis_url=cfg.redis_url,
        retrieve_k=cfg.retrieve_k,
        enable_cache=cfg.enable_cache,
    )
    app.state.cfg = cfg
    logger.info(
        "engine loaded: %d items, backend=%s, mode=%s, cache=%s, artifacts=%s",
        app.state.engine.num_items,
        app.state.engine.index.backend,
        "retrieve+rank" if app.state.engine.two_stage else "retrieval-only",
        cfg.enable_cache,
        path,
    )
    yield


app = FastAPI(title="Real-Time Recommender", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    if request.url.path == "/recommend":
        REQUEST_LATENCY.observe(elapsed)
    response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.2f}"
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    engine: RetrievalEngine = app.state.engine
    return HealthResponse(
        status="ok",
        num_items=engine.num_items,
        index_backend=engine.index.backend,
        mode="retrieve+rank" if engine.two_stage else "retrieval-only",
        version=__version__,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: int = Query(ge=0),
    k: int = Query(default=ServingConfig.default_k, ge=1, le=ServingConfig.max_k),
) -> RecommendResponse:
    engine: RetrievalEngine = app.state.engine
    try:
        result = engine.recommend(user_id, k)
    except Exception:  # pragma: no cover - surfaced as a clean 500 instead of a stack trace
        logger.exception("recommend failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="recommendation failed") from None
    if result.cold_start:
        COLD_STARTS.inc()
    for stage in STAGES:
        STAGE_LATENCY.labels(stage=stage).observe(result.timings_ms[stage])
    return RecommendResponse(
        user_id=result.user_id,
        cold_start=result.cold_start,
        k=len(result.items),
        items=[ItemOut(item_id=i.item_id, score=i.score, title=i.title) for i in result.items],
        timings_ms=result.timings_ms,
    )
