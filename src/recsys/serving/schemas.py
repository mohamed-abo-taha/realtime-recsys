"""Pydantic request/response contracts for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ItemOut(BaseModel):
    item_id: int
    score: float
    title: str | None = None


class RecommendResponse(BaseModel):
    user_id: int
    cold_start: bool = Field(description="unknown user, served by the popularity fallback")
    k: int
    items: list[ItemOut]
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    num_items: int
    index_backend: str
    mode: str  # "retrieve+rank" or "retrieval-only"
    version: str
