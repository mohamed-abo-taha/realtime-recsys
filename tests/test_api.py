import pytest
from fastapi.testclient import TestClient

from recsys.serving import app as app_module


@pytest.fixture()
def client(artifacts_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_path))
    with TestClient(app_module.app) as client:
        yield client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["num_items"] == 25
    assert body["mode"] == "retrieve+rank"
    assert body["index_backend"] in {"faiss", "numpy"}


def test_recommend_known_user(client):
    resp = client.get("/recommend", params={"user_id": 1, "k": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cold_start"] is False
    assert len(body["items"]) == 10
    assert body["items"][0]["title"] is not None
    scores = [i["score"] for i in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert set(body["timings_ms"]) == {"feature_fetch", "encode", "ann", "rank", "total"}
    assert "X-Response-Time-Ms" in resp.headers


def test_recommend_unknown_user_falls_back_to_popularity(client):
    resp = client.get("/recommend", params={"user_id": 999999, "k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cold_start"] is True
    assert len(body["items"]) == 5
    # Popularity fixture is ordered by item_idx, so fallback returns items 100..104.
    assert [i["item_id"] for i in body["items"]] == [100, 101, 102, 103, 104]


def test_recommend_validates_input(client):
    assert client.get("/recommend", params={"user_id": -1}).status_code == 422
    assert client.get("/recommend", params={"user_id": 1, "k": 0}).status_code == 422
    assert client.get("/recommend", params={"user_id": 1, "k": 10_000}).status_code == 422


def test_metrics_endpoint(client):
    client.get("/recommend", params={"user_id": 1, "k": 5})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "recsys_request_latency_seconds" in resp.text
    assert "recsys_stage_latency_ms" in resp.text
