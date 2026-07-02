from recsys.serving.engine import RetrievalEngine


def test_two_stage_mode_when_ranker_present(artifacts_path):
    engine = RetrievalEngine(artifacts_path, retrieve_k=15)
    assert engine.two_stage
    result = engine.recommend(user_id=1, k=5)
    assert not result.cold_start
    assert len(result.items) == 5
    scores = [i.score for i in result.items]
    assert scores == sorted(scores, reverse=True)
    assert set(result.timings_ms) == {"feature_fetch", "encode", "ann", "rank", "total"}


def test_retrieval_only_fallback(retrieval_only_artifacts_path):
    engine = RetrievalEngine(retrieval_only_artifacts_path)
    assert not engine.two_stage
    result = engine.recommend(user_id=1, k=5)
    assert len(result.items) == 5
    assert result.timings_ms["rank"] == 0.0


def test_cache_skips_fetch_and_encode(artifacts_path):
    engine = RetrievalEngine(artifacts_path, enable_cache=True)
    first = engine.recommend(user_id=2, k=3)
    second = engine.recommend(user_id=2, k=3)
    assert first.items[0].item_id == second.items[0].item_id  # deterministic
    assert second.timings_ms["feature_fetch"] == 0.0
    assert second.timings_ms["encode"] == 0.0


def test_cache_disabled_always_encodes(artifacts_path):
    engine = RetrievalEngine(artifacts_path, enable_cache=False)
    engine.recommend(user_id=2, k=3)
    second = engine.recommend(user_id=2, k=3)
    assert second.timings_ms["encode"] > 0.0


def test_unknown_user_cold_start(artifacts_path):
    engine = RetrievalEngine(artifacts_path)
    result = engine.recommend(user_id=99999, k=4)
    assert result.cold_start
    assert len(result.items) == 4
