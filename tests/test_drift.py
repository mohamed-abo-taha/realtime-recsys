from recsys.monitor.drift import item_embedding_drift, topk_retrieval_drift


def test_identical_artifacts_show_no_drift(artifacts_path):
    report = item_embedding_drift(artifacts_path, artifacts_path)
    assert report["matched_items"] == 25
    assert report["item_cosine_mean"] == 1.0
    report = topk_retrieval_drift(artifacts_path, artifacts_path, k=10, sample=20, seed=0)
    assert report["topk_jaccard_mean"] == 1.0


def test_different_models_show_drift(artifacts_path, retrieval_only_artifacts_path):
    report = item_embedding_drift(artifacts_path, retrieval_only_artifacts_path)
    assert report["matched_items"] == 25
    assert report["item_cosine_mean"] < 1.0
    report = topk_retrieval_drift(
        artifacts_path, retrieval_only_artifacts_path, k=10, sample=20, seed=0
    )
    assert 0.0 <= report["topk_jaccard_mean"] < 1.0
