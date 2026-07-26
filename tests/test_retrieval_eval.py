from evaluation.retrieval_eval import _breakdown_by, evidence_subset_match, recall_at_k


def test_recall_at_k_partial_credit():
    assert recall_at_k(["a", "b", "x"], {"a", "b", "c"}, 20) == 2 / 3


def test_recall_at_k_no_expected_evidence_is_trivially_perfect():
    assert recall_at_k(["a"], set(), 20) == 1.0


def test_evidence_subset_match_true_when_expected_is_subset_of_top_k():
    # retrieval returning more than the expected 2-4 titles is normal at k=20;
    # a strict set-equality check would read ~0% almost always.
    assert evidence_subset_match(["a", "b", "c", "d"], {"a", "b"}, 20) is True


def test_evidence_subset_match_false_when_expected_title_missing():
    assert evidence_subset_match(["a", "c"], {"a", "b"}, 20) is False


def test_evidence_subset_match_only_considers_top_k():
    assert evidence_subset_match(["c", "d", "a", "b"], {"a", "b"}, 2) is False


def test_breakdown_by_computes_per_bucket_rates():
    results = [
        {"question_type": "inference_query", "recall_at_20": 1.0, "evidence_subset_match": True},
        {"question_type": "inference_query", "recall_at_20": 0.5, "evidence_subset_match": False},
        {"question_type": "temporal_query", "recall_at_20": 1.0, "evidence_subset_match": True},
    ]

    breakdown = _breakdown_by(results, "question_type")

    assert breakdown["inference_query"]["count"] == 2
    assert breakdown["inference_query"]["recall_at_20"] == 0.75
    assert breakdown["inference_query"]["evidence_subset_match_rate"] == 0.5
    assert breakdown["temporal_query"]["count"] == 1
