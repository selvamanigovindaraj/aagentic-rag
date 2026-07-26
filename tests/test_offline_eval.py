from evaluation.offline_eval import _breakdown_route_accuracy


def test_breakdown_route_accuracy_groups_by_question_type():
    results = [
        {"question_type": "inference_query", "correct": True},
        {"question_type": "inference_query", "correct": False},
        {"question_type": "temporal_query", "correct": True},
    ]

    breakdown = _breakdown_route_accuracy(results)

    assert breakdown["inference_query"] == {"count": 2, "accuracy": 0.5}
    assert breakdown["temporal_query"] == {"count": 1, "accuracy": 1.0}
