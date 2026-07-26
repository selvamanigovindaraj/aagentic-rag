from evaluation.phoenix_experiments import _dataset_examples


def test_dataset_examples_maps_input_and_output_keys():
    cases = [
        {"query": "q1", "retrieved_document_titles": ["a", "b"], "recall_at_20": 0.5},
        {"name": "case-2", "retrieved_document_titles": ["c"], "recall_at_20": 1.0},
    ]

    examples = _dataset_examples(cases, ["query"], ["retrieved_document_titles"])

    assert examples[0]["input"] == {"query": "q1"}
    assert examples[0]["output"] == {"retrieved_document_titles": ["a", "b"]}
    assert examples[0]["metadata"]["name"] == "q1"
    assert examples[1]["metadata"]["name"] == "case-2"
