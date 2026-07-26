import json
from types import SimpleNamespace

import pandas as pd

from evaluation.phoenix_evals import _annotations_frame, _load_generate_spans


def test_load_generate_spans_skips_cases_without_evidence_or_answer():
    spans = pd.DataFrame(
        [
            {
                "name": "generate",
                "attributes.input.value": json.dumps(
                    {"query": "q1", "accepted_evidence": [{"text": "evidence"}]}
                ),
                "attributes.output.value": json.dumps({"answer": "a1"}),
            },
            {
                "name": "generate",
                "attributes.input.value": json.dumps({"query": "q2", "accepted_evidence": []}),
                "attributes.output.value": json.dumps({"answer": "could not find evidence"}),
            },
            {
                "name": "classify",
                "attributes.input.value": json.dumps({"query": "q3"}),
                "attributes.output.value": json.dumps({"answer": "irrelevant"}),
            },
        ],
        index=["span-1", "span-2", "span-3"],
    )
    client = SimpleNamespace(spans=SimpleNamespace(get_spans_dataframe=lambda **_: spans))

    result = _load_generate_spans(client, "agentic-rag", 100)

    assert len(result) == 1
    assert result.iloc[0]["query"] == "q1"
    assert result.iloc[0]["reference"] == "evidence"
    assert result.iloc[0]["context.span_id"] == "span-1"


def test_annotations_frame_skips_rows_where_judge_call_failed():
    evaluator = SimpleNamespace(name="hallucination")
    scores = json.dumps([{"name": "hallucination", "label": "factual", "score": 1}])
    results = pd.DataFrame(
        [
            {"context.span_id": "span-1", "hallucination_score": scores},
            {"context.span_id": "span-2", "hallucination_score": None},  # judge call failed
        ]
    )

    annotations = _annotations_frame(results, [evaluator])

    assert list(annotations.index) == ["span-1"]
    assert annotations.iloc[0]["label"] == "factual"


def test_annotations_frame_empty_when_every_row_failed():
    evaluator = SimpleNamespace(name="hallucination")
    results = pd.DataFrame([{"context.span_id": "span-1", "hallucination_score": None}])

    annotations = _annotations_frame(results, [evaluator])

    assert annotations.empty
