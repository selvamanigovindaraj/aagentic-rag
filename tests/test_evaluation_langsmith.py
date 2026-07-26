from types import SimpleNamespace

import pytest
from app.core.config import Settings
from app.services.rag_pipeline import RagPipeline

from evaluation.generation_eval import (
    _citations_evaluator,
    _grounded_evaluator,
    _refusal_evaluator,
    _run_case,
    _summarize_langsmith_results,
)
from evaluation.langsmith_support import ensure_dataset


class FakeClient:
    def __init__(self, *, existing_dataset: bool, existing_names: set[str]):
        self._existing_dataset = existing_dataset
        self._existing_names = existing_names
        self.created_dataset = False
        self.created_examples: list[dict] | None = None

    def has_dataset(self, *, dataset_name: str) -> bool:
        return self._existing_dataset

    def create_dataset(self, dataset_name: str, *, description: str = "") -> None:
        self.created_dataset = True

    def read_dataset(self, *, dataset_name: str):
        return SimpleNamespace(id="dataset-1")

    def list_examples(self, *, dataset_id: str):
        return [SimpleNamespace(inputs={"name": name}) for name in self._existing_names]

    def create_examples(self, *, dataset_id: str, examples: list[dict]) -> None:
        self.created_examples = examples


def test_ensure_dataset_creates_when_missing_and_upserts_only_new_examples():
    client = FakeClient(existing_dataset=False, existing_names={"case-a"})
    examples = [{"name": "case-a", "query": "old"}, {"name": "case-b", "query": "new"}]

    dataset_id = ensure_dataset(client, "generation-eval", examples)

    assert dataset_id == "dataset-1"
    assert client.created_dataset is True
    assert client.created_examples == [
        {"inputs": {"name": "case-b", "query": "new"}, "outputs": {}}
    ]


def test_ensure_dataset_does_not_recreate_an_existing_dataset():
    client = FakeClient(existing_dataset=True, existing_names={"case-a", "case-b"})
    examples = [{"name": "case-a", "query": "old"}, {"name": "case-b", "query": "new"}]

    ensure_dataset(client, "generation-eval", examples)

    assert client.created_dataset is False
    assert client.created_examples is None


class _DirectRetriever:
    async def retrieve(self, query, route, auth, limit):
        return []


@pytest.mark.asyncio
async def test_run_case_marks_refusal_correct_when_no_evidence_and_refusal_expected():
    pipeline = RagPipeline(_DirectRetriever(), Settings(), None)
    case = {
        "name": "no evidence",
        "query": "anything",
        "tenant_id": "eval",
        "groups": ["research"],
        "expect_refusal": True,
    }

    result = await _run_case(pipeline, case)

    assert result["refused"] is True
    assert result["refusal_correct"] is True
    assert result["grounded"] is None
    assert result["citations_valid"] is None
    assert "error" not in result


@pytest.mark.asyncio
async def test_run_case_reports_error_without_crashing_the_batch():
    class BrokenPipeline:
        async def run(self, query, auth):
            raise RuntimeError("boom")

    case = {
        "name": "broken", "query": "q", "tenant_id": "eval", "groups": [], "expect_refusal": False,
    }

    result = await _run_case(BrokenPipeline(), case)

    assert result["refusal_correct"] is False
    assert result["grounded"] is None
    assert "RuntimeError: boom" in result["error"]


def test_row_evaluators_extract_the_matching_output_field():
    outputs = {"refusal_correct": True, "grounded": None, "citations_valid": False}
    run = SimpleNamespace(outputs=outputs)

    assert _refusal_evaluator(run, None) == {"key": "refusal_correct", "score": True}
    assert _grounded_evaluator(run, None) == {"key": "grounded", "score": None}
    assert _citations_evaluator(run, None) == {"key": "citations_valid", "score": False}


class _FakeExperimentResults:
    def __init__(self, rows: list[dict], *, experiment_name: str, url: str):
        self._rows = rows
        self.experiment_name = experiment_name
        self.url = url

    def __iter__(self):
        return iter(self._rows)


def test_summarize_langsmith_results_averages_scores_and_ignores_none():
    def _row(refusal_correct, grounded, citations_valid):
        results = [
            SimpleNamespace(key="refusal_correct", score=refusal_correct),
            SimpleNamespace(key="grounded", score=grounded),
            SimpleNamespace(key="citations_valid", score=citations_valid),
        ]
        return {
            "run": None,
            "example": None,
            "evaluation_results": SimpleNamespace(results=results),
        }

    rows = [_row(True, True, True), _row(True, None, False), _row(False, False, None)]
    results = _FakeExperimentResults(rows, experiment_name="exp-1", url="https://smith/exp-1")

    summary = _summarize_langsmith_results(results)

    assert summary["refusal_accuracy"] == pytest.approx(2 / 3)
    assert summary["grounded_claim_rate"] == pytest.approx(0.5)
    assert summary["citation_validity"] == pytest.approx(0.5)
    assert summary["experiment_name"] == "exp-1"
    assert summary["experiment_url"] == "https://smith/exp-1"
