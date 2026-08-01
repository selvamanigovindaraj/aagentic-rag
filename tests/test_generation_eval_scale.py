import asyncio
import json

import pytest
from app.core.config import Settings
from app.services.rag_pipeline import RagPipeline

from evaluation.generation_eval import _breakdown_by_type, _case_result, _run_all_cases


class _DirectRetriever:
    async def retrieve(self, query, route, auth, limit):
        return []


def _case(name: str, question_type: str = "inference_query") -> dict:
    return {
        "name": name,
        "query": "anything",
        "tenant_id": "eval",
        "groups": ["research"],
        "expect_refusal": True,
        "question_type": question_type,
    }


@pytest.mark.asyncio
async def test_run_all_cases_writes_incrementally_and_resumes(tmp_path):
    pipeline = RagPipeline(_DirectRetriever(), Settings(), None)
    cases = [_case("a"), _case("b"), _case("c")]
    results_path = tmp_path / "results.jsonl"

    first_pass = await _run_all_cases(pipeline, cases, concurrency=2, results_path=results_path)
    assert len(first_pass) == 3
    assert len(results_path.read_text().splitlines()) == 3

    # Simulate a crash after 2 of 3 cases: truncate the persisted file, then resume.
    lines = results_path.read_text().splitlines()
    results_path.write_text("\n".join(lines[:2]) + "\n")

    resumed = await _run_all_cases(pipeline, cases, concurrency=2, results_path=results_path)

    assert len(resumed) == 3
    assert len(results_path.read_text().splitlines()) == 3  # no duplicate appends
    assert {json.loads(line)["name"] for line in results_path.read_text().splitlines()} == {
        "a", "b", "c",
    }


@pytest.mark.asyncio
async def test_run_all_cases_bounds_concurrency(tmp_path):
    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    class SlowPipeline:
        async def run(self, query, auth):
            nonlocal concurrent, max_concurrent
            async with lock:
                concurrent += 1
                max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                concurrent -= 1
            return {"answer": ""}

    cases = [_case(str(i)) for i in range(6)]

    await _run_all_cases(SlowPipeline(), cases, concurrency=2, results_path=None)

    assert max_concurrent <= 2


def test_breakdown_by_type_excludes_none_and_handles_empty_bucket():
    results = [
        {"question_type": "inference_query", "refusal_correct": True, "grounded": True,
         "citations_valid": None, "answer_correct": None},
        {"question_type": "inference_query", "refusal_correct": False, "grounded": False,
         "citations_valid": None, "answer_correct": None},
        {"question_type": "null_query", "refusal_correct": True, "grounded": None,
         "citations_valid": None, "answer_correct": None},
    ]

    breakdown = _breakdown_by_type(results)

    assert breakdown["inference_query"]["count"] == 2
    assert breakdown["inference_query"]["refusal_accuracy"] == 0.5
    assert breakdown["inference_query"]["grounded_claim_rate"] == 0.5
    assert breakdown["inference_query"]["citation_validity"] is None
    assert breakdown["null_query"]["grounded_claim_rate"] is None


def test_case_result_does_not_read_a_numbered_fallback_dump_as_grounded():
    """_numbered_fallback sets grounded_claims == generated_claims and
    valid_citation_references == citation_references by construction (it never
    checks the evidence is actually relevant), so the plain equality check
    used to read a raw-evidence-dump answer as tautologically 100% grounded
    and 100% cited. answer_source now carries the real signal."""
    case = {"name": "n", "expect_refusal": True, "question_type": "null_query"}
    result = {
        "answer": "some unrelated evidence text [1]",
        "generated_claims": 1,
        "grounded_claims": 1,
        "citation_references": 1,
        "valid_citation_references": 1,
        "answer_source": "fallback",
    }

    case_result = _case_result(case, result)

    assert case_result["grounded"] is False
    assert case_result["citations_valid"] is False
    assert case_result["answer_source"] == "fallback"


def test_case_result_reports_grounded_answer_normally():
    case = {"name": "n", "expect_refusal": False, "question_type": "inference_query"}
    result = {
        "answer": "A real answer. [1]",
        "generated_claims": 1,
        "grounded_claims": 1,
        "citation_references": 1,
        "valid_citation_references": 1,
        "answer_source": "grounded",
    }

    case_result = _case_result(case, result)

    assert case_result["grounded"] is True
    assert case_result["citations_valid"] is True
