import asyncio
from uuid import uuid4

import pytest
from app.schemas.domain import Evidence

from evaluation.retrieval_eval import _evaluate_case, _run_all_cases


def _case(name: str) -> dict:
    return {
        "name": name,
        "query": name,
        "route": "multi_hop",
        "tenant_id": "eval",
        "groups": ["research"],
        "expected_document_titles": ["Doc A"],
    }


@pytest.mark.asyncio
async def test_evaluate_case_computes_recall_and_match():
    class FakeRetriever:
        async def retrieve(self, query, route, auth, limit):
            return [
                Evidence(
                    id=str(uuid4()), document_id=str(uuid4()),
                    document_title="Doc A", text="t", score=1.0,
                )
            ]

    result = await _evaluate_case(FakeRetriever(), _case("case-1"), k=20, fetch_limit=50)

    assert result["recall_at_20"] == 1.0
    assert result["evidence_subset_match"] is True
    assert result["retrieved_document_titles"] == ["Doc A"]


@pytest.mark.asyncio
async def test_run_all_cases_bounds_concurrency():
    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    class SlowRetriever:
        async def retrieve(self, query, route, auth, limit):
            nonlocal concurrent, max_concurrent
            async with lock:
                concurrent += 1
                max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                concurrent -= 1
            return []

    cases = [_case(str(i)) for i in range(6)]

    await _run_all_cases(SlowRetriever(), cases, k=20, fetch_limit=50, concurrency=2)

    assert max_concurrent <= 2
