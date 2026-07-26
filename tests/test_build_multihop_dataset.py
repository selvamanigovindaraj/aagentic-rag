import json
import random

import httpx
import pytest

from evaluation.build_multihop_dataset import (
    ROUTE_BY_TYPE,
    _build_generation,
    _build_golden,
    _build_retrieval,
    _write_corpus,
    ingest,
)


def _queries():
    return [
        {
            "query": f"inference {i}",
            "question_type": "inference_query",
            "answer": "A",
            "evidence_list": [{"title": "t1"}, {"title": "t2"}],
        }
        for i in range(3)
    ] + [
        {
            "query": f"comparison {i}",
            "question_type": "comparison_query",
            "answer": "B",
            "evidence_list": [
                {"title": "t1"}, {"title": "t2"}, {"title": "t3"}, {"title": "t4"}, {"title": "t5"},
            ],
        }
        for i in range(2)
    ] + [
        {
            "query": f"temporal {i}",
            "question_type": "temporal_query",
            "answer": "C",
            "evidence_list": [{"title": "t1"}, {"title": "t2"}],
        }
        for i in range(2)
    ] + [
        {"query": f"null {i}", "question_type": "null_query", "answer": "", "evidence_list": []}
        for i in range(4)
    ]


def _corpus_titles():
    return {"t1", "t2", "t3", "t4", "t5"}


def test_build_golden_full_pool_returns_every_case_per_type():
    golden = _build_golden(_queries(), random.Random(1), sample_per_type=None)

    assert len(golden) == 3 + 2 + 2  # every inference/comparison/temporal case, no null
    assert all("question_type" in case for case in golden)


def test_build_golden_sampled_respects_cap():
    golden = _build_golden(_queries(), random.Random(1), sample_per_type=1)

    assert len(golden) == len(ROUTE_BY_TYPE)  # one per type


def test_build_retrieval_full_mode_uncaps_evidence_count():
    cases = _build_retrieval(
        _queries(), _corpus_titles(), random.Random(1), sample_per_type=None, max_evidence=None
    )

    comparison_cases = [c for c in cases if c["question_type"] == "comparison_query"]
    assert comparison_cases
    assert comparison_cases[0]["evidence_count"] == 5


def test_build_retrieval_sampled_mode_still_caps_evidence_count():
    cases = _build_retrieval(
        _queries(), _corpus_titles(), random.Random(1), sample_per_type=10, max_evidence=4
    )

    assert not any(c["question_type"] == "comparison_query" for c in cases)


def test_write_corpus_writes_every_requested_title_not_just_referenced(tmp_path, monkeypatch):
    import evaluation.build_multihop_dataset as module

    monkeypatch.setattr(module, "CORPUS_DIR", tmp_path)
    corpus_by_title = {t: {"title": t, "body": "body"} for t in _corpus_titles()}

    count = _write_corpus(_corpus_titles(), corpus_by_title, "manifest.full.json")

    assert count == len(_corpus_titles())
    import json

    manifest = json.loads((tmp_path / "manifest.full.json").read_text())
    assert {entry["title"] for entry in manifest} == _corpus_titles()


def test_build_generation_full_mode_uses_all_retrieval_cases_and_all_null_queries():
    retrieval_cases = _build_retrieval(
        _queries(), _corpus_titles(), random.Random(1), sample_per_type=None, max_evidence=None
    )

    generation = _build_generation(
        _queries(), retrieval_cases, random.Random(1), answerable_per_type=None, null_sample=None
    )

    refusal_cases = [c for c in generation if c["expect_refusal"]]
    answerable_cases = [c for c in generation if not c["expect_refusal"]]
    assert len(answerable_cases) == len(retrieval_cases)
    assert len(refusal_cases) == 4  # every null_query case


def test_build_generation_sampled_mode_slices_retrieval_cases():
    retrieval_cases = _build_retrieval(
        _queries(), _corpus_titles(), random.Random(1), sample_per_type=None, max_evidence=None
    )

    generation = _build_generation(
        _queries(), retrieval_cases, random.Random(1), answerable_per_type=1, null_sample=2
    )

    answerable_cases = [c for c in generation if not c["expect_refusal"]]
    refusal_cases = [c for c in generation if c["expect_refusal"]]
    assert len(answerable_cases) == 1 * len(ROUTE_BY_TYPE)
    assert len(refusal_cases) == 2


@pytest.mark.asyncio
async def test_ingest_retries_upload_past_a_rate_limited_minute(tmp_path, monkeypatch):
    import evaluation.build_multihop_dataset as module

    # First retry's wait_random_exponential(multiplier=1, max=90) window is
    # random(0, 1) -- fast enough not to slow the test down noticeably.
    (tmp_path / "a.md").write_text("body")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"title": "A", "path": "a.md"}]))

    attempts = {"upload": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/v1/documents":
            return httpx.Response(200, json=[])
        if request.url.path == "/api/v1/documents":
            attempts["upload"] += 1
            if attempts["upload"] == 1:
                return httpx.Response(429, json={"detail": "rate limited"})
            return httpx.Response(200, json={"ingestion_job": {"id": "job-1"}})
        return httpx.Response(200, json={"status": "complete"})

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **_: real_async_client(transport=httpx.MockTransport(handler), timeout=30),
    )

    await ingest("http://test", manifest, timeout_seconds=30)

    assert attempts["upload"] == 2


@pytest.mark.asyncio
async def test_ingest_skips_titles_already_present_for_the_tenant(tmp_path, monkeypatch):
    import evaluation.build_multihop_dataset as module

    (tmp_path / "a.md").write_text("body")
    (tmp_path / "b.md").write_text("body")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [{"title": "Already ingested", "path": "a.md"}, {"title": "New", "path": "b.md"}]
        )
    )

    uploaded_titles = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/v1/documents":
            return httpx.Response(200, json=[{"title": "Already ingested"}])
        if request.url.path == "/api/v1/documents":
            uploaded_titles.append(request.content)
            return httpx.Response(200, json={"ingestion_job": {"id": "job-1"}})
        return httpx.Response(200, json={"status": "complete"})

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **_: real_async_client(transport=httpx.MockTransport(handler), timeout=30),
    )

    await ingest("http://test", manifest, timeout_seconds=30)

    assert len(uploaded_titles) == 1  # only "New" got uploaded, "Already ingested" was skipped
