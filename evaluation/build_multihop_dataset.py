"""Build MultiHop-RAG-derived eval fixtures for route accuracy, retrieval recall,
and grounded-generation testing.

Source: https://huggingface.co/datasets/yixuantt/MultiHopRAG (ODC-BY licensed).
2,556 queries over 609 news articles, with question_type in
{inference_query, comparison_query, temporal_query, null_query}. The first three
map onto this app's Route enum; null_query ("unanswerable from this corpus") does
NOT map to Route.OUT_OF_SCOPE -- this app's OUT_OF_SCOPE route is a jailbreak/abuse
guard (see adaptive_router.py), not a knowledge-boundary classifier. null_query is
used only for the generation-eval refusal check.

Usage:
    python evaluation/build_multihop_dataset.py build            # write fixtures
    python evaluation/build_multihop_dataset.py ingest [--base-url URL]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import httpx

HF_BASE = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"
ROUTE_BY_TYPE = {
    "inference_query": "multi_hop",
    "comparison_query": "comparison",
    "temporal_query": "temporal_causal",
}
RETRIEVAL_SAMPLE_PER_TYPE = 10
GOLDEN_SAMPLE_PER_TYPE = 40
GENERATION_ANSWERABLE_PER_TYPE = 3
GENERATION_NULL_SAMPLE = 10
SEED = 42

HERE = Path(__file__).parent
CACHE_DIR = HERE / "multihop_cache"
CORPUS_DIR = HERE / "multihop_corpus"


def _download(name: str) -> list[dict]:
    cache_path = CACHE_DIR / name
    if not cache_path.exists():
        CACHE_DIR.mkdir(exist_ok=True)
        response = httpx.get(f"{HF_BASE}/{name}", follow_redirects=True, timeout=120)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    return json.loads(cache_path.read_text())


def _safe_filename(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")[:80]


def _build_golden(queries: list[dict], rng: random.Random) -> list[dict]:
    golden = []
    for qtype, route in ROUTE_BY_TYPE.items():
        pool = [item for item in queries if item["question_type"] == qtype]
        for item in rng.sample(pool, min(GOLDEN_SAMPLE_PER_TYPE, len(pool))):
            golden.append({"query": item["query"], "route": route})
    return golden


def _build_retrieval(
    queries: list[dict], corpus_titles: set[str], rng: random.Random
) -> list[dict]:
    retrieval_cases = []
    for qtype, route in ROUTE_BY_TYPE.items():
        pool = [
            item
            for item in queries
            if item["question_type"] == qtype
            and 2 <= len(item["evidence_list"]) <= 4
            and all(e["title"] in corpus_titles for e in item["evidence_list"])
        ]
        for item in rng.sample(pool, min(RETRIEVAL_SAMPLE_PER_TYPE, len(pool))):
            titles = sorted({e["title"] for e in item["evidence_list"]})
            retrieval_cases.append(
                {
                    "name": item["query"][:80],
                    "query": item["query"],
                    "route": route,
                    "tenant_id": "eval",
                    "groups": ["research"],
                    "expected_document_titles": titles,
                }
            )
    return retrieval_cases


def _build_generation(
    queries: list[dict], retrieval_cases: list[dict], rng: random.Random
) -> list[dict]:
    null_pool = [item for item in queries if item["question_type"] == "null_query"]
    generation_cases = [
        {**case, "expect_refusal": False}
        for case in retrieval_cases[: GENERATION_ANSWERABLE_PER_TYPE * len(ROUTE_BY_TYPE)]
    ]
    generation_cases += [
        {
            "name": item["query"][:80],
            "query": item["query"],
            "route": None,
            "tenant_id": "eval",
            "groups": ["research"],
            "expected_document_titles": [],
            "expect_refusal": True,
        }
        for item in rng.sample(null_pool, GENERATION_NULL_SAMPLE)
    ]
    return generation_cases


def _write_corpus(retrieval_cases: list[dict], corpus_by_title: dict[str, dict]) -> int:
    required_titles = {
        title for case in retrieval_cases for title in case["expected_document_titles"]
    }
    CORPUS_DIR.mkdir(exist_ok=True)
    manifest = []
    for title in sorted(required_titles):
        doc = corpus_by_title[title]
        path = CORPUS_DIR / f"{_safe_filename(title)}.md"
        path.write_text(f"# {doc['title']}\n\n{doc['body']}\n")
        manifest.append({"title": doc["title"], "path": path.name})
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return len(required_titles)


def build() -> None:
    queries = _download("MultiHopRAG.json")
    corpus = _download("corpus.json")
    corpus_titles = {item["title"] for item in corpus}
    corpus_by_title = {item["title"]: item for item in corpus}
    rng = random.Random(SEED)

    golden = _build_golden(queries, rng)
    (HERE / "multihop_golden_dataset.json").write_text(json.dumps(golden, indent=2))

    retrieval_cases = _build_retrieval(queries, corpus_titles, rng)
    (HERE / "multihop_retrieval_dataset.json").write_text(json.dumps(retrieval_cases, indent=2))

    generation_cases = _build_generation(queries, retrieval_cases, rng)
    (HERE / "generation_dataset.json").write_text(json.dumps(generation_cases, indent=2))

    article_count = _write_corpus(retrieval_cases, corpus_by_title)

    print(f"golden: {len(golden)} cases (route accuracy, no ingestion needed)")
    print(f"retrieval: {len(retrieval_cases)} cases, {article_count} articles to ingest")
    print(f"generation: {len(generation_cases)} cases")


async def ingest(base_url: str, concurrency: int = 6, timeout_seconds: int = 900) -> None:
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    headers = {"X-Tenant-ID": "eval", "X-User-ID": "eval-builder", "X-Groups": "research"}
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=30) as client:

        async def upload(entry: dict) -> str:
            async with semaphore:
                content = (CORPUS_DIR / entry["path"]).read_bytes()
                response = await client.post(
                    f"{base_url}/api/v1/documents",
                    headers=headers,
                    files={"file": (entry["path"], content, "text/markdown")},
                    data={"title": entry["title"], "acl_groups": "research"},
                )
                response.raise_for_status()
                return response.json()["ingestion_job"]["id"]

        job_ids = await asyncio.gather(*(upload(entry) for entry in manifest))
        print(f"submitted {len(job_ids)} ingestion jobs")
        pending = set(job_ids)
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while pending and asyncio.get_event_loop().time() < deadline:
            for job_id in list(pending):
                # Stay well under requests_per_minute (shared by all polls in this
                # sweep, same tenant): space individual polls out instead of
                # firing len(pending) requests in the same instant.
                await asyncio.sleep(0.5)
                response = await client.get(
                    f"{base_url}/api/v1/ingestion-jobs/{job_id}", headers=headers
                )
                if response.status_code == 429:
                    continue
                response.raise_for_status()
                status = response.json()["status"]
                if status in {"complete", "failed"}:
                    pending.discard(job_id)
                    if status == "failed":
                        print(f"WARNING: job {job_id} failed")
            if pending:
                print(f"{len(pending)} jobs still pending...")
                await asyncio.sleep(5)
        if pending:
            raise TimeoutError(
                f"{len(pending)} ingestion jobs still pending after {timeout_seconds}s"
            )
        print("all ingestion jobs settled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "ingest"])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.action == "build":
        build()
    else:
        asyncio.run(ingest(arguments.base_url, arguments.concurrency))
