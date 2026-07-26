from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import asyncpg
from app.components.retrieval import (
    ActiveDocumentRetriever,
    CompositeRetriever,
    Neo4jRetriever,
    RerankingRetriever,
    WeaviateRetriever,
)
from app.components.voyage import VoyageGateway
from app.core.config import Settings
from app.repositories.postgres import PostgresStore
from app.schemas.domain import AuthContext, Route


def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 1.0
    return len(set(retrieved[:k]) & expected) / len(expected)


def evidence_subset_match(retrieved: list[str], expected: set[str], k: int) -> bool:
    """Whether every expected title landed somewhere in the top-k -- distinct
    signal from recall_at_k's partial credit (e.g. 3/4 expected -> recall=0.75,
    match=False), and deliberately NOT strict set equality: top-k=20 retrieving
    more than the 2-4 expected titles is normal, so equality would read ~0% always."""
    return expected <= set(retrieved[:k])


def _breakdown_by(results: list[dict], key: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in results:
        grouped[str(item[key])].append(item)
    return {
        bucket: {
            "count": len(items),
            "recall_at_20": sum(i["recall_at_20"] for i in items) / len(items),
            "evidence_subset_match_rate": sum(i["evidence_subset_match"] for i in items)
            / len(items),
        }
        for bucket, items in grouped.items()
    }


async def _evaluate_case(retriever, case: dict, k: int, fetch_limit: int) -> dict:
    started = perf_counter()
    evidence = await retriever.retrieve(
        case["query"],
        Route(case["route"]),
        AuthContext(
            subject="retrieval-evaluator",
            tenant_id=case["tenant_id"],
            groups=frozenset(case.get("groups", [])),
        ),
        fetch_limit,
    )
    titles = [item.document_title for item in evidence]
    expected = set(case["expected_document_titles"])
    return {
        "name": case["name"],
        "question_type": case.get("question_type"),
        "evidence_count": case.get("evidence_count"),
        "recall_at_20": recall_at_k(titles, expected, k),
        "evidence_subset_match": evidence_subset_match(titles, expected, k),
        "retrieved_document_titles": list(dict.fromkeys(titles)),
        "duration_ms": round((perf_counter() - started) * 1000, 2),
    }


async def _run_all_cases(
    retriever, cases: list[dict], k: int, fetch_limit: int, concurrency: int
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(case: dict) -> dict:
        async with semaphore:
            return await _evaluate_case(retriever, case, k, fetch_limit)

    return list(await asyncio.gather(*(_bounded(case) for case in cases)))


async def evaluate(
    dataset: Path, k: int = 20, fetch_limit: int | None = None, concurrency: int = 8
) -> dict:
    settings = Settings()
    # Recall@k means "top k after ranking", not "top k of whatever the first-stage
    # fetch happened to return". Production (RagPipeline._retrieve) always fetches
    # settings.max_retrieval_candidates candidates before RerankingRetriever narrows
    # to max_reranked_candidates -- fetching only k here (as this script used to)
    # under-measures recall for anything ranked outside the first k raw candidates,
    # e.g. a fact buried in an otherwise off-topic multi-story roundup article.
    fetch_limit = fetch_limit or settings.max_retrieval_candidates
    cases = json.loads(await asyncio.to_thread(dataset.read_text))
    pool = await asyncpg.create_pool(settings.database_url)
    voyage = VoyageGateway(settings)
    graph = Neo4jRetriever(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.index_version,
        embedder=voyage,
    )
    retriever = ActiveDocumentRetriever(
        RerankingRetriever(
            CompositeRetriever(
                WeaviateRetriever(
                    settings.weaviate_url,
                    settings.weaviate_api_key,
                    voyage,
                    settings.weaviate_collection,
                    settings.index_version,
                ),
                graph,
            ),
            voyage,
            k,
        ),
        PostgresStore(pool, settings.index_version),
    )
    try:
        results = await _run_all_cases(retriever, cases, k, fetch_limit, concurrency)
    finally:
        await graph.driver.close()
        await pool.close()
    by_evidence_count = _breakdown_by(
        [item for item in results if item["evidence_count"] is not None], "evidence_count"
    )
    return {
        "retrieval_recall_at_20": sum(item["recall_at_20"] for item in results) / len(results),
        "evidence_subset_match_rate": sum(item["evidence_subset_match"] for item in results)
        / len(results),
        "by_question_type": _breakdown_by(results, "question_type"),
        "by_evidence_count": by_evidence_count,
        "cases": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("retrieval_dataset.json"),
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument(
        "--fetch-limit",
        type=int,
        default=None,
        help="First-stage candidate count before reranking (default: max_retrieval_candidates)",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    arguments = parser.parse_args()
    report = asyncio.run(
        evaluate(
            arguments.dataset,
            fetch_limit=arguments.fetch_limit,
            concurrency=arguments.concurrency,
        )
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(report["retrieval_recall_at_20"] < arguments.threshold)
