from __future__ import annotations

import argparse
import asyncio
import json
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


async def evaluate(dataset: Path, k: int = 20, fetch_limit: int | None = None) -> dict:
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
    results = []
    try:
        for case in cases:
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
            results.append(
                {
                    "name": case["name"],
                    "recall_at_20": recall_at_k(
                        titles, set(case["expected_document_titles"]), k
                    ),
                    "retrieved_document_titles": list(dict.fromkeys(titles)),
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                }
            )
    finally:
        await graph.driver.close()
        await pool.close()
    return {
        "retrieval_recall_at_20": sum(item["recall_at_20"] for item in results) / len(results),
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
    arguments = parser.parse_args()
    report = asyncio.run(evaluate(arguments.dataset, fetch_limit=arguments.fetch_limit))
    print(json.dumps(report, indent=2))
    raise SystemExit(report["retrieval_recall_at_20"] < arguments.threshold)
