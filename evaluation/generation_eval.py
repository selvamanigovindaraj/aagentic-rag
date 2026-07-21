from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import asyncpg
from app.components.llm import DeepSeekGateway
from app.components.retrieval import (
    ActiveDocumentRetriever,
    CachedRetriever,
    CompositeRetriever,
    Neo4jRetriever,
    RerankingRetriever,
    WeaviateRetriever,
)
from app.components.voyage import VoyageGateway
from app.core.config import Settings
from app.repositories.postgres import PostgresStore
from app.schemas.domain import AuthContext
from app.services.rag_pipeline import RagPipeline
from app.services.semantic_cache import SemanticCache
from redis.asyncio import Redis

REFUSAL_TEXT = "could not find enough authorized evidence"


async def evaluate(dataset: Path) -> dict:
    settings = Settings()
    cases = json.loads(await asyncio.to_thread(dataset.read_text))
    pool = await asyncpg.create_pool(settings.database_url)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    voyage = VoyageGateway(settings)
    graph = Neo4jRetriever(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.index_version,
        embedder=voyage,
    )
    retriever = ActiveDocumentRetriever(
        CachedRetriever(
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
                settings.max_reranked_candidates,
            ),
            SemanticCache(redis),
            settings.index_version,
        ),
        PostgresStore(pool, settings.index_version),
    )
    pipeline = RagPipeline(retriever, settings, DeepSeekGateway(settings))
    results = []
    try:
        for case in cases:
            try:
                result = await pipeline.run(
                    case["query"],
                    AuthContext(
                        subject="generation-evaluator",
                        tenant_id=case["tenant_id"],
                        groups=frozenset(case.get("groups", [])),
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - one bad case must not sink the report
                results.append(
                    {
                        "name": case["name"],
                        "expect_refusal": case["expect_refusal"],
                        "refused": False,
                        "refusal_correct": False,
                        "grounded": None,
                        "citations_valid": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            answer = result.get("answer", "")
            refused = REFUSAL_TEXT in answer
            generated_claims = result.get("generated_claims", 0)
            citation_references = result.get("citation_references", 0)
            results.append(
                {
                    "name": case["name"],
                    "expect_refusal": case["expect_refusal"],
                    "refused": refused,
                    "refusal_correct": refused == case["expect_refusal"],
                    "grounded": (
                        result.get("grounded_claims", 0) == generated_claims
                        if generated_claims
                        else None
                    ),
                    "citations_valid": (
                        result.get("valid_citation_references", 0) == citation_references
                        if citation_references
                        else None
                    ),
                }
            )
    finally:
        await graph.driver.close()
        await redis.aclose()
        await pool.close()

    answerable = [item for item in results if not item["expect_refusal"]]
    grounded_checks = [item["grounded"] for item in answerable if item["grounded"] is not None]
    citation_checks = [
        item["citations_valid"] for item in answerable if item["citations_valid"] is not None
    ]
    return {
        "refusal_accuracy": sum(item["refusal_correct"] for item in results) / len(results),
        "grounded_claim_rate": (
            sum(grounded_checks) / len(grounded_checks) if grounded_checks else None
        ),
        "citation_validity": (
            sum(citation_checks) / len(citation_checks) if citation_checks else None
        ),
        "cases": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).with_name("generation_dataset.json")
    )
    parser.add_argument(
        "--thresholds", type=Path, default=Path(__file__).with_name("thresholds.json")
    )
    arguments = parser.parse_args()
    report = asyncio.run(evaluate(arguments.dataset))
    print(json.dumps(report, indent=2))
    thresholds = json.loads(arguments.thresholds.read_text())
    failed = (
        report["refusal_accuracy"] < thresholds["refusal_accuracy"]
        or (report["grounded_claim_rate"] or 1.0) < thresholds["grounded_claim_rate"]
        or (report["citation_validity"] or 1.0) < thresholds["citation_validity"]
    )
    raise SystemExit(failed)
