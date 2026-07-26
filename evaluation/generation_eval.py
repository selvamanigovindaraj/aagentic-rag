from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
from app.components.hybrid_retriever import Neo4jRetriever, build_retrieval_stack
from app.components.llm import LiteLLMGateway
from app.components.voyage import VoyageGateway
from app.core.config import Settings
from app.repositories.postgres import PostgresStore
from app.schemas.domain import AuthContext
from app.services.rag_pipeline import RagPipeline
from redis.asyncio import Redis

try:
    from evaluation.langsmith_support import ensure_dataset
except ImportError:  # running as `python evaluation/generation_eval.py`, not `-m`
    from langsmith_support import ensure_dataset

REFUSAL_TEXT = "could not find enough authorized evidence"


def _error_case_result(case: dict, exc: Exception) -> dict:
    return {
        "name": case["name"],
        "expect_refusal": case["expect_refusal"],
        "refused": False,
        "refusal_correct": False,
        "grounded": None,
        "citations_valid": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _case_result(case: dict, result: dict) -> dict:
    answer = result.get("answer", "")
    refused = REFUSAL_TEXT in answer
    generated_claims = result.get("generated_claims", 0)
    citation_references = result.get("citation_references", 0)
    return {
        "name": case["name"],
        "expect_refusal": case["expect_refusal"],
        "refused": refused,
        "refusal_correct": refused == case["expect_refusal"],
        "grounded": (
            result.get("grounded_claims", 0) == generated_claims if generated_claims else None
        ),
        "citations_valid": (
            result.get("valid_citation_references", 0) == citation_references
            if citation_references
            else None
        ),
    }


async def _run_case(pipeline: RagPipeline, case: dict) -> dict:
    auth = AuthContext(
        subject="generation-evaluator",
        tenant_id=case["tenant_id"],
        groups=frozenset(case.get("groups", [])),
    )
    try:
        result = await pipeline.run(case["query"], auth)
    except Exception as exc:  # noqa: BLE001 - one bad case must not sink the report
        return _error_case_result(case, exc)
    return _case_result(case, result)


@asynccontextmanager
async def _evaluation_session(dataset: Path, settings: Settings):
    """Loads golden cases and wires a live RagPipeline; shared by both eval modes."""
    cases = json.loads(await asyncio.to_thread(dataset.read_text))
    pool = await asyncpg.create_pool(settings.database_url)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    voyage = VoyageGateway(settings)
    graph = Neo4jRetriever(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password,
        settings.index_version, embedder=voyage,
    )
    store = PostgresStore(pool, settings.index_version)
    retriever = build_retrieval_stack(settings, voyage, graph, redis, store)
    pipeline = RagPipeline(retriever, settings, LiteLLMGateway(settings))
    try:
        yield cases, pipeline
    finally:
        await graph.driver.close()
        await redis.aclose()
        await pool.close()


async def evaluate(dataset: Path) -> dict:
    async with _evaluation_session(dataset, Settings()) as (cases, pipeline):
        results = [await _run_case(pipeline, case) for case in cases]

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


def _refusal_evaluator(run, example) -> dict:
    return {"key": "refusal_correct", "score": run.outputs["refusal_correct"]}


def _grounded_evaluator(run, example) -> dict:
    return {"key": "grounded", "score": run.outputs["grounded"]}


def _citations_evaluator(run, example) -> dict:
    return {"key": "citations_valid", "score": run.outputs["citations_valid"]}


async def _summarize_langsmith_results(results) -> dict:
    scores: dict[str, list] = {}
    async for row in results:
        for item in row["evaluation_results"].results:
            scores.setdefault(item.key, []).append(item.score)

    def _rate(key: str) -> float | None:
        values = [value for value in scores.get(key, []) if value is not None]
        return sum(values) / len(values) if values else None

    return {
        "refusal_accuracy": _rate("refusal_correct"),
        "grounded_claim_rate": _rate("grounded"),
        "citation_validity": _rate("citations_valid"),
        "experiment_name": results.experiment_name,
        "experiment_url": results.url,
    }


async def evaluate_via_langsmith(
    dataset: Path, dataset_name: str, settings: Settings | None = None
) -> dict:
    """Same generation eval, via LangSmith's aevaluate() -- recorded there, not just stdout."""
    from langsmith import Client
    from langsmith.evaluation import aevaluate

    client = Client()
    evaluators = [_refusal_evaluator, _grounded_evaluator, _citations_evaluator]
    async with _evaluation_session(dataset, settings or Settings()) as (cases, pipeline):
        ensure_dataset(client, dataset_name, cases, description="Generation eval golden set")

        async def predict(inputs: dict) -> dict:
            return await _run_case(pipeline, inputs)

        results = await aevaluate(
            predict, data=dataset_name, evaluators=evaluators,
            experiment_prefix="generation-eval", client=client,
        )
    return await _summarize_langsmith_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).with_name("generation_dataset.json")
    )
    parser.add_argument(
        "--thresholds", type=Path, default=Path(__file__).with_name("thresholds.json")
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Run through LangSmith's aevaluate(), syncing the dataset and logging an experiment",
    )
    arguments = parser.parse_args()
    if arguments.langsmith:
        report = asyncio.run(evaluate_via_langsmith(arguments.dataset, "generation-eval"))
    else:
        report = asyncio.run(evaluate(arguments.dataset))
    print(json.dumps(report, indent=2))
    thresholds = json.loads(arguments.thresholds.read_text())
    failed = (
        (report["refusal_accuracy"] or 0.0) < thresholds["refusal_accuracy"]
        or (report["grounded_claim_rate"] or 1.0) < thresholds["grounded_claim_rate"]
        or (report["citation_validity"] or 1.0) < thresholds["citation_validity"]
    )
    raise SystemExit(failed)
