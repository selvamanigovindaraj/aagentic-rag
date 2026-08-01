from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
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
        "question_type": case.get("question_type"),
        "expect_refusal": case["expect_refusal"],
        "refused": False,
        "refusal_correct": False,
        "answer_source": None,
        "grounded": None,
        "citations_valid": None,
        "answer_correct": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _answer_correct(case: dict, answer: str, refused: bool) -> bool | None:
    """Substring match against the source dataset's gold answer (a named entity
    or short phrase in this corpus) -- cheap, deterministic, no extra LLM call.
    None when there's no gold answer to check against (refusal cases)."""
    expected = case.get("expected_answer")
    if not expected or refused:
        return None
    return expected.casefold() in answer.casefold()


def _verified_metric(is_fallback: bool, actual: int, expected: int) -> bool | None:
    # _numbered_fallback sets actual == expected by construction, not by
    # checking the evidence is actually relevant -- a plain equality check
    # would read that dump as tautologically 100% grounded/cited.
    if is_fallback:
        return False
    return actual == expected if expected else None


def _case_result(case: dict, result: dict) -> dict:
    answer = result.get("answer", "")
    refused = REFUSAL_TEXT in answer
    answer_source = result.get("answer_source")
    is_fallback = answer_source == "fallback"
    return {
        "name": case["name"],
        "question_type": case.get("question_type"),
        "expect_refusal": case["expect_refusal"],
        "refused": refused,
        "refusal_correct": refused == case["expect_refusal"],
        "answer_source": answer_source,
        "grounded": _verified_metric(
            is_fallback, result.get("grounded_claims", 0), result.get("generated_claims", 0)
        ),
        "citations_valid": _verified_metric(
            is_fallback,
            result.get("valid_citation_references", 0),
            result.get("citation_references", 0),
        ),
        "answer_correct": _answer_correct(case, answer, refused),
    }


def _breakdown_by_type(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in results:
        grouped[str(item.get("question_type"))].append(item)

    def _rate(items: list[dict], key: str) -> float | None:
        values = [item[key] for item in items if item.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        bucket: {
            "count": len(items),
            "refusal_accuracy": _rate(items, "refusal_correct"),
            "grounded_claim_rate": _rate(items, "grounded"),
            "citation_validity": _rate(items, "citations_valid"),
            "answer_accuracy": _rate(items, "answer_correct"),
        }
        for bucket, items in grouped.items()
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


def _load_resume_results(results_path: Path) -> dict[str, dict]:
    if not results_path.exists():
        return {}
    return {
        json.loads(line)["name"]: json.loads(line)
        for line in results_path.read_text().splitlines()
        if line.strip()
    }


async def _run_all_cases(
    pipeline: RagPipeline,
    cases: list[dict],
    concurrency: int,
    results_path: Path | None,
) -> list[dict]:
    """Bounded-concurrency case runner with incremental persistence: at full
    dataset scale (~2500 live-LLM-pipeline calls) a crash partway through must
    not lose already-computed results, so each result is appended to
    results_path as soon as it's ready, and a rerun with the same path skips
    cases already recorded there."""
    done = _load_resume_results(results_path) if results_path else {}
    pending = [case for case in cases if case["name"] not in done]
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def _run_and_persist(case: dict) -> dict:
        async with semaphore:
            result = await _run_case(pipeline, case)
        if results_path:
            async with write_lock:
                with results_path.open("a") as handle:
                    handle.write(json.dumps(result) + "\n")
        return result

    new_results = await asyncio.gather(*(_run_and_persist(case) for case in pending))
    return list(done.values()) + list(new_results)


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


async def evaluate(
    dataset: Path, *, concurrency: int = 8, results_path: Path | None = None
) -> dict:
    async with _evaluation_session(dataset, Settings()) as (cases, pipeline):
        results = await _run_all_cases(pipeline, cases, concurrency, results_path)

    answerable = [item for item in results if not item["expect_refusal"]]
    grounded_checks = [item["grounded"] for item in answerable if item["grounded"] is not None]
    citation_checks = [
        item["citations_valid"] for item in answerable if item["citations_valid"] is not None
    ]
    answer_checks = [
        item["answer_correct"] for item in answerable if item["answer_correct"] is not None
    ]
    # Errored cases (e.g. transient rate limits -- ChatLiteLLM already retries
    # internally via max_retries, so what surfaces here already exhausted
    # that) are excluded from the correctness rates above via the `is not
    # None` filters; report the count separately so a wall of errors doesn't
    # silently masquerade as a low grounded/citation score.
    errored_cases = sum(1 for item in results if item.get("error"))
    return {
        "refusal_accuracy": sum(item["refusal_correct"] for item in results) / len(results),
        "grounded_claim_rate": (
            sum(grounded_checks) / len(grounded_checks) if grounded_checks else None
        ),
        "citation_validity": (
            sum(citation_checks) / len(citation_checks) if citation_checks else None
        ),
        "answer_accuracy": (
            sum(answer_checks) / len(answer_checks) if answer_checks else None
        ),
        "errored_cases": errored_cases,
        "by_question_type": _breakdown_by_type(results),
        "cases": results,
    }


def _refusal_evaluator(run, example) -> dict:
    return {"key": "refusal_correct", "score": run.outputs["refusal_correct"]}


def _grounded_evaluator(run, example) -> dict:
    return {"key": "grounded", "score": run.outputs["grounded"]}


def _citations_evaluator(run, example) -> dict:
    return {"key": "citations_valid", "score": run.outputs["citations_valid"]}


def _answer_correct_evaluator(run, example) -> dict:
    return {"key": "answer_correct", "score": run.outputs["answer_correct"]}


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
        "answer_accuracy": _rate("answer_correct"),
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
    evaluators = [
        _refusal_evaluator, _grounded_evaluator, _citations_evaluator, _answer_correct_evaluator,
    ]
    async with _evaluation_session(dataset, settings or Settings()) as (cases, pipeline):
        ensure_dataset(client, dataset_name, cases, description="Generation eval golden set")

        async def predict(inputs: dict) -> dict:
            return await _run_case(pipeline, inputs)

        results = await aevaluate(
            predict, data=dataset_name, evaluators=evaluators,
            experiment_prefix="generation-eval", client=client, max_concurrency=8,
        )
    return await _summarize_langsmith_results(results)


if __name__ == "__main__":
    from app.core.tracing import configure_tracing

    configure_tracing()
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
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Max concurrent live pipeline runs"
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Append per-case results here as they complete; rerunning with the same "
        "path skips cases already recorded (resume after a crash/timeout)",
    )
    arguments = parser.parse_args()
    if arguments.langsmith:
        report = asyncio.run(evaluate_via_langsmith(arguments.dataset, "generation-eval"))
    else:
        report = asyncio.run(
            evaluate(
                arguments.dataset,
                concurrency=arguments.concurrency,
                results_path=arguments.results_path,
            )
        )
    print(json.dumps(report, indent=2))
    thresholds = json.loads(arguments.thresholds.read_text())
    failed = (
        (report["refusal_accuracy"] or 0.0) < thresholds["refusal_accuracy"]
        or (report["grounded_claim_rate"] or 1.0) < thresholds["grounded_claim_rate"]
        or (report["citation_validity"] or 1.0) < thresholds["citation_validity"]
    )
    raise SystemExit(failed)
