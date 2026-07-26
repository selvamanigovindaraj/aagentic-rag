import asyncio
import json
import logging
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from uuid import UUID

import asyncpg
import httpx
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from opentelemetry import trace
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .components.hybrid_retriever import Neo4jRetriever, build_retrieval_stack
from .components.llm import LiteLLMGateway
from .components.object_store import S3ObjectStore
from .components.voyage import VoyageGateway
from .core.config import Settings, get_settings
from .core.errors import AppError
from .core.logging import configure_logging
from .core.tracing import configure_tracing
from .repositories.postgres import PostgresStore
from .repositories.store import Store
from .schemas.domain import (
    AuthContext,
    ChatRun,
    Citation,
    CorpusRebuild,
    Document,
    IngestionJob,
    JobStatus,
)
from .services.events import RedisEventBroker
from .services.graph_index import Neo4jIndexer
from .services.ingestion import WeaviateIndexer, chunk_document, parse_document
from .services.rag_pipeline import RagPipeline

logger = logging.getLogger(__name__)
# OTEL's API is a safe no-op when no SDK/exporter is configured (Phoenix
# disabled), so this doesn't need gating on settings.phoenix_enabled -- only
# parse/chunk/vector-index/graph-index are wrapped here because those aren't
# LangChain calls and so aren't already covered by LangChainInstrumentor;
# the RAPTOR/graph-extraction LLM calls inside indexer.index/graph_indexer.index
# still nest correctly as their own auto-instrumented child spans.
_tracer = trace.get_tracer("app.worker.ingestion")


@dataclass
class _JobContext:
    job: IngestionJob
    document: Document
    store: Store
    indexer: WeaviateIndexer
    graph_indexer: Neo4jIndexer
    object_store: S3ObjectStore | None
    path: Path
    index_version: str


async def _load_job(job_id: UUID, store: Store) -> tuple[IngestionJob, Document] | None:
    """Returns the job with its document, or None when there is nothing to process."""
    job = await store.get_job_by_id(job_id)
    if not job or job.status == JobStatus.COMPLETE:
        return None
    document = await store.get_document(job.document_id)
    if not document or not document.object_key:
        job.status, job.stage, job.error = JobStatus.FAILED, "failed", "Document source is missing"
        await store.update_job(job)
        return None
    return job, document


async def process_job(
    job_id: UUID,
    store: Store,
    indexer: WeaviateIndexer,
    graph_indexer: Neo4jIndexer,
    root: Path,
    object_store: S3ObjectStore | None = None,
    index_version: str = "v1",
) -> None:
    loaded = await _load_job(job_id, store)
    if not loaded:
        return
    job, document = loaded
    initial_model_calls = getattr(indexer.models, "calls", 0)
    job.index_version = index_version
    job.status, job.stage, job.progress = JobStatus.RUNNING, "parsing", 10
    await store.update_job(job)
    previous = await store.get_document(document.revision_of) if document.revision_of else None
    ctx = _JobContext(
        job, document, store, indexer, graph_indexer, object_store,
        root / document.object_key, index_version,
    )
    try:
        if object_store and job.operation != "delete":
            ctx.path.parent.mkdir(parents=True, exist_ok=True)
            await object_store.download(document.object_key, ctx.path)
        if job.operation == "delete":
            await _delete_document(ctx)
            return
        await _ingest_document(ctx, previous, initial_model_calls)
    except AppError as exc:
        job.status, job.stage, job.error = JobStatus.FAILED, "failed", exc.message
        job.worker_id, job.lease_until = None, None
    except Exception:
        logger.exception("Ingestion failed", extra={"job_id": str(job_id)})
        _mark_for_retry(job)
    job.model_calls = getattr(indexer.models, "calls", 0) - initial_model_calls
    await store.update_job(job)
    if object_store:
        await asyncio.to_thread(ctx.path.unlink, missing_ok=True)


def _mark_for_retry(job: IngestionJob) -> None:
    job.status = JobStatus.QUEUED if job.attempts < 3 else JobStatus.FAILED
    job.stage = "retrying" if job.status == JobStatus.QUEUED else "failed"
    job.error = "Indexing will retry" if job.status == JobStatus.QUEUED else "Indexing failed"
    job.worker_id, job.lease_until = None, None


async def _delete_document(ctx: _JobContext) -> None:
    job, document = ctx.job, ctx.document
    job.stage, job.progress = "deleting_indexes", 40
    await ctx.store.update_job(job)
    await ctx.indexer.delete(document)
    await ctx.graph_indexer.delete(document)
    await _delete_source_and_layout(ctx)
    await ctx.store.finalize_document_delete(document.id)
    await ctx.store.request_corpus_rebuild(
        document.tenant_id, document.acl_groups, ctx.index_version
    )
    job.status, job.stage, job.progress = JobStatus.COMPLETE, "complete", 100
    job.worker_id, job.lease_until = None, None
    await ctx.store.update_job(job)


async def _delete_source_and_layout(ctx: _JobContext) -> None:
    # The layout manifest lives beside the original and must die with it.
    if ctx.object_store:
        await ctx.object_store.delete(ctx.document.object_key)
        await ctx.object_store.delete(f"{ctx.document.object_key}.layout.json")
    else:
        await asyncio.to_thread(ctx.path.unlink, missing_ok=True)
        await asyncio.to_thread(
            ctx.path.with_suffix(ctx.path.suffix + ".layout.json").unlink, missing_ok=True
        )


async def _ingest_document(
    ctx: _JobContext, previous: Document | None, initial_model_calls: int
) -> None:
    job, document = ctx.job, ctx.document
    with _tracer.start_as_current_span(
        "ingest_document",
        attributes={
            "document.id": str(document.id),
            "document.title": document.title,
            "tenant.id": document.tenant_id,
        },
    ):
        with _tracer.start_as_current_span("parse_document"):
            parsed = parse_document(ctx.path)
        await _store_layout_manifest(ctx, parsed)
        with _tracer.start_as_current_span("chunk_document") as chunk_span:
            chunks = chunk_document(parsed)
            chunk_span.set_attribute("chunk.count", len(chunks))
        job.stage, job.progress = "vector_indexing", 35
        await ctx.store.update_job(job)
        with _tracer.start_as_current_span("vector_index"):
            await ctx.indexer.index(document, ctx.path, chunks=chunks)
        job.model_calls = getattr(ctx.indexer.models, "calls", 0) - initial_model_calls
        job.stage, job.progress = "graph_indexing", 75
        await ctx.store.update_job(job)
        with _tracer.start_as_current_span("graph_index"):
            await ctx.graph_indexer.index(document, chunks)
        job.model_calls = getattr(ctx.indexer.models, "calls", 0) - initial_model_calls
        await ctx.store.activate_document(document)
        if previous:
            await ctx.indexer.delete(previous)
            await ctx.graph_indexer.delete(previous)
        job.stage, job.progress = "corpus_queued", 90
        await ctx.store.update_job(job)
        await ctx.store.request_corpus_rebuild(
            document.tenant_id, document.acl_groups, ctx.index_version
        )
        job.status, job.stage, job.progress = JobStatus.COMPLETE, "complete", 100
        job.worker_id, job.lease_until = None, None


async def _store_layout_manifest(ctx: _JobContext, parsed) -> None:
    manifest = json.dumps(
        {
            "page_count": parsed.page_count,
            "pages_without_text": parsed.pages_without_text,
            "blocks": [asdict(block) for block in parsed.blocks],
            "artifacts": [asdict(artifact) for artifact in parsed.artifacts],
        }
    ).encode()
    if ctx.object_store:
        await ctx.object_store.put(
            f"{ctx.document.object_key}.layout.json", manifest, "application/json"
        )
    else:
        await asyncio.to_thread(
            ctx.path.with_suffix(ctx.path.suffix + ".layout.json").write_bytes, manifest
        )


async def process_corpus_rebuild(
    task: CorpusRebuild, store: Store, indexer: WeaviateIndexer
) -> None:
    document = Document(
        tenant_id=task.tenant_id,
        title="Corpus maintenance",
        acl_groups=task.acl_groups,
    )
    try:
        await indexer.rebuild_corpus(document)
        await store.finish_corpus_rebuild(task, True)
    except Exception:
        logger.exception(
            "Corpus rebuild failed",
            extra={"tenant_id": task.tenant_id, "acl_cohort": task.acl_cohort},
        )
        await store.finish_corpus_rebuild(task, False)


def _model_usage(models) -> tuple[int, int, float]:
    return (
        getattr(models, "input_tokens", 0),
        getattr(models, "output_tokens", 0),
        getattr(models, "estimated_cost_usd", 0.0),
    )


def _run_metrics(
    result: dict,
    citations: list[Citation],
    agent: RagPipeline,
    duration_ms: float,
    usage_before: tuple[int, int, float],
) -> dict:
    input_before, output_before, cost_before = usage_before
    input_now, output_now, cost_now = _model_usage(agent.models)
    return {
        "duration_ms": duration_ms,
        "model_calls": result.get("model_calls", 0),
        "retrieval_calls": result.get("retrieval_calls", 0),
        "confidence": result.get("confidence", 0.0),
        "citations": len(citations),
        "citation_verification": result.get("citation_verification", "unsupported"),
        "generated_claims": result.get("generated_claims", 0),
        "grounded_claims": result.get("grounded_claims", 0),
        "citation_references": result.get("citation_references", 0),
        "valid_citation_references": result.get("valid_citation_references", 0),
        "escalated": result.get("escalated", False),
        "critic_candidates": result.get("critic_candidates", 0),
        "critic_accepted": result.get("critic_accepted", 0),
        "input_tokens": input_now - input_before,
        "output_tokens": output_now - output_before,
        "estimated_cost_usd": round(cost_now - cost_before, 8),
        "flash_model": agent.settings.llm_flash_model,
        "pro_model": agent.settings.llm_pro_model,
        "index_version": agent.settings.index_version,
    }


async def _stream_tokens(events: RedisEventBroker, run: ChatRun) -> None:
    # Token events are operational streaming; the durable final answer remains in PostgreSQL.
    for token in (run.answer or "").split():
        await events.publish(run.id, {"type": "token", "content": token + " "})


async def _publish_final_answer(events: RedisEventBroker, run: ChatRun) -> None:
    await events.publish(
        run.id,
        {
            "type": "answer",
            "content": run.answer,
            "citations": [str(item) for item in run.citation_ids],
        },
    )
    await events.publish(run.id, {"type": "complete"})


async def _fail_or_requeue_run(run: ChatRun, store: Store, events: RedisEventBroker) -> None:
    run.status = "queued" if run.attempts < 3 else "failed"
    run.error = "Research will retry" if run.status == "queued" else "The research run failed"
    run.worker_id, run.lease_until = None, None
    await store.update_run(run)
    if run.status == "failed":
        await events.publish(run.id, {"type": "error", "message": run.error})


async def process_run(
    run: ChatRun, store: Store, agent: RagPipeline, events: RedisEventBroker
) -> None:
    started = perf_counter()
    usage_before = _model_usage(agent.models)
    try:
        await events.publish(run.id, {"type": "status", "stage": "planning"})
        await events.publish(run.id, {"type": "status", "stage": "searching"})
        result = await agent.run(
            run.query,
            AuthContext(subject=run.user_id, tenant_id=run.tenant_id, groups=frozenset(run.groups)),
            str(run.id),
        )
        run.route = result.get("route")
        await events.publish(run.id, {"type": "status", "stage": "verifying"})
        run.answer = result.get("answer")
        run.error = None
        citations = [Citation.model_validate(item) for item in result.get("citations", [])]
        run.citation_ids = [item.id for item in citations]
        duration_ms = round((perf_counter() - started) * 1000, 2)
        run.metrics = _run_metrics(result, citations, agent, duration_ms, usage_before)
        await store.save_citations(citations)
        await _stream_tokens(events, run)
        run.status, run.worker_id, run.lease_until = "complete", None, None
        await store.update_run(run)
        await _publish_final_answer(events, run)
    except Exception:
        logger.exception("Research run failed", extra={"run_id": str(run.id)})
        await _fail_or_requeue_run(run, store, events)


@dataclass
class _WorkerComponents:
    store: PostgresStore
    indexer: WeaviateIndexer
    graph_indexer: Neo4jIndexer
    graph_retriever: Neo4jRetriever
    object_store: S3ObjectStore
    agent: RagPipeline
    events: RedisEventBroker


async def _build_components(
    settings: Settings,
    redis: Redis,
    pool: asyncpg.Pool,
    client: httpx.AsyncClient,
    checkpointer: AsyncPostgresSaver,
) -> _WorkerComponents:
    models = LiteLLMGateway(settings)
    embedder = VoyageGateway(settings)
    object_store = S3ObjectStore(settings)
    await object_store.setup()
    indexer = WeaviateIndexer(
        settings.weaviate_url,
        settings.weaviate_api_key,
        embedder,
        models,
        client,
        settings.weaviate_collection,
        settings.index_version,
    )
    graph_indexer = Neo4jIndexer(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        models,
        embedder,
        settings.index_version,
    )
    store = PostgresStore(pool, settings.index_version)
    graph_retriever = Neo4jRetriever(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.index_version,
        embedder=embedder,
    )
    retriever = build_retrieval_stack(settings, embedder, graph_retriever, redis, store)
    agent = RagPipeline(retriever, settings, models, checkpointer)
    return _WorkerComponents(
        store, indexer, graph_indexer, graph_retriever, object_store, agent,
        RedisEventBroker(redis),
    )


async def _work_loop(settings: Settings, redis: Redis, c: _WorkerComponents) -> None:
    worker_id = socket.gethostname()
    while True:
        try:
            await redis.blpop("ingestion:jobs", timeout=1)
        except RedisError:
            await asyncio.sleep(1)
        job = await c.store.claim_ingestion_job(
            worker_id, lease_seconds=900, index_version=settings.index_version
        )
        if job:
            await process_job(
                job.id,
                c.store,
                c.indexer,
                c.graph_indexer,
                Path(settings.object_storage_path),
                c.object_store,
                settings.index_version,
            )
        run_item = await c.store.claim_chat_run(worker_id, lease_seconds=900)
        if run_item:
            await process_run(run_item, c.store, c.agent, c.events)
        corpus_task = await c.store.claim_corpus_rebuild(worker_id, settings.index_version)
        if corpus_task:
            await process_corpus_rebuild(corpus_task, c.store, c.indexer)


async def run() -> None:
    settings = get_settings()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    pool = await asyncpg.create_pool(settings.database_url)
    try:
        async with (
            httpx.AsyncClient(timeout=120) as client,
            AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer,
        ):
            await checkpointer.setup()
            components = await _build_components(settings, redis, pool, client, checkpointer)
            try:
                await _work_loop(settings, redis, components)
            finally:
                await components.graph_indexer.close()
                await components.graph_retriever.driver.close()
    finally:
        await redis.aclose()
        await pool.close()


if __name__ == "__main__":
    # Without this, the root logger has no handler and the default "last resort"
    # handler only surfaces WARNING+, silently dropping every INFO-level event
    # (e.g. the weaviate_search instrumentation) even though caplog-based tests
    # can't detect the gap.
    configure_logging()
    configure_tracing()
    asyncio.run(run())
