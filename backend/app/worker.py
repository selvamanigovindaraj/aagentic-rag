import asyncio
import json
import logging
import socket
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from uuid import UUID

import asyncpg
import httpx
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .components.llm import DeepSeekGateway
from .components.object_store import S3ObjectStore
from .components.retrieval import (
    ActiveDocumentRetriever,
    CachedRetriever,
    CompositeRetriever,
    Neo4jRetriever,
    RerankingRetriever,
    WeaviateRetriever,
)
from .components.voyage import VoyageGateway
from .core.config import get_settings
from .core.errors import AppError
from .core.logging import configure_logging
from .repositories.postgres import PostgresStore
from .repositories.store import Store
from .schemas.domain import AuthContext, ChatRun, Citation, CorpusRebuild, Document, JobStatus
from .services.events import RedisEventBroker
from .services.graph_index import Neo4jIndexer
from .services.ingestion import WeaviateIndexer, chunk_document, parse_document
from .services.rag_pipeline import RagPipeline
from .services.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


async def process_job(
    job_id: UUID,
    store: Store,
    indexer: WeaviateIndexer,
    graph_indexer: Neo4jIndexer,
    root: Path,
    object_store: S3ObjectStore | None = None,
    index_version: str = "v1",
) -> None:
    job = await store.get_job_by_id(job_id)
    if not job or job.status == JobStatus.COMPLETE:
        return
    document = await store.get_document(job.document_id)
    if not document or not document.object_key:
        job.status, job.stage, job.error = JobStatus.FAILED, "failed", "Document source is missing"
        await store.update_job(job)
        return
    initial_model_calls = getattr(indexer.models, "calls", 0)
    job.index_version = index_version
    job.status, job.stage, job.progress = JobStatus.RUNNING, "parsing", 10
    await store.update_job(job)
    previous = await store.get_document(document.revision_of) if document.revision_of else None
    path = root / document.object_key
    try:
        if object_store and job.operation != "delete":
            path.parent.mkdir(parents=True, exist_ok=True)
            await object_store.download(document.object_key, path)
        if job.operation == "delete":
            job.stage, job.progress = "deleting_indexes", 40
            await store.update_job(job)
            await indexer.delete(document)
            await graph_indexer.delete(document)
            if object_store:
                await object_store.delete(document.object_key)
                await object_store.delete(f"{document.object_key}.layout.json")
            else:
                await asyncio.to_thread(path.unlink, missing_ok=True)
                await asyncio.to_thread(
                    path.with_suffix(path.suffix + ".layout.json").unlink, missing_ok=True
                )
            await store.finalize_document_delete(document.id)
            await store.request_corpus_rebuild(
                document.tenant_id, document.acl_groups, index_version
            )
            job.status, job.stage, job.progress = JobStatus.COMPLETE, "complete", 100
            job.worker_id, job.lease_until = None, None
            await store.update_job(job)
            return
        parsed = parse_document(path)
        manifest = json.dumps(
            {
                "page_count": parsed.page_count,
                "pages_without_text": parsed.pages_without_text,
                "blocks": [asdict(block) for block in parsed.blocks],
                "artifacts": [asdict(artifact) for artifact in parsed.artifacts],
            }
        ).encode()
        if object_store:
            await object_store.put(
                f"{document.object_key}.layout.json", manifest, "application/json"
            )
        else:
            await asyncio.to_thread(
                path.with_suffix(path.suffix + ".layout.json").write_bytes, manifest
            )
        chunks = chunk_document(parsed)
        job.stage, job.progress = "vector_indexing", 35
        await store.update_job(job)
        await indexer.index(document, path, chunks=chunks)
        job.model_calls = getattr(indexer.models, "calls", 0) - initial_model_calls
        job.stage, job.progress = "graph_indexing", 75
        await store.update_job(job)
        await graph_indexer.index(document, chunks)
        job.model_calls = getattr(indexer.models, "calls", 0) - initial_model_calls
        await store.activate_document(document)
        if previous:
            await indexer.delete(previous)
            await graph_indexer.delete(previous)
        job.stage, job.progress = "corpus_queued", 90
        await store.update_job(job)
        await store.request_corpus_rebuild(
            document.tenant_id, document.acl_groups, index_version
        )
        job.status, job.stage, job.progress = JobStatus.COMPLETE, "complete", 100
        job.worker_id, job.lease_until = None, None
    except AppError as exc:
        job.status, job.stage, job.error = JobStatus.FAILED, "failed", exc.message
        job.worker_id, job.lease_until = None, None
    except Exception:
        logger.exception("Ingestion failed", extra={"job_id": str(job_id)})
        job.status = JobStatus.QUEUED if job.attempts < 3 else JobStatus.FAILED
        job.stage = "retrying" if job.status == JobStatus.QUEUED else "failed"
        job.error = "Indexing will retry" if job.status == JobStatus.QUEUED else "Indexing failed"
        job.worker_id, job.lease_until = None, None
    job.model_calls = getattr(indexer.models, "calls", 0) - initial_model_calls
    await store.update_job(job)
    if object_store:
        await asyncio.to_thread(path.unlink, missing_ok=True)


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


async def process_run(
    run: ChatRun, store: Store, agent: RagPipeline, events: RedisEventBroker
) -> None:
    started = perf_counter()
    models = agent.models
    initial_input_tokens = getattr(models, "input_tokens", 0)
    initial_output_tokens = getattr(models, "output_tokens", 0)
    initial_cost = getattr(models, "estimated_cost_usd", 0.0)
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
        run.metrics = {
            "duration_ms": round((perf_counter() - started) * 1000, 2),
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
            "input_tokens": getattr(models, "input_tokens", 0) - initial_input_tokens,
            "output_tokens": getattr(models, "output_tokens", 0) - initial_output_tokens,
            "estimated_cost_usd": round(
                getattr(models, "estimated_cost_usd", 0.0) - initial_cost, 8
            ),
            "flash_model": agent.settings.deepseek_flash_model,
            "pro_model": agent.settings.deepseek_pro_model,
            "index_version": agent.settings.index_version,
        }
        await store.save_citations(citations)
        # Token events are operational streaming; the durable final answer remains in PostgreSQL.
        for token in (run.answer or "").split():
            await events.publish(run.id, {"type": "token", "content": token + " "})
        run.status, run.worker_id, run.lease_until = "complete", None, None
        await store.update_run(run)
        await events.publish(
            run.id,
            {
                "type": "answer",
                "content": run.answer,
                "citations": [str(item) for item in run.citation_ids],
            },
        )
        await events.publish(run.id, {"type": "complete"})
    except Exception:
        logger.exception("Research run failed", extra={"run_id": str(run.id)})
        run.status = "queued" if run.attempts < 3 else "failed"
        run.error = "Research will retry" if run.status == "queued" else "The research run failed"
        run.worker_id, run.lease_until = None, None
        await store.update_run(run)
        if run.status == "failed":
            await events.publish(run.id, {"type": "error", "message": run.error})


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
            models = DeepSeekGateway(settings)
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
            retriever = ActiveDocumentRetriever(
                CachedRetriever(
                    RerankingRetriever(
                        CompositeRetriever(
                            WeaviateRetriever(
                                settings.weaviate_url,
                                settings.weaviate_api_key,
                                embedder,
                                settings.weaviate_collection,
                                settings.index_version,
                            ),
                            graph_retriever,
                        ),
                        embedder,
                        settings.max_reranked_candidates,
                    ),
                    SemanticCache(redis),
                    settings.index_version,
                ),
                store,
            )
            agent = RagPipeline(retriever, settings, models, checkpointer)
            events = RedisEventBroker(redis)
            worker_id = socket.gethostname()
            try:
                while True:
                    try:
                        await redis.blpop("ingestion:jobs", timeout=1)
                    except RedisError:
                        await asyncio.sleep(1)
                    job = await store.claim_ingestion_job(
                        worker_id, lease_seconds=900, index_version=settings.index_version
                    )
                    if job:
                        await process_job(
                            job.id,
                            store,
                            indexer,
                            graph_indexer,
                            Path(settings.object_storage_path),
                            object_store,
                            settings.index_version,
                        )
                    run_item = await store.claim_chat_run(worker_id, lease_seconds=900)
                    if run_item:
                        await process_run(run_item, store, agent, events)
                    corpus_task = await store.claim_corpus_rebuild(
                        worker_id, settings.index_version
                    )
                    if corpus_task:
                        await process_corpus_rebuild(corpus_task, store, indexer)
            finally:
                await graph_indexer.close()
                await graph_retriever.driver.close()
    finally:
        await redis.aclose()
        await pool.close()


if __name__ == "__main__":
    # Without this, the root logger has no handler and the default "last resort"
    # handler only surfaces WARNING+, silently dropping every INFO-level event
    # (e.g. the weaviate_search instrumentation) even though caplog-based tests
    # can't detect the gap.
    configure_logging()
    asyncio.run(run())
