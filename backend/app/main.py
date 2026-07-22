from collections import deque
from contextlib import asynccontextmanager
from math import ceil
from time import perf_counter

import asyncpg
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from redis.asyncio import Redis

from .api.routes import router
from .components.hybrid_retriever import Neo4jRetriever, build_retrieval_stack
from .components.llm import DeepSeekGateway
from .components.object_store import S3ObjectStore
from .components.voyage import VoyageGateway
from .core.config import get_settings
from .core.errors import AppError, app_error_handler
from .core.logging import configure_logging
from .repositories.postgres import PostgresStore
from .services.events import MemoryEventBroker, RedisEventBroker
from .services.rag_pipeline import RagPipeline

# Uvicorn configures its own "uvicorn.*" loggers but not the root logger, so
# app.* loggers (propagate=True) would otherwise fall through to Python's
# "last resort" handler, which only surfaces WARNING+ and drops INFO events
# like the weaviate_search retrieval instrumentation entirely.
configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "store", None):
        app.state.events = getattr(app.state, "events", MemoryEventBroker())
        yield
        return
    settings = get_settings()
    app.state.settings = settings
    async with (
        httpx.AsyncClient(timeout=5) as app.state.http,
        AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer,
    ):
        await checkpointer.setup()
        pool = await asyncpg.create_pool(settings.database_url)
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        app.state.redis = redis
        app.state.object_store = S3ObjectStore(settings)
        await app.state.object_store.setup()
        voyage = VoyageGateway(settings)
        graph = Neo4jRetriever(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            settings.index_version,
            embedder=voyage,
        )
        try:
            store = PostgresStore(pool, settings.index_version)
            retriever = build_retrieval_stack(settings, voyage, graph, redis, store)
            app.state.store = store
            app.state.events = RedisEventBroker(redis)
            app.state.agent = RagPipeline(
                retriever, settings, DeepSeekGateway(settings), checkpointer
            )
            yield
        finally:
            await graph.driver.close()
            await redis.aclose()
            await pool.close()


app = FastAPI(title="Agentic RAG", version="0.1.0", lifespan=lifespan)
app.add_exception_handler(AppError, app_error_handler)
app.include_router(router)
request_latencies: deque[float] = deque(maxlen=1000)


@app.middleware("http")
async def measure_request_latency(request: Request, call_next):
    started = perf_counter()
    try:
        return await call_next(request)
    finally:
        if request.url.path not in {"/healthz", "/metrics"}:
            request_latencies.append(perf_counter() - started)


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics() -> str:
    values = sorted(request_latencies)
    p95 = values[max(ceil(len(values) * 0.95) - 1, 0)] if values else 0
    return f"rag_api_request_latency_p95_seconds {p95}\n"
