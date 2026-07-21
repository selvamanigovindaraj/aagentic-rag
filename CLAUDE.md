# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cp .env.example .env
uv sync --extra dev
uv run uvicorn app.main:app --app-dir backend --reload   # API only, host dev

uv run ruff check backend evaluation tests               # lint
uv run --extra dev pytest -q                              # all tests
uv run --extra dev pytest tests/test_agentic_loop.py -q   # single file
uv run --extra dev pytest tests/test_api.py::test_name -q # single test

uv run python evaluation/runtime_report.py
uv run python evaluation/load_test.py --count 100000 --concurrency 20

docker compose up --build   # full stack: api :8000, frontend :3000, postgres, redis, minio
```

Dev API calls require `X-Tenant-ID` and `X-User-ID` headers; `X-Groups` is a comma-separated ACL list (`DEV_AUTH=true` locally; OIDC in deployed envs).

Weaviate (Cloud) and Neo4j (Aura) are always external/hosted — never add local containers for them; configure via `WEAVIATE_URL`/`WEAVIATE_API_KEY` and `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` in `.env`. Compose Postgres is on host port `5433` (container port stays `5432`) to avoid colliding with a system Postgres.

For a shadow reindex: `uv run python scripts/enqueue_shadow_reindex.py --version v2`, deploy a worker with `INDEX_VERSION=v2`, wait for its jobs, evaluate v2, then cut over API + workers together. `scripts/backfill_index_version.py` is a one-time-use migration for pre-versioning records.

## Architecture

Two processes, one codebase: the FastAPI **api** (`backend/app/main.py`) and the **worker** (`backend/app/worker.py`), both built from `backend/app/`. PostgreSQL is the single canonical store; Weaviate and Neo4j are rebuildable indexes; Redis is wake-up/event transport only (Postgres holds durable job/run leases, so losing Redis interrupts progress events, not correctness).

Request path: `security → routing → planning → retrieval → grading → grounded generation → output verification`, implemented as a LangGraph `StateGraph` in `backend/app/services/rag_pipeline.py` (`RagPipeline`). Nodes: `classify → expand → plan → retrieve → critique → (rewrite → retrieve loop | escalate to plan | compose) → generate`. Every chat run's ID is also its LangGraph checkpoint `thread_id`, checkpointed into Postgres via `AsyncPostgresSaver`.

Package ownership (`backend/app/`):
- `components/` — datastore-backed retrieval and reranking. `hybrid_retriever.py` composes retrievers via decoration: `ActiveDocumentRetriever(CachedRetriever(RerankingRetriever(CompositeRetriever(WeaviateRetriever, Neo4jRetriever), voyage), SemanticCache), store)`. Both `main.py` (api) and `worker.py` build this same stack independently.
- `services/` — orchestration: `rag_pipeline.py` (the agent graph), `ingestion.py` (parse/chunk), `graph_index.py`/`raptor.py` (Neo4j + RAPTOR summarization), `conversation.py`, `query_router.py`, `query_rewriter.py`, `semantic_cache.py`, `events.py` (Redis pub/sub broker).
- `agents/` — pure decision logic called from graph nodes: `adaptive_router.py` (route classification), `query_decomposer.py` (plan leaves), `document_grader.py` (evidence acceptance).
- `prompts/` — versioned prompt registry (`registry.py` + `templates.py`); pipeline references prompts by `"name:v1"` key.
- `security/` — trust boundaries: `input_guard.py` (query validation), `content_filter.py` (`authorized_evidence` — tenant/ACL filtering applied before grading/retrieval scoring), `output_filter.py`.
- `tools/` — agent-callable retrieval operations.
- `repositories/` — `postgres.py` (`PostgresStore`, canonical persistence) implementing the `store.py` protocol.
- `schemas/` — `domain.py` (core models: `AgentState`, `Evidence`, `Citation`, `Route`, `AuthContext`) and `agent.py` (structured LLM I/O models like `ReasoningPlan`, `GroundedAnswer`, `CritiqueBatch`).

Top-level: `evaluation/` (golden datasets, offline/online eval, load test), `observability/` (tracing, cost tracking, feedback), `frontend/` (separately containerized React/Vite app, built by its own Dockerfile), `scripts/` (migrations, index version tooling), `data/` (raw/processed/index_config).

## Invariants (see also `AGENTS.md`)

- Every retrieval filter includes tenant and ACL constraints before scoring or graph traversal; never traverse Neo4j entities before restricting contributing `Statement` nodes by tenant/ACL.
- Synthetic expansion text can be used to *retrieve* evidence but can never itself become evidence returned to the user.
- Answers without accepted evidence refuse rather than improvise (`_generate` in `rag_pipeline.py` returns the refusal string when `accepted_evidence` is empty).
- A malformed structured LLM response falls back inside its graph node (see the `try/except (ValueError, json.JSONDecodeError)` pattern throughout `rag_pipeline.py`) — never retry the whole durable run.
- DeepSeek Flash (`deepseek-v4-flash`) is the default model for routing and ordinary answers; Pro (`deepseek-v4-pro`) is reserved for temporal/causal routing and multi-hop final synthesis. Always go through LangChain provider integrations (`langchain-deepseek`, `langchain-voyageai`) — never hand-write provider HTTP clients.
- PDF parsing is PyMuPDF, native-text only (no OCR): an image-only PDF must fail with `OCR_REQUIRED` rather than index empty; DOCX/PPTX use native OOXML text. Layout manifests (`<object_key>.layout.json`) live beside originals in object storage and must be deleted together with the source.
- Weaviate schema changes must be rolling-upgrade safe: retrieval tolerates a missing optional property until ingestion's schema reconciliation adds it. The Weaviate Cloud plan has one collection (`FilingSection` by default); every object/query is isolated by `tenantId` and `nodeType`, and `__public__` is the marker used for public ACLs (translate to an empty ACL set before authorization checks).
- RAPTOR (`services/raptor.py`) uses soft-membership GMM clustering with BIC selection; clusters above the target size need ≥2 components so a recursive parent level is always produced. Summaries are navigation-only, bounded to 4,000 characters — overlong valid model output must not fail the durable corpus rebuild.
- JSONB values must be serialized before passing to asyncpg and decoded at repository boundaries (its default codec accepts strings, not dicts).
- Frontend Nginx must resolve the `api` upstream through Docker DNS at request time, not startup, or it goes stale after the API container is recreated.
- Neither `main.py` nor `worker.py` gets logging configuration for free: uvicorn only configures its own `uvicorn.*` loggers, so `app.*` module loggers (`logging.getLogger(__name__)`) fall through to Python's "last resort" handler, which silently drops everything below WARNING. Both entrypoints call `core.logging.configure_logging()` at startup — use it (not a bare `logging.basicConfig`) for any new entrypoint, since its `ExtraFieldsFormatter` is also what makes `extra={...}` fields (tenant_id, reason, duration_ms, ...) actually show up in the log line instead of being silently discarded by the default formatter. `caplog`-based tests can't catch either gap — only live container logs can.
- `RagPipeline._classify` always invokes the LLM router-expansion override now (not just when the keyword router already picked a non-direct route) — the keyword router's fallback for anything it doesn't recognize is `direct`, which is exactly the case that most needed the LLM's second look; live eval went from 16.7% to ~82% route accuracy on realistic phrasing after this change. Trade-off: `direct_p95_ms` moved from 2000 to 3500 in `evaluation/thresholds.json` to reflect the now-mandatory extra sequential LLM call on the direct path (measured live, clean steady-state p95 ≈ 3.2–3.7s). If direct-path latency ever needs to drop back under ~2s, that requires actually removing the always-on call for high-confidence keyword matches, not just tuning the prompt.
