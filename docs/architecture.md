# Architecture

The API (`app/main.py`) and ingestion worker (`app/worker.py`) are separate processes built
from one Python codebase (`backend/app/`). PostgreSQL is canonical — it holds documents, jobs,
chat runs, and job leases. Weaviate Cloud and Neo4j Aura are rebuildable indexes: both are
external/hosted (no local containers), tagged with `index_version` so a shadow reindex can build
a new version alongside the old one and cut over without downtime. Redis is wake-up/event
transport only; losing it interrupts progress streaming, never correctness, since the durable
queue and leases live in Postgres.

## Request path

Each chat run is a LangGraph `StateGraph` (`RagPipeline` in `services/rag_pipeline.py`):
`classify → expand → plan → retrieve → critique → (rewrite → retrieve loop | escalate to plan |
compose) → generate`. Two feedback loops make this a self-correcting agent rather than a
straight pipeline: a **rewrite loop** re-queries a sub-question whose evidence the critique step
rejected (up to `max_leaf_retries`), and an **escalation loop** re-plans a `direct` lookup as
`multi_hop` when its best evidence scores below `direct_confidence_threshold`. Every run's ID is
also its LangGraph checkpoint `thread_id`, checkpointed into Postgres per node via
`AsyncPostgresSaver` — a crashed worker's run resumes from its last checkpoint, not from scratch.

DeepSeek Flash handles routing, expansion, critique, and ordinary answers; Pro is reserved for
temporal/causal routing and multi-hop final synthesis (`_PRO_ROUTES` in `rag_pipeline.py`). The
keyword router's classification is always double-checked by an LLM call, since its fallback for
anything it doesn't recognize is `direct` — the case most likely to be misrouted.

## Retrieval

`components/hybrid_retriever.py`'s `build_retrieval_stack()` composes the production retriever
by decoration, shared by both the API and the worker:

```text
ActiveDocumentRetriever(          # final gate: drops stale/deleted/revoked entries
  CachedRetriever(                # ACL-scoped semantic cache (Redis)
    RerankingRetriever(           # Voyage rerank-2.5-lite narrows candidates
      CompositeRetriever(
        WeaviateRetriever,        # hybrid BM25+vector; RAPTOR summary-tree navigation
        Neo4jRetriever))))        # provenance-first knowledge graph
```

`CompositeRetriever` only fuses the graph retriever in for `multi_hop`/`temporal_causal` routes;
`direct`/`comparison` use vector search alone. For `synthesis`-routed queries, `WeaviateRetriever`
ranks every RAPTOR summary node — any level, any scope — in one flat similarity query instead of
walking `isRoot → childIds` top-down (the collapsed-tree strategy the RAPTOR paper's own ablation
found performs as well as layer-by-layer traversal); a corpus-scope hit that comes back with only
`childIds` (not `sourceKeys`, to keep that property bounded) gets a small, bounded resolution
step (up to 6 hops, for chained corpus→corpus nodes) before its sourceKeys are used to scope the
real chunk search. Summary text is navigation-only and is
never itself returned as evidence. The knowledge graph ranks candidate statements with
personalized PageRank over the statement–entity bipartite graph, seeded by lexical term overlap
or, for paraphrased queries with no lexical overlap, by embedding similarity against entity
surface strings. A final evidence-grading pass (`agents/document_grader.py`) uses a cheap lexical
overlap heuristic (`components/reranker.py`) as the fallback grader when the LLM critic doesn't
run — a separate, cheaper mechanism from Voyage's retrieval-time reranking above.

## Ingestion

PDF parsing is PyMuPDF native-text only (`services/ingestion.py`); an image-only PDF fails with
`OCR_REQUIRED` rather than indexing empty text. DOCX/PPTX use native OOXML text. Small-to-big
chunking embeds 200-word children and returns their 1,000-word parent as generation context;
chunk boundaries are sentence-safe — a window that would end mid-sentence trims to the last
sentence-ending word and carries the remainder into the next chunk rather than cutting across it.
RAPTOR (`services/raptor.py`) builds a bottom-up summary tree over each document's chunks using
soft-membership GMM clustering with BIC model selection; `services/graph_index.py` extracts
subject-predicate-object triples per parent chunk via LLM, rejecting any extraction or summary
that shares no vocabulary with its own source text (`services/groundedness.py`) as a defense
against prompt injection from untrusted document content.

## Package ownership (`backend/app/`)

- `components/` — retrieval and reranking: `hybrid_retriever.py` (the shared retriever-stack
  builder), `retrieval.py` (Weaviate/Neo4j/composite/reranking/cache/active-document retrievers),
  `voyage.py` (embedding + rerank gateway), `llm.py` (DeepSeek gateway), `object_store.py` (S3/
  MinIO), `reranker.py` (cheap lexical grading heuristic).
- `services/` — orchestration: `rag_pipeline.py` (the agent graph), `ingestion.py` (parse/chunk/
  index), `graph_index.py`/`raptor.py` (Neo4j + RAPTOR), `conversation.py` (session ownership),
  `query_router.py`, `query_rewriter.py`, `semantic_cache.py`, `events.py` (Redis pub/sub
  broker), `groundedness.py` (injection defense shared by graph and RAPTOR).
- `agents/` — pure decision logic: `adaptive_router.py`, `query_decomposer.py`,
  `document_grader.py`.
- `prompts/` — versioned prompt registry (`registry.py` + `templates.py`).
- `security/` — trust boundaries: `input_guard.py`, `content_filter.py` (`authorized_evidence`),
  `output_filter.py`, `auth.py` (OIDC/dev-header identity + rate limiting).
- `tools/` — agent-callable retrieval operations (`vector_search.py`).
- `repositories/` — `postgres.py` (`PostgresStore`, canonical persistence) implementing the
  `store.py` protocol (also has an in-memory `MemoryStore` used by hermetic API tests).
- `schemas/` — `domain.py` (core models: `AgentState`, `Evidence`, `Citation`, `Route`,
  `AuthContext`) and `agent.py` (structured LLM I/O: `ReasoningPlan`, `GroundedAnswer`, etc).

Top-level: `evaluation/` (golden datasets, offline/online eval, load test, LangSmith dataset
sync + `aevaluate()` integration), `observability/` (tracing, cost tracking, feedback),
`frontend/` (separately containerized React/Vite app), `scripts/` (migrations, index-version
tooling), `data/` (raw/processed/index_config).
