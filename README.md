# Agentic RAG

Production-shaped foundation for hybrid, RAPTOR, and graph retrieval over an enterprise corpus.

## Repository layers

```text
backend/app/
├── components/   # hybrid retrieval and reranking
├── services/     # pipeline, cache, conversations, rewriting
├── prompts/      # versioned prompt registry
├── agents/       # routing, decomposition, evidence grading
├── tools/        # agent-callable retrieval operations
└── security/     # input, evidence, and output guards
evaluation/       # golden data and offline/online evaluation
observability/    # tracing, feedback, and cost accounting
data/             # raw, processed, and index configuration
scripts/          # migration and dependency health checks
frontend/         # separately containerized React application
tests/            # API, retrieval, model, security, and ingestion checks
docs/             # architecture, API, and deployment guides
```

## Run locally

```bash
cp .env.example .env
uv sync --extra dev
uv run uvicorn app.main:app --app-dir backend --reload
```

Development authentication uses `X-Tenant-ID`, `X-User-ID`, and optional `X-Groups` headers. Set `DEV_AUTH=false` and configure OIDC for deployed environments.

Run the complete stack with `docker compose up --build`; the UI is at `http://localhost:3000` and the OpenAPI document at `http://localhost:8000/docs`.

Weaviate is external: set `WEAVIATE_URL` and `WEAVIATE_API_KEY` to your Weaviate Cloud cluster before starting Compose.
Set `WEAVIATE_COLLECTION` when the cluster already has a collection; the included Compose setup uses `FilingSection` to fit a one-collection cloud plan.

Neo4j uses Aura through `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`; no local Neo4j container is started.

The application uses the Compose PostgreSQL database by default. Containers connect to `postgres:5432`; host-side development connects to the mapped `localhost:5433` port.

DeepSeek uses `deepseek-v4-flash` for routing and ordinary grounded answers. Temporal/causal and multi-hop synthesis escalates to `deepseek-v4-pro`; configure the existing `DEEPSEEK_API_KEY` in `.env`.

Voyage is connected through LangChain: `voyage-4-lite` embeds queries/documents and `rerank-2.5-lite` reranks retrieved evidence. Configure `VOYAGE_API_KEY` in `.env`.

PDF ingestion uses PyMuPDF native-text extraction. DOCX and PPTX use their native OOXML text;
headings, tables, pages/slides, text coordinates, and figure boxes are saved as a layout manifest
beside the original object. Image-only/scanned PDFs fail with `OCR_REQUIRED` until an OCR provider
is configured; partially scanned files report their textless page numbers.

## Verify and evaluate

```bash
uv run ruff check backend evaluation tests
uv run --extra dev pytest -q
uv run python evaluation/runtime_report.py
uv run python evaluation/load_test.py --count 100000 --concurrency 20
```

The load test calls the real asynchronous ingestion API and therefore consumes configured model
and managed-index capacity. Review `evaluation/thresholds.json` and provider budgets before a full
100,000-document run; set `REQUESTS_PER_MINUTE` high enough for the test traffic. A scaffold or
small canary is not recorded as a passing scale test.

For a shadow reindex, run `uv run python scripts/enqueue_shadow_reindex.py --version v2`, deploy a
worker with `INDEX_VERSION=v2`, wait for its durable jobs to complete, evaluate v2, then switch API
and ordinary workers together. Version filters keep v1 and v2 objects/statements isolated in the
shared cloud indexes. Use `scripts/backfill_index_version.py` only once for pre-versioning records.

## Architecture invariants

- PostgreSQL is canonical; Weaviate and Neo4j are rebuildable indexes.
- Every retrieval filter includes tenant and ACL constraints before scoring or traversal.
- Synthetic expansion text can retrieve evidence but can never become evidence.
- Redis loss may interrupt progress events but cannot change persisted correctness.
- Answers without accepted evidence refuse rather than improvise.
