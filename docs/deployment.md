# Deployment

## Local (Docker Compose)

1. `cp .env.example .env` and fill in `DEEPSEEK_API_KEY`, `VOYAGE_API_KEY`, `WEAVIATE_URL` +
   `WEAVIATE_API_KEY` (Weaviate Cloud), `NEO4J_URI` + `NEO4J_USER` + `NEO4J_PASSWORD` (Neo4j
   Aura — the URI must use `neo4j+s://`/`neo4j+ssc://`, enforced by `Settings`). LangSmith
   tracing is optional: set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to enable it.
2. `docker compose up --build` — this starts `api` (:8000), `worker`, `frontend` (:3000),
   `postgres` (host `5433` → container `5432`), `redis`, and `minio` (S3-compatible object
   storage for uploaded documents; no local container for Weaviate or Neo4j — both stay
   Cloud/Aura). Compose injects container-network URLs (`postgres:5432`, `redis:6379`,
   `minio:9000`) for the `api`/`worker` services automatically via the shared `backend-env`
   anchor in `compose.yaml`; `.env`'s host-side values (`localhost:5433`, etc.) are only for
   running the API/worker directly on the host instead of in Compose.
3. `docker compose logs api worker postgres redis minio` to check startup; Neo4j and Weaviate
   only run in their managed clouds, so they never appear here.
4. Open the frontend at `http://localhost:3000` and the API docs at `http://localhost:8000/docs`.

Compose mounts `./backend/migrations` into Postgres's `docker-entrypoint-initdb.d`, so migrations
run automatically on first volume initialization. Against an existing volume (schema changes on a
running database), run `uv run python scripts/migrate.py` instead — it's idempotent.

Development auth (`DEV_AUTH=true`, the Compose default) reads identity from `X-Tenant-ID`,
`X-User-ID`, and comma-separated `X-Groups` headers; set `DEV_AUTH=false` and configure the
`oidc_*` settings for a real OIDC provider in any deployed environment.

## Shadow reindex (zero-downtime reindexing)

Because Weaviate/Neo4j are rebuildable and every object carries `index_version`, a new retrieval
or ingestion strategy ships without downtime:

1. `uv run python scripts/enqueue_shadow_reindex.py --version v2` to queue reindex jobs.
2. Deploy a worker with `INDEX_VERSION=v2` (existing `v1` workers keep serving unaffected).
3. Wait for the `v2` jobs to complete, then evaluate `v2` with the `evaluation/` scripts.
4. Cut the API and remaining workers over to `v2` together.

`scripts/backfill_index_version.py` is a one-time migration for records written before index
versioning existed — not part of the routine reindex flow.

## Kubernetes (`deploy/kubernetes.yaml`)

- `rag-api` — a 2-replica Deployment (`/healthz` readiness + liveness probes, Prometheus scrape
  annotations on `/metrics`) fronted by a ClusterIP `Service`, plus a KEDA `ScaledObject`
  (`rag-api-latency`) that scales 2–20 replicas on the exported `rag_api_request_latency_p95_seconds`
  Prometheus gauge (threshold `1.5`).
- `rag-worker` — a 1-replica Deployment (`command: [python, -m, app.worker]`) plus a KEDA
  `ScaledObject` (`rag-worker-backlog`) that scales 1–50 replicas on a live PostgreSQL query
  against the `ingestion_jobs` backlog — the correctness queue, so losing Redis cannot scale it
  to zero (Redis only carries wake-up events, not the durable job backlog).
- `rag-frontend` — a 2-replica Deployment + Service serving the built static assets on port 80.

Both `rag-api` and `rag-worker` read all configuration from a single `agentic-rag-secrets`
Secret via `envFrom` — populate it with the same variables as `.env`. Apply the manifest after
installing KEDA and Prometheus in-cluster and pointing the KEDA trigger's `serverAddress` at your
Prometheus instance; replace the `agentic-rag-backend:latest`/`agentic-rag-frontend:latest` image
names with your built/pushed images first.
