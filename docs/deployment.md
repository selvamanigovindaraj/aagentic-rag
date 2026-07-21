# Deployment

1. Configure `.env` with DeepSeek and Weaviate Cloud credentials.
2. Run `docker compose up --build`.
3. Verify `docker compose logs api worker postgres redis`; Neo4j runs only in Aura.
4. Open the frontend at `http://localhost:3000` and API docs at `http://localhost:8000/docs`.

PostgreSQL migrations run on first Compose volume initialization. For an existing volume, run `PYTHONPATH=backend python scripts/migrate.py` from the API environment.

For Kubernetes, create `agentic-rag-secrets` with the same environment variables, replace the
image names in `deploy/kubernetes.yaml`, install KEDA and Prometheus, then apply the manifest. The
API scaler reads the exported p95 latency gauge; the worker scaler queries the durable PostgreSQL
backlog, so Redis loss cannot scale the correctness queue to zero.
