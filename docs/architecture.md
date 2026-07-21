# Architecture

The API and ingestion worker are separate processes from one Python codebase. PostgreSQL is canonical; Redis holds ephemeral events, Weaviate Cloud holds hybrid-search nodes, and Neo4j holds evidence-backed graph statements.

The request path is `security → routing → planning → retrieval → grading → grounded generation → output verification`. DeepSeek Flash handles routing and ordinary answers; Pro is limited to temporal/causal and multi-hop final synthesis.

Package ownership:

- `app/components`: datastore-backed retrieval and reranking.
- `app/services`: application orchestration, conversations, rewriting, and cache.
- `app/agents`: intelligence decisions used by LangGraph nodes.
- `app/prompts`: versioned prompt contracts.
- `app/security`: input, evidence, and output trust boundaries.
- `app/tools`: callable retrieval operations.
- `evaluation`: offline golden sets and online quality calculations.
- `observability`: stage timing, feedback, and cost calculations.
