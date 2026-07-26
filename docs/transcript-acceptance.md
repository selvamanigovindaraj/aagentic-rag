# Transcript architecture acceptance

An item is complete only when it has an implemented runtime path, a runnable test, defined
failure behavior, trace evidence, and (where it touches a managed provider) a cloud canary.

| Transcript requirement | Current status | Completion evidence |
|---|---|---|
| Layout-aware native document parsing | Implemented within configured scope | PyMuPDF preserves native PDF pages, headings, tables, text coordinates, figure boxes, and textless-page diagnostics; native OOXML parsing preserves DOCX headings/tables and PPTX slide boundaries. Layout manifests are stored beside originals. OCR and image understanding remain explicitly unavailable, and scanned PDFs fail with `OCR_REQUIRED`. |
| Metadata, hashing, deduplication, revisions | Implemented | Tenant-safe hash dedupe, immutable logical versions, shadow activation, source metadata, and revision tests pass. |
| 200-word children with 1,000-word parents | Implemented | Chunking and parent-context restoration tests pass. |
| Hybrid sparse/vector retrieval and bounded reranking | Implemented | Weaviate hybrid plus LangChain Voyage reranking and ACL tests. |
| Recursive RAPTOR summaries | Implemented | Document trees and exact-ACL corpus cohorts use PCA plus soft-membership GMM, retain original-evidence lineage, traverse top-down, collapse on weak retrieval, and rebuild after revision/deletion. A cloud canary produced one corpus root over two authorized document roots. |
| Corpus maintenance at ingestion scale | Implemented | PostgreSQL coalesces rebuilds by tenant/exact ACL/index version and claims them only after the document batch drains, preventing one full corpus rebuild per upload. A live job recorded `corpus_queued`, completed its source indexes, then cleared exactly one maintenance task. |
| HippoRAG triples, synonyms, dates, PageRank | Implemented | Aura indexes provenance-backed statements and synonyms; retrieval expands only an authorized bounded neighborhood, runs personalized PageRank, and returns original spans. |
| LLM classification and predictive/HyDE expansion | Implemented | One structured DeepSeek call classifies complex queries and produces retrieval-only predictive vocabulary; schema failure falls back locally, single-leaf routes use the expansion, and synthetic text never enters evidence. |
| Up-front reasoning tree | Implemented | Typed known/unknown entities, validated dependencies, and independently retryable leaves are covered by agent-loop tests. |
| Active critic and leaf-specific rewrite | Implemented | One bounded critic pass grades at most eight 2,000-character candidates per leaf and only failed leaves are rewritten/retried. Independent leaf retrieval and the graph/vector branches run concurrently. |
| Direct-route confidence fallback | Implemented | Weak direct retrieval escalates to the complex graph in tests. |
| Source-grouped parent context | Implemented | Evidence is deduplicated, grouped by source in first-plan-occurrence order, retains exact evidence IDs, and restores original parent text. |
| Claim-level citation verification | Implemented | Structured claims must reference existing authorized source IDs; summaries, triples, HyDE, and plans are rejected as evidence. |
| Durable ingestion and chat execution | Implemented | PostgreSQL `SKIP LOCKED` leases and checkpoints survive process/Redis loss; Redis is only wake-up and ephemeral SSE transport. |
| Ingestion stage/model audit trail | Implemented | An append-only PostgreSQL event records status, stage, progress, attempt, model calls, index version, and safe error. A live DOCX canary recorded queued, parsing, vector, graph, corpus, and complete events with model calls increasing from zero to three. |
| Redis cache and rate limits | Implemented | Exact tenant/ACL/index-version retrieval keys cache reranked candidates, PostgreSQL revalidates active documents after every hit, and a shared tenant/user request counter fails open when Redis is unavailable. |
| OIDC and managed graph boundaries | Implemented | Generic RS256 OIDC resolves tenant/groups per request; every datastore receives that authorization context, feedback writes require the owning user as well as the tenant, and configuration rejects non-encrypted/non-Aura Neo4j URI schemes. |
| Version-safe update and deletion | Implemented | New revisions publish only after both indexes succeed; canonical active-version gates and async fail-closed purge are tested. |
| Shadow index versioning | Implemented | Weaviate objects, Neo4j statements/synonyms, Redis keys, jobs, and workers are isolated by index version. A target-version enqueue script supports side-by-side rebuild/evaluation before switching API and worker configuration; 389 legacy vector objects, 226 statements, and 20 synonym edges were safely labeled v1. |
| Streaming operational progress | Implemented | SSE emits planning/searching/verifying, token, citation, terminal, retry, and error states without chain-of-thought. |
| Observability and evaluation | Partial | PostgreSQL checkpoints, redacted LangSmith traces, durable per-run route/retrieval/critic/verified-claim/citation-reference/token/model/estimated-cost metrics, weighted claim-groundedness and citation-validity reporting, a threshold file, API latency gauge, and concurrent load harness are implemented. Route accuracy scored 1.00 and a cloud retrieval-recall@20 canary covering exact, conceptual, graph, and broad RAPTOR routes scored 1.00 against their 0.90 and 0.85 gates. Live route latency and the full 100,000-document cloud run still fail or remain unexecuted. |
| Material UI workflow | Implemented | Upload progress, active states, streaming chat, source dialogs, retry, and operational stages build successfully. |
| Independent deployment and autoscaling | Implemented | Compose runs API/worker/frontend separately without local Weaviate or Neo4j. Kubernetes manifests provide probes/resources, p95-latency API scaling through Prometheus/KEDA, and durable PostgreSQL backlog scaling for version-matched ingestion workers. |

Live verification ingested the supplied transcript through MinIO, generated Weaviate Cloud and
Neo4j Aura indexes in one leased attempt, built a corpus RAPTOR root over two authorized documents,
and produced a synthesis answer with five authorized citations and passing citation verification.
The route-aware synthesis path reduced the same live canary from 35.65 seconds and five model calls
to 12.43 seconds and two model calls. An identical ACL-scoped warm-cache request completed in 5.44
seconds with the same passing citation verification. This is still not evidence of a cold or
mixed-workload broad-synthesis p95 below 10 seconds. The
latest cold live samples were direct 7.15 seconds (target 2) and synthesis 11.30 seconds (target 10).
Disabling redundant Pro thinking after evidence verification reduced one full frontend multi-hop
sample to 29.47 seconds; the next three durable worker samples were 29.57, 26.07, and 25.72 seconds
against the 30-second target, though a larger mixed-workload p95 is still required. Direct expansion removal, two-context exact lookup,
bounded critic input, concurrent leaf retrieval, and concurrent Neo4j/Weaviate retrieval reduced
prompt and wall-clock waste without weakening the gates. Direct and synthesis managed-model
latency still prevent a full latency pass. The local regression suite has 66 passing tests and Ruff reports no violations. The
100,000-document release test also remains cost-gated; a full cloud run requires an
explicit time/cost budget and agreed quality thresholds. These are release-validation gaps, not
claims of completed acceptance.
