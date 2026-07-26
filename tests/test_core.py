import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.agents.adaptive_router import classify
from app.components.retrieval import (
    ActiveDocumentRetriever,
    CachedRetriever,
    WeaviateRetriever,
    weaviate_acl_filter,
)
from app.core.config import Settings, assert_safe_for_environment
from app.core.errors import AppError
from app.repositories.store import MemoryStore, visible
from app.schemas.domain import AuthContext, Document, Evidence, IngestionJob, JobStatus, Route
from app.security.auth import enforce_rate_limit
from app.security.content_filter import authorized_evidence
from app.services.rag_pipeline import RagPipeline


class FakeRetriever:
    async def retrieve(self, query, route, auth, limit):
        return [
            Evidence(
                id="e1",
                document_id=uuid4(),
                document_title="Policy",
                text="Remote work policy changed in 2024.",
                score=0.9,
                acl_groups={"hr"},
            )
        ]


@pytest.mark.asyncio
async def test_retrieval_cache_is_acl_scoped_and_reuses_results():
    class Cache:
        values = {}

        async def get(self, key):
            return self.values.get(key)

        async def set(self, key, value):
            self.values[key] = value

    class CountingRetriever(FakeRetriever):
        calls = 0

        async def retrieve(self, query, route, auth, limit):
            self.calls += 1
            return await super().retrieve(query, route, auth, limit)

    source = CountingRetriever()
    cached = CachedRetriever(source, Cache(), "v1")
    hr = AuthContext(subject="u", tenant_id="tenant", groups=frozenset({"hr"}))
    legal = AuthContext(subject="u", tenant_id="tenant", groups=frozenset({"legal"}))

    await cached.retrieve("policy", Route.DIRECT, hr, 10)
    await cached.retrieve("policy", Route.DIRECT, hr, 10)
    await cached.retrieve("policy", Route.DIRECT, legal, 10)

    assert source.calls == 2


@pytest.mark.asyncio
async def test_shared_rate_limit_rejects_after_configured_window_limit():
    class Pipeline:
        def incr(self, _key):
            return self

        def expire(self, _key, _seconds):
            return self

        async def execute(self):
            return [2, True]

    redis = SimpleNamespace(pipeline=lambda: Pipeline())
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=redis)))
    auth = AuthContext(subject="user", tenant_id="tenant")

    with pytest.raises(AppError) as caught:
        await enforce_rate_limit(request, auth, Settings(requests_per_minute=1))

    assert caught.value.status == 429


@pytest.mark.asyncio
async def test_oidc_claims_resolve_request_tenant_and_groups(monkeypatch):
    from app.security.auth import auth_context

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"keys": []}

    class Http:
        async def get(self, _url):
            return Response()

    monkeypatch.setattr(
        "app.security.auth.jwt.decode",
        lambda *args, **kwargs: {
            "sub": "user",
            "tenant_id": "tenant",
            "groups": ["research"],
        },
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(http=Http())))
    settings = Settings(
        dev_auth=False,
        oidc_jwks_url="https://issuer.example/jwks",
        oidc_issuer="https://issuer.example",
    )

    auth = await auth_context(request, "Bearer token", None, None, "", settings)

    assert auth == AuthContext(
        subject="user", tenant_id="tenant", groups=frozenset({"research"})
    )


@pytest.mark.parametrize(
    ("query", "route"),
    [
        ("What is Q3 revenue?", Route.DIRECT),
        ("Summarize policy trends", Route.SYNTHESIS),
        ("Compare drug A versus drug B", Route.COMPARISON),
        ("Did failure cause the resignation?", Route.TEMPORAL),
        ("Find the connection between Stanford and Alzheimer's", Route.MULTIHOP),
    ],
)
def test_classify_routes_query(query, route):
    assert classify(query) == route


def test_acl_requires_intersection_but_allows_public_documents():
    assert visible(set(), frozenset())
    assert visible({"hr"}, frozenset({"hr"}))
    assert not visible({"legal"}, frozenset({"hr"}))
    where = weaviate_acl_filter(AuthContext(subject="u", tenant_id="t", groups=frozenset({"hr"})))
    assert where["operands"][0]["valueText"] == "t"
    assert where["operands"][1]["valueText"] == ["__public__", "hr"]


def test_generated_navigation_nodes_can_never_become_evidence():
    document_id = uuid4()
    items = [
        Evidence(
            id="source",
            document_id=document_id,
            document_title="Policy",
            text="Original policy text.",
            score=0.8,
            source_kind="source",
        ),
        Evidence(
            id="summary",
            document_id=document_id,
            document_title="Policy",
            text="Model-created summary.",
            score=0.99,
            source_kind="summary",
        ),
        Evidence(
            id="triple",
            document_id=document_id,
            document_title="Policy",
            text="Model-created triple.",
            score=0.98,
            source_kind="triple",
        ),
    ]

    assert authorized_evidence(items, "tenant", frozenset()) == [items[0]]


def test_weaviate_cloud_credentials_are_attached():
    retriever = WeaviateRetriever("https://cluster.weaviate.network/", "secret")
    assert retriever.url == "https://cluster.weaviate.network"
    assert retriever.headers == {"Authorization": "Bearer secret"}


def test_dev_auth_must_be_disabled_outside_development():
    assert_safe_for_environment(Settings(app_env="development", dev_auth=True))
    assert_safe_for_environment(Settings(app_env="production", dev_auth=False, neo4j_password="x"))
    with pytest.raises(RuntimeError, match="dev_auth"):
        assert_safe_for_environment(Settings(app_env="production", dev_auth=True))


def test_neo4j_password_default_rejected_outside_development():
    with pytest.raises(RuntimeError, match="neo4j_password"):
        assert_safe_for_environment(Settings(app_env="production", dev_auth=False))


def test_public_marker_overrides_restricted_acl_groups():
    # A doc mixing a restricted group with the public marker (acl_groups=
    # {"finance", "__public__"}) is matched by the Weaviate ContainsAny filter
    # for any user, so the final authorization gate must agree -- otherwise a
    # user with no matching restricted group is silently denied evidence the
    # index says they can read.
    document_id = uuid4()
    mixed = Evidence(
        id="mixed",
        document_id=document_id,
        document_title="Policy",
        text="text",
        score=0.9,
        acl_groups={"finance", "__public__"},
    )
    restricted = Evidence(
        id="restricted",
        document_id=document_id,
        document_title="Policy",
        text="text",
        score=0.9,
        acl_groups={"finance"},
    )

    assert authorized_evidence([mixed], "tenant", frozenset({"hr"})) == [mixed]
    assert authorized_evidence([restricted], "tenant", frozenset({"hr"})) == []


def test_weaviate_chunk_query_escapes_newlines_and_quotes():
    # A raw newline or unescaped quote in the query breaks the GraphQL request
    # with "Unterminated string" -- json.dumps' escaping (used for `escaped`)
    # must survive being embedded directly in the hybrid.query GraphQL field.
    retriever = WeaviateRetriever("https://cloud", "key")
    escaped = json.dumps('line one\nline two "quoted"')

    query = retriever._chunk_query(
        escaped, None, json.dumps("tenant"), json.dumps(["group"]), 10, "", False
    )

    assert "\n" not in query.split("hybrid:")[1].split("}")[0]
    assert '\\"quoted\\"' in query


@pytest.mark.asyncio
async def test_weaviate_search_logs_per_tenant_candidate_pool(monkeypatch, caplog):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "Get": {
                        "RagNode": [
                            {
                                "nodeId": "chunk-1",
                                "documentId": str(uuid4()),
                                "documentTitle": "Policy",
                                "text": "text",
                            }
                        ]
                    }
                }
            }

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            return Response()

    monkeypatch.setattr("app.components.retrieval.httpx.AsyncClient", lambda **_: Client())

    with caplog.at_level("INFO"):
        await WeaviateRetriever("https://cloud", "key").retrieve(
            "policy",
            Route.DIRECT,
            AuthContext(subject="u", tenant_id="tenant-noisy", groups=frozenset({"hr"})),
            10,
        )

    records = [r for r in caplog.records if r.getMessage() == "weaviate_search"]
    assert records
    assert records[0].tenant_id == "tenant-noisy"
    assert records[0].node_type == "chunk"
    assert records[0].result_count == 1
    assert records[0].duration_ms >= 0


@pytest.mark.asyncio
async def test_agent_generates_only_from_retrieved_evidence():
    agent = RagPipeline(FakeRetriever(), Settings())
    result = await agent.run(
        "What changed in the remote work policy?",
        AuthContext(subject="u", tenant_id="t", groups=frozenset({"hr"})),
    )
    assert result["route"] == Route.DIRECT
    assert "Remote work policy changed in 2024. [1]" in result["answer"]
    assert len(result["citations"]) == 1


@pytest.mark.asyncio
async def test_agent_refuses_when_no_evidence():
    from app.components.retrieval import EmptyRetriever

    result = await RagPipeline(EmptyRetriever(), Settings(max_leaf_retries=1)).run(
        "What is Q3 revenue?", AuthContext(subject="u", tenant_id="t")
    )
    assert result["attempts"]["What is Q3 revenue?"] == 2
    assert "could not find enough authorized evidence" in result["answer"]


@pytest.mark.asyncio
async def test_store_enforces_tenant_and_group_visibility():
    store = MemoryStore()
    allowed = Document(tenant_id="a", title="A", acl_groups={"hr"})
    denied = Document(tenant_id="a", title="B", acl_groups={"legal"})
    other = Document(tenant_id="b", title="C")
    store.documents = {item.id: item for item in (allowed, denied, other)}
    assert await store.list_documents("a", frozenset({"hr"})) == [allowed]


@pytest.mark.asyncio
async def test_deleted_or_inactive_documents_are_removed_after_index_retrieval():
    store = MemoryStore()
    active = Document(tenant_id="tenant", title="Active", acl_groups={"hr"})
    deleted = Document(tenant_id="tenant", title="Deleted", acl_groups={"hr"})
    store.documents[active.id] = active

    class StaleIndex:
        async def retrieve(self, query, route, auth, limit):
            return [
                Evidence(
                    id=str(document.id),
                    document_id=document.id,
                    document_title=document.title,
                    text=document.title,
                    score=1,
                    acl_groups={"hr"},
                )
                for document in (active, deleted)
            ]

    evidence = await ActiveDocumentRetriever(StaleIndex(), store).retrieve(
        "policy",
        Route.DIRECT,
        AuthContext(subject="user", tenant_id="tenant", groups=frozenset({"hr"})),
        10,
    )

    assert [item.document_id for item in evidence] == [active.id]


@pytest.mark.asyncio
async def test_active_document_retriever_backfills_when_stale_filtering_hits_the_limit():
    store = MemoryStore()
    active_docs = [
        Document(tenant_id="tenant", title=f"Active-{i}", acl_groups={"hr"}) for i in range(2)
    ]
    stale_doc = Document(tenant_id="tenant", title="Stale", acl_groups={"hr"})
    for document in active_docs:
        store.documents[document.id] = document

    class LimitedIndex:
        def __init__(self):
            self.calls: list[int] = []

        async def retrieve(self, query, route, auth, limit):
            self.calls.append(limit)
            pool = [stale_doc, *active_docs]
            return [
                Evidence(
                    id=str(document.id),
                    document_id=document.id,
                    document_title=document.title,
                    text=document.title,
                    score=1,
                    acl_groups={"hr"},
                )
                for document in pool[:limit]
            ]

    index = LimitedIndex()
    evidence = await ActiveDocumentRetriever(index, store).retrieve(
        "policy",
        Route.DIRECT,
        AuthContext(subject="user", tenant_id="tenant", groups=frozenset({"hr"})),
        2,
    )

    assert index.calls == [2, 4]
    assert {item.document_id for item in evidence} == {document.id for document in active_docs}


@pytest.mark.asyncio
async def test_active_document_retriever_does_not_backfill_when_upstream_was_not_truncated():
    store = MemoryStore()
    active = Document(tenant_id="tenant", title="Active", acl_groups={"hr"})
    store.documents[active.id] = active

    class StaleIndex:
        calls = 0

        async def retrieve(self, query, route, auth, limit):
            StaleIndex.calls += 1
            return [
                Evidence(
                    id=str(active.id),
                    document_id=active.id,
                    document_title=active.title,
                    text=active.title,
                    score=1,
                    acl_groups={"hr"},
                )
            ]

    await ActiveDocumentRetriever(StaleIndex(), store).retrieve(
        "policy",
        Route.DIRECT,
        AuthContext(subject="user", tenant_id="tenant", groups=frozenset({"hr"})),
        10,
    )

    assert StaleIndex.calls == 1


@pytest.mark.asyncio
async def test_postgres_style_job_lease_recovers_abandoned_work():
    store = MemoryStore()
    document = Document(tenant_id="tenant", title="Policy")
    job = IngestionJob(tenant_id="tenant", document_id=document.id)
    await store.create_document(document, job)

    first = await store.claim_ingestion_job("worker-a", lease_seconds=60)
    assert first and first.attempts == 1 and first.status == "running"
    assert await store.claim_ingestion_job("worker-b", lease_seconds=60) is None

    first.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    await store.update_job(first)
    recovered = await store.claim_ingestion_job("worker-b", lease_seconds=60)

    assert recovered and recovered.id == job.id and recovered.attempts == 2


@pytest.mark.asyncio
async def test_shadow_worker_claims_only_its_index_version():
    store = MemoryStore()
    document = Document(tenant_id="tenant", title="Policy")
    job = IngestionJob(tenant_id="tenant", document_id=document.id, index_version="v2")
    await store.create_document(document, job)

    assert await store.claim_ingestion_job("v1-worker", lease_seconds=60) is None
    claimed = await store.claim_ingestion_job(
        "v2-worker", lease_seconds=60, index_version="v2"
    )

    assert claimed and claimed.id == job.id


@pytest.mark.asyncio
async def test_corpus_rebuilds_coalesce_until_ingestion_batch_drains():
    store = MemoryStore()
    document = Document(tenant_id="tenant", title="Policy")
    job = IngestionJob(tenant_id="tenant", document_id=document.id)
    await store.create_document(document, job)
    await store.request_corpus_rebuild("tenant", {"hr"}, "v1")
    await store.request_corpus_rebuild("tenant", {"hr"}, "v1")

    assert await store.claim_corpus_rebuild("worker", "v1") is None
    job.status = JobStatus.COMPLETE
    await store.update_job(job)
    task = await store.claim_corpus_rebuild("worker", "v1")

    assert task and task.acl_groups == {"hr"}
    assert len(store.corpus_rebuilds) == 1


@pytest.mark.asyncio
async def test_raptor_flat_search_ranks_all_levels_then_resolves_corpus_hits(monkeypatch):
    """Flat (collapsed-tree) retrieval: one unscoped similarity query ranks summary
    nodes at every level together (no isRoot walk); a corpus-scope hit that comes
    back with empty sourceKeys (see ingestion.py's _corpus_object -- corpus nodes
    store childIds only, to avoid unbounded arrays) still needs one bounded
    resolution hop down to a node with real sourceKeys."""
    from app.components.retrieval import WeaviateRetriever

    calls: list[str] = []

    class Response:
        def __init__(self, rows):
            self.rows = rows

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"Get": {"RagNode": self.rows}}}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            calls.append(json["query"])
            if len(calls) == 1:
                return Response(
                    [{"nodeId": "corpus", "childIds": ["document-root"], "sourceKeys": []}]
                )
            return Response(
                [{"nodeId": "document-root", "childIds": [], "sourceKeys": ["chunk-1"]}]
            )

    monkeypatch.setattr("app.components.retrieval.httpx.AsyncClient", lambda **_: Client())
    sources = await WeaviateRetriever("https://cloud", "key")._summary_sources(
        "query", None, '"tenant"', '["research"]', 10
    )

    assert sources == ["chunk-1"]
    assert len(calls) == 2, "expected one ranking query + one bounded resolution hop"
    assert "isRoot" not in calls[0], "primary ranking query must be unscoped (flat)"


def test_postgres_chat_run_decodes_jsonb_metrics():
    from app.repositories.postgres import PostgresStore
    from app.schemas.domain import ChatRun

    run = ChatRun(tenant_id="tenant", session_id=uuid4(), user_id="user", query="question")
    row = run.model_dump()
    row["metrics"] = '{"duration_ms": 12.5}'

    assert PostgresStore._run(row).metrics == {"duration_ms": 12.5}


@pytest.mark.asyncio
async def test_postgres_chat_run_serializes_jsonb_metrics_on_update():
    from app.repositories.postgres import PostgresStore
    from app.schemas.domain import ChatRun

    class Pool:
        async def execute(self, _query, *values):
            self.values = values

    pool = Pool()
    run = ChatRun(tenant_id="tenant", session_id=uuid4(), user_id="user", query="question")
    run.metrics = {"duration_ms": 12.5}

    await PostgresStore(pool).update_run(run)

    assert pool.values[11] == '{"duration_ms": 12.5}'


@pytest.mark.asyncio
async def test_postgres_job_update_appends_durable_stage_event():
    from app.repositories.postgres import PostgresStore

    class Pool:
        calls = []

        async def execute(self, query, *values):
            self.calls.append((query, values))

    pool = Pool()
    job = IngestionJob(tenant_id="tenant", document_id=uuid4(), index_version="v7")
    job.stage, job.progress, job.model_calls = "graph_indexing", 75, 4

    await PostgresStore(pool).update_job(job)

    assert len(pool.calls) == 2
    assert "ingestion_job_events" in pool.calls[1][0]
    assert pool.calls[1][1][6:8] == (4, "v7")
