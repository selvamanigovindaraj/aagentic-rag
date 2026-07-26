import pytest
from app.components.retrieval import Neo4jRetriever, _personalized_pagerank, _query_terms
from app.schemas.domain import AuthContext, Route


def test_personalized_pagerank_expands_only_through_returned_statement_graph():
    rows = [
        {
            "id": "seed",
            "subject_key": "raptor",
            "object_key": "recursive summaries",
            "lexical_score": 1,
        },
        {
            "id": "connected",
            "subject_key": "recursive summaries",
            "object_key": "broad synthesis",
            "lexical_score": 0,
        },
        {
            "id": "disconnected",
            "subject_key": "payroll",
            "object_key": "salary",
            "lexical_score": 0,
        },
    ]

    ranked = _personalized_pagerank(rows, _query_terms("How does RAPTOR work?"))
    scores = {row["id"]: score for row, score in ranked}

    assert scores["seed"] > scores["connected"] > scores["disconnected"]


def test_query_terms_drop_stop_words_and_punctuation():
    assert _query_terms("Why is the RAPTOR graph useful?") == ["graph", "raptor", "useful"]


@pytest.mark.asyncio
async def test_graph_navigation_returns_original_source_text_not_extracted_triple():
    class Record:
        def data(self):
            return {
                "id": "statement-1",
                "document_id": "07b2caec-4b52-4e74-a48f-c274ed4ad1ca",
                "document_title": "Research",
                "text": "Original paragraph containing Stanford and Alzheimer research.",
                "page": 3,
                "section": "Findings",
                "acl_groups": ["research"],
                "subject_key": "stanford",
                "object_key": "alzheimer research",
                "lexical_score": 2,
            }

    class Driver:
        async def execute_query(self, query, **parameters):
            assert "s.source_text AS text" in query
            assert "coalesce(s.index_version, 'v1') = $index_version" in query
            assert parameters["index_version"] == "v1"
            return [Record()], None, None

    evidence = await Neo4jRetriever(
        "neo4j+s://example", "neo4j", "secret", driver=Driver()
    ).retrieve(
        "Stanford Alzheimer",
        Route.MULTIHOP,
        AuthContext(subject="user", tenant_id="tenant", groups=frozenset({"research"})),
        5,
    )

    assert evidence[0].text.startswith("Original paragraph")
    assert evidence[0].source_kind == "source"


@pytest.mark.asyncio
async def test_paraphrase_query_falls_back_to_embedding_seeded_entities():
    class Record:
        def __init__(self, data):
            self._data = data

        def data(self):
            return self._data

    class Driver:
        def __init__(self):
            self.calls = []

        async def execute_query(self, query, **parameters):
            self.calls.append((query, parameters))
            if "e.embedding" in query:
                return (
                    [
                        Record({"key": "stanford", "embedding": [1.0, 0.0]}),
                        Record({"key": "unrelated", "embedding": [0.0, 1.0]}),
                    ],
                    None,
                    None,
                )
            if parameters["seed_keys"]:
                return (
                    [
                        Record(
                            {
                                "id": "statement-1",
                                "document_id": "07b2caec-4b52-4e74-a48f-c274ed4ad1ca",
                                "document_title": "Research",
                                "text": "Stanford leads the Alzheimer research program.",
                                "page": 3,
                                "section": "Findings",
                                "acl_groups": ["research"],
                                "subject_key": "stanford",
                                "object_key": "alzheimer research program",
                                "lexical_score": 0,
                            }
                        )
                    ],
                    None,
                    None,
                )
            return [], None, None

    class Embedder:
        async def embed(self, texts, input_type):
            assert input_type == "query"
            return [[1.0, 0.0]]

    driver = Driver()
    evidence = await Neo4jRetriever(
        "neo4j+s://example", "neo4j", "secret", driver=driver, embedder=Embedder()
    ).retrieve(
        "who is running the dementia study",
        Route.MULTIHOP,
        AuthContext(subject="user", tenant_id="tenant", groups=frozenset({"research"})),
        5,
    )

    assert evidence[0].text.startswith("Stanford leads")
    seed_key_calls = [
        parameters["seed_keys"] for _, parameters in driver.calls if "seed_keys" in parameters
    ]
    assert seed_key_calls[-1] == ["stanford", "unrelated"]


@pytest.mark.asyncio
async def test_lexical_hit_skips_embedding_fallback_entirely():
    class Record:
        def data(self):
            return {
                "id": "statement-1",
                "document_id": "07b2caec-4b52-4e74-a48f-c274ed4ad1ca",
                "document_title": "Research",
                "text": "Stanford leads the research program.",
                "page": 1,
                "section": None,
                "acl_groups": [],
                "subject_key": "stanford",
                "object_key": "research program",
                "lexical_score": 1,
            }

    class Driver:
        calls = 0

        async def execute_query(self, query, **parameters):
            Driver.calls += 1
            return [Record()], None, None

    class Embedder:
        async def embed(self, texts, input_type):
            raise AssertionError("embedding fallback must not run when lexical seeding succeeds")

    await Neo4jRetriever(
        "neo4j+s://example", "neo4j", "secret", driver=Driver(), embedder=Embedder()
    ).retrieve(
        "Stanford research",
        Route.MULTIHOP,
        AuthContext(subject="user", tenant_id="tenant", groups=frozenset()),
        5,
    )

    assert Driver.calls == 1
