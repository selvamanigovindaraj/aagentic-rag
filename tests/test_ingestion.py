from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
import pymupdf
import pytest
from app.core.errors import AppError
from app.schemas.domain import Document
from app.services.graph_index import Neo4jIndexer
from app.services.ingestion import (
    Chunk,
    WeaviateIndexer,
    _sentence_safe_windows,
    chunk_document,
    parse_document,
    parse_pdf,
)
from app.services.raptor import _reduce_dimensions, build_raptor


class FakeModels:
    async def complete(self, messages, *, use_pro=False, json_output=False):
        system = messages[0]["content"]
        if "knowledge graph" in system:
            return (
                '{"statements":[{"subject":"RAPTOR","predicate":"builds",'
                '"object":"recursive summaries","date":null}]}'
            )
        return '{"summary":"Evidence-linked recursive summary."}'


class FakeEmbedder:
    async def embed(self, texts, input_type):
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]


def test_parse_pdf_preserves_page_coordinates_and_heading(tmp_path: Path):
    path = tmp_path / "document.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Revenue Report", fontsize=20)
    page.insert_text((72, 110), "Q3 revenue was 42 million dollars.", fontsize=11)
    document.save(path)
    document.close()

    parsed = parse_pdf(path)

    assert parsed.page_count == 1
    assert parsed.pages_without_text == ()
    assert parsed.blocks[0].page == 1
    assert parsed.blocks[0].heading
    assert parsed.blocks[0].bbox[0] > 0
    assert "Q3 revenue" in parsed.blocks[1].text


def test_image_only_pdf_requires_ocr(tmp_path: Path):
    path = tmp_path / "scan.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()

    with pytest.raises(AppError) as caught:
        parse_pdf(path)

    assert caught.value.code == "OCR_REQUIRED"


def test_plain_text_is_chunked_with_parent_context(tmp_path: Path):
    path = tmp_path / "architecture.txt"
    words = [f"word{index}" for index in range(520)]
    path.write_text("Architecture\n\n" + " ".join(words))

    parsed = parse_document(path)
    chunks = chunk_document(parsed, child_words=200, parent_words=1_000)

    assert parsed.page_count == 1
    assert len(chunks) == 3
    assert chunks[0].section == "Architecture"
    assert len(chunks[0].text.split()) == 200
    assert len(chunks[0].parent_text.split()) >= 520


def _word(text: str) -> tuple[str, int, str | None, tuple[float, float, float, float]]:
    return (text, 1, None, (0.0, 0.0, 0.0, 0.0))


def test_sentence_safe_windows_trims_to_last_sentence_and_carries_remainder_forward():
    # A period lands at index 5 (0-indexed); a window_size of 10 would otherwise
    # cut mid-sentence at index 10, inside the second sentence.
    words = [_word("word0"), _word("word1"), _word("word2"), _word("word3"),
             _word("word4"), _word("word5."), _word("word6"), _word("word7"),
             _word("word8"), _word("word9"), _word("word10"), _word("word11")]

    windows = _sentence_safe_windows(words, window_size=10)

    assert [w[0] for w in windows[0]] == [
        "word0", "word1", "word2", "word3", "word4", "word5.",
    ]
    assert [w[0] for w in windows[1]] == [
        "word6", "word7", "word8", "word9", "word10", "word11",
    ]
    assert sum(windows, []) == words


def test_last_window_is_never_trimmed_even_mid_sentence():
    words = [_word(f"word{index}") for index in range(15)]

    windows = _sentence_safe_windows(words, window_size=10)

    assert len(windows) == 2
    assert windows[1] == words[10:]


def test_window_without_any_sentence_boundary_falls_back_to_raw_cut():
    words = [_word(f"word{index}") for index in range(30)]

    windows = _sentence_safe_windows(words, window_size=10)

    assert [len(w) for w in windows] == [10, 10, 10]
    assert sum(windows, []) == words


def test_docx_preserves_headings_and_table_rows(tmp_path: Path):
    path = tmp_path / "policy.docx"
    content = """<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Policy</w:t></w:r></w:p>
        <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Region</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Owner</w:t></w:r></w:p></w:tc></w:tr>
        <w:tr><w:tc><w:p><w:r><w:t>India</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Operations</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
      </w:body>
    </w:document>"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", content)

    parsed = parse_document(path)

    assert parsed.blocks[0].heading
    assert parsed.blocks[0].text == "Policy"
    assert parsed.blocks[1].text == "Region | Owner\nIndia | Operations"
    assert parsed.artifacts[0].kind == "table"


def test_pptx_preserves_slide_boundaries(tmp_path: Path):
    path = tmp_path / "briefing.pptx"
    slide = """<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
      xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>%s</a:t></a:r></a:p>
      </p:txBody></p:sp></p:spTree></p:cSld></p:sld>"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("ppt/slides/slide2.xml", slide % "Deployment")
        archive.writestr("ppt/slides/slide1.xml", slide % "Architecture")

    parsed = parse_document(path)

    assert parsed.page_count == 2
    assert [(block.page, block.text) for block in parsed.blocks] == [
        (1, "Architecture"),
        (2, "Deployment"),
    ]


@pytest.mark.asyncio
async def test_weaviate_index_contains_acl_child_and_parent_nodes(tmp_path: Path):
    path = tmp_path / "architecture.txt"
    path.write_text("Architecture\n\nRAPTOR builds recursive summaries from evidence chunks.")
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"class": "RagNode"})
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await WeaviateIndexer(
            "https://cloud.example", "secret", FakeEmbedder(), FakeModels(), client
        ).index(Document(tenant_id="tenant", title="Architecture", acl_groups={"research"}), path)

    batch_request = next(
        request
        for request in requests
        if request.url.path == "/v1/batch/objects" and request.method == "POST"
    )
    payload = __import__("json").loads(batch_request.content)
    objects = payload["objects"]
    assert {item["properties"]["nodeType"] for item in objects} == {
        "chunk",
        "parent",
        "summary",
    }
    assert all(item["properties"]["tenantId"] == "tenant" for item in objects)
    assert all(item["properties"]["aclGroups"] == ["research"] for item in objects)
    assert all(item["properties"]["indexVersion"] == "v1" for item in objects)


@pytest.mark.asyncio
async def test_raptor_builds_recursive_evidence_linked_tree():
    class ClusteredEmbedder:
        async def embed(self, texts, input_type):
            if all(text.startswith("evidence") for text in texts):
                return [
                    [-10.0 + index * 0.01, 0.0] if index < 3 else [10.0 + index * 0.01, 0.0]
                    for index, _ in enumerate(texts)
                ]
            return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

    chunks = [Chunk(index, f"evidence {index}", "parent", 1, "Section") for index in range(6)]

    summaries = await build_raptor(chunks, FakeModels(), ClusteredEmbedder(), cluster_size=3)

    assert len(summaries) == 3
    assert summaries[-1].level == 2
    level_one_keys = {node.key for node in summaries if node.level == 1}
    assert set(summaries[-1].child_ids) == level_one_keys
    assert all(node.child_ids for node in summaries)


def test_raptor_summary_keys_are_content_stable_not_positional():
    from app.services.raptor import _cluster_key

    assert _cluster_key(("chunk-0", "chunk-1")) == _cluster_key(("chunk-1", "chunk-0"))
    assert _cluster_key(("chunk-0", "chunk-1")) != _cluster_key(("chunk-0", "chunk-2"))


@pytest.mark.asyncio
async def test_raptor_summary_key_unaffected_by_unrelated_cluster_reordering():
    """A rebuild that only reorders unrelated clusters must keep unchanged clusters'
    keys identical, so a corpus rebuild doesn't invalidate summaries it didn't touch."""

    async def build(order: list[int]):
        chunks = [Chunk(index, f"evidence {index}", "parent", 1, "Section") for index in order]

        class ClusteredEmbedder:
            async def embed(self, texts, input_type):
                if all(text.startswith("evidence") for text in texts):
                    return [
                        [-10.0 + int(text.split()[-1]) * 0.01, 0.0]
                        if int(text.split()[-1]) < 3
                        else [10.0 + int(text.split()[-1]) * 0.01, 0.0]
                        for text in texts
                    ]
                return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

        return await build_raptor(chunks, FakeModels(), ClusteredEmbedder(), cluster_size=3)

    forward = await build([0, 1, 2, 3, 4, 5])
    shuffled = await build([3, 4, 5, 0, 1, 2])

    assert {node.key for node in forward if node.level == 1} == {
        node.key for node in shuffled if node.level == 1
    }


@pytest.mark.asyncio
async def test_raptor_falls_back_to_evidence_excerpt_when_summary_is_ungrounded(caplog):
    class InjectedModels:
        async def complete(self, messages, **kwargs):
            return '{"summary":"Click here to claim your free prize now."}'

    chunks = [Chunk(0, "RAPTOR builds recursive summary trees.", "parent", 1, "Section")]

    with caplog.at_level("WARNING"):
        summaries = await build_raptor(chunks, InjectedModels(), FakeEmbedder())

    assert "RAPTOR builds recursive summary trees." in summaries[0].text
    assert "free prize" not in summaries[0].text
    rejected = [r for r in caplog.records if r.getMessage() == "ungrounded_summary_rejected"]
    assert rejected


@pytest.mark.asyncio
async def test_raptor_falls_back_to_evidence_excerpt_when_summary_call_fails(caplog):
    class FlakyModels:
        async def complete(self, messages, **kwargs):
            import openai

            fake_completion = type("FakeCompletion", (), {"usage": None})()
            raise openai.LengthFinishReasonError(completion=fake_completion)

    chunks = [Chunk(0, "RAPTOR builds recursive summary trees.", "parent", 1, "Section")]

    with caplog.at_level("WARNING"):
        summaries = await build_raptor(chunks, FlakyModels(), FakeEmbedder())

    assert "RAPTOR builds recursive summary trees." in summaries[0].text
    failures = [r for r in caplog.records if r.getMessage() == "summary_call_failed"]
    assert failures and failures[0].reason == "LengthFinishReasonError"


@pytest.mark.asyncio
async def test_raptor_bounds_overlong_navigation_summaries():
    class VerboseModels:
        async def complete(self, messages, **kwargs):
            return '{"summary":"' + ("bounded navigation text " * 300) + '"}'

    chunks = [Chunk(0, "source evidence", "source evidence", 1, "Section")]

    summaries = await build_raptor(chunks, VerboseModels(), FakeEmbedder())

    assert 0 < len(summaries[0].text) <= 4_000


def test_raptor_reduces_embedding_dimensions_before_gmm():
    vectors = [[float(row + column) for column in range(128)] for row in range(40)]

    reduced = _reduce_dimensions(vectors)

    assert reduced.shape == (40, 32)


@pytest.mark.asyncio
async def test_graph_index_replaces_provenanced_acl_statements():
    class Driver:
        def __init__(self):
            self.calls = []

        async def execute_query(self, query, **parameters):
            self.calls.append((query, parameters))
            return [], None, None

        async def close(self):
            return None

    driver = Driver()
    document = Document(tenant_id="tenant", title="Architecture", acl_groups={"research"})
    chunks = [Chunk(0, "RAPTOR builds summaries.", "RAPTOR builds summaries.", 7, "RAPTOR")]

    count = await Neo4jIndexer(
        "neo4j+s://aura.example",
        "neo4j",
        "secret",
        FakeModels(),
        FakeEmbedder(),
        driver=driver,
    ).index(document, chunks)

    assert count == 1
    create = next(
        parameters for query, parameters in driver.calls if "CREATE (s:Statement" in query
    )
    create_query = next(query for query, _ in driver.calls if "CREATE (s:Statement" in query)
    assert create["acl_groups"] == ["research"]
    assert create["rows"][0]["page"] == 7
    assert create["rows"][0]["section"] == "RAPTOR"
    assert create["rows"][0]["source_text"] == "RAPTOR builds summaries."
    assert create["index_version"] == "v1"
    assert "index_version: $index_version" in create_query
    assert "MERGE (subject:Entity {tenant_id: $tenant, key: row.subject_key})" in create_query
    assert "MERGE (object:Entity {tenant_id: $tenant, key: row.object_key})" in create_query


@pytest.mark.asyncio
async def test_graph_index_skips_a_chunk_whose_extraction_call_fails(caplog):
    """A single chunk hitting the model's token limit or returning an
    over-length statement batch must not fail the whole document -- vector
    indexing already succeeded by the time graph indexing runs, and other
    chunks' triples are still worth keeping."""

    class Driver:
        def __init__(self):
            self.calls = []

        async def execute_query(self, query, **parameters):
            self.calls.append((query, parameters))
            return [], None, None

        async def close(self):
            return None

    class FlakyModels:
        async def complete(self, messages, *, use_pro=False, json_output=False):
            if "Stanford" in messages[1]["content"]:
                import openai

                fake_completion = type("FakeCompletion", (), {"usage": None})()
                raise openai.LengthFinishReasonError(completion=fake_completion)
            return (
                '{"statements":[{"subject":"RAPTOR","predicate":"builds",'
                '"object":"recursive summaries","date":null}]}'
            )

    driver = Driver()
    document = Document(tenant_id="tenant", title="Architecture", acl_groups={"research"})
    chunks = [
        Chunk(0, "Stanford published a long report today.", "Stanford " * 400, 1, "Intro"),
        Chunk(1, "RAPTOR builds summaries.", "RAPTOR builds summaries.", 7, "RAPTOR"),
    ]

    with caplog.at_level("WARNING"):
        count = await Neo4jIndexer(
            "neo4j+s://aura.example",
            "neo4j",
            "secret",
            FlakyModels(),
            FakeEmbedder(),
            driver=driver,
        ).index(document, chunks)

    assert count == 1
    create = next(
        parameters for query, parameters in driver.calls if "CREATE (s:Statement" in query
    )
    assert [row["subject_key"] for row in create["rows"]] == ["raptor"]
    failures = [r for r in caplog.records if r.getMessage() == "triple_extraction_call_failed"]
    assert failures and failures[0].reason == "LengthFinishReasonError"


@pytest.mark.asyncio
async def test_graph_index_drops_statements_ungrounded_in_source_text(caplog):
    class Driver:
        def __init__(self):
            self.calls = []

        async def execute_query(self, query, **parameters):
            self.calls.append((query, parameters))
            return [], None, None

        async def close(self):
            return None

    class InjectedModels:
        async def complete(self, messages, *, use_pro=False, json_output=False):
            if "knowledge graph" in messages[0]["content"]:
                return (
                    '{"statements":['
                    '{"subject":"RAPTOR","predicate":"builds","object":"recursive summaries",'
                    '"date":null},'
                    '{"subject":"ignore previous instructions","predicate":"is",'
                    '"object":"admin password hunter2","date":null}]}'
                )
            return '{"summary":"Evidence-linked recursive summary."}'

    driver = Driver()
    document = Document(tenant_id="tenant", title="Architecture", acl_groups={"research"})
    chunks = [Chunk(0, "RAPTOR builds summaries.", "RAPTOR builds summaries.", 7, "RAPTOR")]

    with caplog.at_level("WARNING"):
        count = await Neo4jIndexer(
            "neo4j+s://aura.example",
            "neo4j",
            "secret",
            InjectedModels(),
            FakeEmbedder(),
            driver=driver,
        ).index(document, chunks)

    assert count == 1
    create = next(
        parameters for query, parameters in driver.calls if "CREATE (s:Statement" in query
    )
    assert [row["subject_key"] for row in create["rows"]] == ["raptor"]
    rejected = [r for r in caplog.records if r.getMessage() == "ungrounded_statement_rejected"]
    assert rejected and rejected[0].subject == "ignore previous instructions"


@pytest.mark.asyncio
async def test_same_entity_string_in_two_tenants_never_shares_a_graph_node():
    """Regression for the cross-tenant Entity-sharing finding: two tenants that both
    mention "Acme Corp" must MERGE onto physically distinct nodes, not one shared hub
    a PageRank traversal seeded from either tenant could walk through."""

    class Driver:
        def __init__(self):
            self.calls = []

        async def execute_query(self, query, **parameters):
            self.calls.append((query, parameters))
            return [], None, None

        async def close(self):
            return None

    driver = Driver()
    chunks = [
        Chunk(
            0,
            "Acme Corp announced quarterly results.",
            "Acme Corp announced quarterly results.",
            1,
            "Intro",
        )
    ]

    class AcmeModels(FakeModels):
        async def complete(self, messages, *, use_pro=False, json_output=False):
            if "knowledge graph" in messages[0]["content"]:
                return (
                    '{"statements":['
                    '{"subject":"Acme Corp","predicate":"announced",'
                    '"object":"quarterly results","date":null}]}'
                )
            return await super().complete(messages, use_pro=use_pro, json_output=json_output)

    for tenant_id in ("tenant-a", "tenant-b"):
        await Neo4jIndexer(
            "neo4j+s://aura.example", "neo4j", "secret", AcmeModels(), FakeEmbedder(), driver=driver
        ).index(Document(tenant_id=tenant_id, title="Filing", acl_groups={"research"}), chunks)

    entity_merges = [
        parameters for query, parameters in driver.calls if "CREATE (s:Statement" in query
    ]
    assert {call["tenant"] for call in entity_merges} == {"tenant-a", "tenant-b"}
    # Same MERGE cypher text, but the MERGE key includes $tenant, so the two calls
    # above create/attach to distinct Entity nodes despite an identical subject_key.
    assert all(call["rows"][0]["subject_key"] == "acme corp" for call in entity_merges)


@pytest.mark.asyncio
async def test_graph_index_scopes_synonym_entities_to_tenant():
    class Driver:
        def __init__(self):
            self.calls = []

        async def execute_query(self, query, **parameters):
            self.calls.append((query, parameters))
            return [], None, None

        async def close(self):
            return None

    driver = Driver()
    document = Document(tenant_id="tenant-a", title="Filing", acl_groups={"research"})
    chunks = [
        Chunk(
            0,
            "Stanford University partnered with Acme Corp on the research initiative.",
            "Stanford University partnered with Acme Corp on the research initiative.",
            1,
            "Intro",
        )
    ]

    class SynonymModels(FakeModels):
        async def complete(self, messages, *, use_pro=False, json_output=False):
            if "knowledge graph" in messages[0]["content"]:
                return (
                    '{"statements":['
                    '{"subject":"Stanford University","predicate":"partnered_with",'
                    '"object":"Acme Corp","date":null}]}'
                )
            return await super().complete(messages, use_pro=use_pro, json_output=json_output)

    await Neo4jIndexer(
        "neo4j+s://aura.example",
        "neo4j",
        "secret",
        SynonymModels(),
        FakeEmbedder(),
        driver=driver,
    ).index(document, chunks)

    synonym_query, synonym_params = next(
        (query, parameters) for query, parameters in driver.calls if "SYNONYM_OF" in query
    )
    assert "MATCH (left:Entity {tenant_id: $tenant, key: pair.left})" in synonym_query
    assert "(right:Entity {tenant_id: $tenant, key: pair.right})" in synonym_query
    assert synonym_params["tenant"] == "tenant-a"

    embedding_query, embedding_params = next(
        (query, parameters) for query, parameters in driver.calls if "e.embedding" in query
    )
    assert "MATCH (e:Entity {tenant_id: $tenant, key: row.key})" in embedding_query
    assert embedding_params["tenant"] == "tenant-a"
    persisted_keys = {row["key"] for row in embedding_params["rows"]}
    assert persisted_keys == {"stanford university", "acme corp"}


@pytest.mark.asyncio
async def test_corpus_raptor_rebuilds_an_exact_acl_cohort():
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"properties": WeaviateIndexer._schema_properties()},
            )
        if request.url.path == "/v1/graphql":
            exact_roots = [
                {
                    "nodeId": "11111111-1111-1111-1111-111111111111",
                    "text": "Finance evidence",
                    "sourceKeys": ["chunk-a"],
                    "aclGroups": ["finance"],
                    "summaryScope": "document",
                    "_additional": {"vector": [1.0, 0.0]},
                },
                {
                    "nodeId": "22222222-2222-2222-2222-222222222222",
                    "text": "Budget evidence",
                    "sourceKeys": ["chunk-b"],
                    "aclGroups": ["finance"],
                    "summaryScope": "document",
                    "_additional": {"vector": [0.9, 0.1]},
                },
            ]
            legacy_root = {
                "nodeId": "33333333-3333-3333-3333-333333333333",
                "text": "Legacy evidence",
                "sourceKeys": ["chunk-c"],
                "aclGroups": ["finance"],
                "summaryScope": None,
                "_additional": {"vector": [0.8, 0.2]},
            }
            query = __import__("json").loads(request.content)["query"]
            return httpx.Response(
                200,
                json={
                    "data": {
                        "Get": {
                            "RagNode": exact_roots
                            if "aclCohort" in query
                            else [*exact_roots, legacy_root]
                        }
                    }
                },
            )
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        count = await WeaviateIndexer(
            "https://cloud.example", "secret", FakeEmbedder(), FakeModels(), client
        ).rebuild_corpus(Document(tenant_id="tenant", title="Finance", acl_groups={"finance"}))

    batch = next(
        request
        for request in requests
        if request.url.path == "/v1/batch/objects" and request.method == "POST"
    )
    objects = __import__("json").loads(batch.content)["objects"]
    assert count == 1
    assert objects[0]["properties"]["summaryScope"] == "corpus"
    assert objects[0]["properties"]["aclGroups"] == ["finance"]
    assert set(objects[0]["properties"]["childIds"]) == {
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333",
    }
