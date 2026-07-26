from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from uuid import NAMESPACE_URL, uuid5
from xml.etree import ElementTree

import httpx
import pymupdf

from ..components.llm import ModelGateway
from ..components.voyage import Embedder
from ..core.errors import AppError
from ..schemas.domain import Document


@dataclass(frozen=True, slots=True)
class TextBlock:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    heading: bool


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    blocks: tuple[TextBlock, ...]
    page_count: int
    pages_without_text: tuple[int, ...]
    artifacts: tuple[LayoutArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class LayoutArtifact:
    kind: str
    page: int
    bbox: tuple[float, float, float, float]
    text: str | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    text: str
    parent_text: str
    page: int
    section: str | None
    bbox: tuple[float, float, float, float] | None = None


def parse_document(path: str | Path) -> ParsedDocument:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return parse_pdf(path)
    if path.suffix.lower() in {".docx", ".pptx"}:
        return parse_ooxml(path)
    if path.suffix.lower() not in {".txt", ".md"}:
        raise AppError(
            422,
            "UNSUPPORTED_DOCUMENT",
            "Only PDF, DOCX, PPTX, TXT, and Markdown are supported",
        )
    return _parse_text(path)


def _parse_text(path: Path) -> ParsedDocument:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AppError(422, "INVALID_TEXT", "The text document could not be read") from exc
    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    if not paragraphs:
        raise AppError(422, "EMPTY_DOCUMENT", "The document contains no text")
    blocks = tuple(
        TextBlock(1, item, (0.0, 0.0, 0.0, 0.0), index == 0 and len(item.split()) <= 12)
        for index, item in enumerate(paragraphs)
    )
    return ParsedDocument(blocks, 1, ())


def parse_ooxml(path: str | Path) -> ParsedDocument:
    """Extract text-based Word/PowerPoint content with native ZIP/XML support."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            if path.suffix.lower() == ".docx":
                blocks, artifacts = _docx_blocks(archive)
                page_count = 1
            else:
                blocks, page_count = _pptx_document_blocks(archive)
                artifacts = []
    except (OSError, zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise AppError(
            422, "INVALID_OFFICE_DOCUMENT", "The Office document could not be read"
        ) from exc
    if not blocks:
        raise AppError(422, "EMPTY_DOCUMENT", "The Office document contains no text")
    return ParsedDocument(tuple(blocks), page_count, (), tuple(artifacts))


def _pptx_document_blocks(archive: zipfile.ZipFile) -> tuple[list[TextBlock], int]:
    slide_names = sorted(
        (name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
        key=lambda name: int(re.search(r"\d+", name).group()),
    )
    blocks = [
        block
        for page, name in enumerate(slide_names, 1)
        for block in _pptx_blocks(archive.read(name), page)
    ]
    return blocks, len(slide_names)


def _docx_paragraph(element, namespace: str) -> tuple[str, bool]:
    text = "".join(item.text or "" for item in element.iter(f"{namespace}t")).strip()
    style = element.find(f".//{namespace}pStyle")
    heading = bool(style is not None and style.get(f"{namespace}val", "").startswith("Heading"))
    return text, heading


def _docx_table_text(element, namespace: str) -> str:
    rows = [
        " | ".join(
            "".join(item.text or "" for item in cell.iter(f"{namespace}t")).strip()
            for cell in row.findall(f"{namespace}tc")
        )
        for row in element.findall(f"{namespace}tr")
    ]
    return "\n".join(row for row in rows if row.strip(" |"))


def _docx_blocks(archive: zipfile.ZipFile) -> tuple[list[TextBlock], list[LayoutArtifact]]:
    root = ElementTree.fromstring(archive.read("word/document.xml"))
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    body = root.find(f"{namespace}body")
    result: list[TextBlock] = []
    artifacts: list[LayoutArtifact] = []
    for element in body if body is not None else ():
        if element.tag == f"{namespace}p":
            text, heading = _docx_paragraph(element, namespace)
        elif element.tag == f"{namespace}tbl":
            text, heading = _docx_table_text(element, namespace), False
            if text:
                artifacts.append(LayoutArtifact("table", 1, (0.0, 0.0, 0.0, 0.0), text))
        else:
            continue
        if text:
            result.append(TextBlock(1, text, (0.0, 0.0, 0.0, 0.0), heading))
    return result, artifacts


def _pptx_blocks(content: bytes, page: int) -> list[TextBlock]:
    root = ElementTree.fromstring(content)
    namespace = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    paragraphs = []
    for paragraph in root.iter(f"{namespace}p"):
        text = "".join(item.text or "" for item in paragraph.iter(f"{namespace}t")).strip()
        if text:
            paragraphs.append(text)
    return [
        TextBlock(page, text, (0.0, 0.0, 0.0, 0.0), index == 0 and len(text) <= 200)
        for index, text in enumerate(paragraphs)
    ]


ProvenancedWord = tuple[str, int, str | None, tuple[float, float, float, float]]


def _provenanced_words(document: ParsedDocument) -> list[ProvenancedWord]:
    """Flatten blocks to (word, page, section, bbox); headings become section labels."""
    words: list[ProvenancedWord] = []
    section: str | None = None
    for block in document.blocks:
        if block.heading:
            section = block.text
        else:
            words.extend((word, block.page, section, block.bbox) for word in block.text.split())
    return words


def _ends_sentence(word: str) -> bool:
    return word.rstrip("\"')]").endswith((".", "!", "?"))


def _sentence_safe_windows(
    words: list[ProvenancedWord], window_size: int
) -> list[list[ProvenancedWord]]:
    """Slice into ~window_size groups without cutting a sentence: a would-be
    mid-sentence window trims to the last sentence-ending word, carrying the tail
    into the next window (the last window is never trimmed).
    ponytail: punctuation heuristic, not an NLP tokenizer (none installed; fooled
    by "Dr." etc) -- acceptable chunk-boundary ceiling."""
    windows = []
    start, total = 0, len(words)
    while start < total:
        end = min(start + window_size, total)
        if end < total:
            boundary = end
            while boundary > start and not _ends_sentence(words[boundary - 1][0]):
                boundary -= 1
            end = boundary if boundary > start else end
        windows.append(words[start:end])
        start = end
    return windows


def chunk_document(
    document: ParsedDocument, *, child_words: int = 200, parent_words: int = 1_000
) -> list[Chunk]:
    # ponytail: hand-rolled instead of langchain-text-splitters because the
    # per-word (page, section, bbox) provenance must survive into each Chunk
    # for citations; a generic splitter only preserves text.
    chunks: list[Chunk] = []
    for parent in _sentence_safe_windows(_provenanced_words(document), parent_words):
        parent_text = " ".join(word for word, _, _, _ in parent)
        for child in _sentence_safe_windows(parent, child_words):
            text = " ".join(word for word, _, _, _ in child)
            first = child[0]
            chunks.append(Chunk(len(chunks), text, parent_text, first[1], first[2], first[3]))
    return chunks


class WeaviateIndexer:
    def __init__(
        self,
        url: str,
        api_key: str,
        embedder: Embedder,
        models: ModelGateway,
        client: httpx.AsyncClient,
        collection: str = "RagNode",
        index_version: str = "v1",
    ) -> None:
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.embedder = embedder
        self.models = models
        self.client = client
        self.collection = collection
        self.index_version = index_version

    async def index(
        self, document: Document, path: str | Path, *, chunks: list[Chunk] | None = None
    ) -> int:
        await self._ensure_schema()
        chunks = chunks if chunks is not None else chunk_document(parse_document(path))
        if not chunks:
            raise AppError(422, "EMPTY_DOCUMENT", "The document contains no searchable text")
        objects = await self._document_objects(document, chunks)
        await self.delete(document)
        await self._upload_objects(objects)
        return len(objects)

    async def _document_objects(self, document: Document, chunks: list[Chunk]) -> list[dict]:
        from .raptor import build_raptor

        summaries = await build_raptor(chunks, self.models, self.embedder)
        parents = list(dict.fromkeys(chunk.parent_text for chunk in chunks))
        texts = [chunk.text for chunk in chunks] + parents
        vectors = await self.embedder.embed(texts, "document")
        objects = [
            self._object(document, f"chunk-{chunk.index}", chunk.text, "chunk", vector, chunk)
            for chunk, vector in zip(chunks, vectors[: len(chunks)], strict=True)
        ]
        objects.extend(
            self._object(document, f"parent-{index}", text, "parent", vector)
            for index, (text, vector) in enumerate(
                zip(parents, vectors[len(chunks) :], strict=True)
            )
        )
        objects.extend(self._summary_object(document, summary) for summary in summaries)
        return objects

    def _summary_object(self, document: Document, summary) -> dict:
        return self._object(
            document,
            summary.key,
            summary.text,
            "summary",
            summary.vector,
            page=summary.page,
            section=summary.section,
            level=summary.level,
            child_ids=list(summary.child_ids),
            source_ids=list(summary.source_ids),
            is_root=summary.is_root,
        )

    async def _upload_objects(self, objects: list[dict]) -> None:
        response = await self.client.post(
            f"{self.url}/v1/batch/objects", headers=self.headers, json={"objects": objects}
        )
        response.raise_for_status()
        payload = response.json()
        failures = (
            [
                item
                for item in payload
                if item.get("result", {}).get("errors")
                or item.get("result", {}).get("status") == "FAILED"
            ]
            if isinstance(payload, list)
            else []
        )
        if failures:
            raise AppError(502, "WEAVIATE_INDEX_FAILED", "Weaviate rejected indexed objects")

    async def _batch_delete(self, operands: list[dict]) -> None:
        response = await self.client.request(
            "DELETE",
            f"{self.url}/v1/batch/objects",
            headers=self.headers,
            json={
                "match": {
                    "class": self.collection,
                    "where": {"operator": "And", "operands": operands},
                }
            },
        )
        response.raise_for_status()

    async def delete(self, document: Document) -> None:
        await self._batch_delete(
            [
                {"path": ["tenantId"], "operator": "Equal", "valueText": document.tenant_id},
                {"path": ["documentId"], "operator": "Equal", "valueText": str(document.id)},
                {"path": ["indexVersion"], "operator": "Equal", "valueText": self.index_version},
            ]
        )

    async def rebuild_corpus(self, document: Document) -> int:
        """Rebuild the upper corpus tree for one exact ACL cohort."""
        await self._ensure_schema()
        cohort = acl_cohort(document.acl_groups)
        roots = await self._all_document_roots(document, cohort)
        await self._delete_corpus(document.tenant_id, cohort)
        if len(roots) < 2:
            return 0
        from .raptor import build_summary_tree

        nodes = [
            (row["nodeId"], row["text"], None, None, row["_additional"]["vector"], ())
            for row in roots
        ]
        summaries = await build_summary_tree(
            nodes, self.models, self.embedder, cluster_size=8, key_prefix="corpus"
        )
        objects = [self._corpus_object(document, cohort, summary) for summary in summaries]
        response = await self.client.post(
            f"{self.url}/v1/batch/objects", headers=self.headers, json={"objects": objects}
        )
        response.raise_for_status()
        return len(objects)

    async def _all_document_roots(self, document: Document, cohort: str) -> list[dict]:
        roots = await self._document_roots(document.tenant_id, cohort)
        known = {row["nodeId"] for row in roots}
        roots.extend(
            row
            for row in await self._legacy_document_roots(document.tenant_id, document.acl_groups)
            if row["nodeId"] not in known
        )
        return roots

    async def _graphql_rows(self, gql: str) -> list[dict]:
        response = await self.client.post(
            f"{self.url}/v1/graphql", headers=self.headers, json={"query": gql}
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise AppError(502, "WEAVIATE_QUERY_FAILED", payload["errors"][0]["message"])
        return payload.get("data", {}).get("Get", {}).get(self.collection, [])

    async def _document_roots(self, tenant_id: str, cohort: str) -> list[dict]:
        rows: list[dict] = []
        after = ""
        while True:
            cursor = f',after:"{after}"' if after else ""
            gql = (
                """query Roots {
              Get { %s(where:{operator:And,operands:[
                {path:["tenantId"],operator:Equal,valueText:%s},
                {path:["indexVersion"],operator:Equal,valueText:%s},
                {path:["aclCohort"],operator:Equal,valueText:%s},
                {path:["nodeType"],operator:Equal,valueText:"summary"},
                {path:["summaryScope"],operator:Equal,valueText:"document"},
                {path:["isRoot"],operator:Equal,valueBoolean:true}
              ]},limit:1000%s) { nodeId text sourceKeys _additional { vector } }}
            }"""  # noqa: UP031
                % (
                    self.collection,
                    json.dumps(tenant_id),
                    json.dumps(self.index_version),
                    json.dumps(cohort),
                    cursor,
                )
            )
            page = await self._graphql_rows(gql)
            rows.extend(page)
            if len(page) < 1000:
                return rows
            after = page[-1]["nodeId"]

    async def _legacy_document_roots(self, tenant_id: str, groups: set[str]) -> list[dict]:
        """Include roots written before aclCohort existed during rolling upgrades."""
        gql = """query LegacyRoots {
          Get { %s(where:{operator:And,operands:[
            {path:["tenantId"],operator:Equal,valueText:%s},
            {path:["indexVersion"],operator:Equal,valueText:%s},
            {path:["nodeType"],operator:Equal,valueText:"summary"},
            {path:["isRoot"],operator:Equal,valueBoolean:true}
          ]},limit:10000) {
            nodeId text sourceKeys aclGroups summaryScope _additional { vector }
          }}
        }""" % (  # noqa: UP031
            self.collection,
            json.dumps(tenant_id),
            json.dumps(self.index_version),
        )
        rows = await self._graphql_rows(gql)
        expected = groups or {"__public__"}
        return [
            row
            for row in rows
            if set(row.get("aclGroups") or []) == expected
            and row.get("summaryScope") in {None, "document"}
        ]

    async def _delete_corpus(self, tenant_id: str, cohort: str) -> None:
        await self._batch_delete(
            [
                {"path": ["tenantId"], "operator": "Equal", "valueText": tenant_id},
                {"path": ["indexVersion"], "operator": "Equal", "valueText": self.index_version},
                {"path": ["aclCohort"], "operator": "Equal", "valueText": cohort},
                {"path": ["summaryScope"], "operator": "Equal", "valueText": "corpus"},
            ]
        )

    async def _ensure_schema(self) -> None:
        response = await self.client.get(
            f"{self.url}/v1/schema/{self.collection}", headers=self.headers
        )
        if response.status_code == 200:
            existing = {item["name"] for item in response.json().get("properties", [])}
            for prop in self._schema_properties():
                if prop["name"] not in existing:
                    created = await self.client.post(
                        f"{self.url}/v1/schema/{self.collection}/properties",
                        headers=self.headers,
                        json=prop,
                    )
                    created.raise_for_status()
            return
        if response.status_code != 404:
            response.raise_for_status()
        schema = {
            "class": self.collection,
            "vectorizer": "none",
            "properties": self._schema_properties(),
        }
        created = await self.client.post(f"{self.url}/v1/schema", headers=self.headers, json=schema)
        created.raise_for_status()

    @staticmethod
    def _schema_properties() -> list[dict]:
        return [
            {"name": "nodeId", "dataType": ["text"]},
            {"name": "tenantId", "dataType": ["text"]},
            {"name": "documentId", "dataType": ["text"]},
            {"name": "documentTitle", "dataType": ["text"]},
            {"name": "text", "dataType": ["text"]},
            {"name": "page", "dataType": ["int"]},
            {"name": "section", "dataType": ["text"]},
            {"name": "aclGroups", "dataType": ["text[]"]},
            {"name": "nodeType", "dataType": ["text"]},
            {"name": "version", "dataType": ["int"]},
            {"name": "summaryLevel", "dataType": ["int"]},
            {"name": "childIds", "dataType": ["text[]"]},
            {"name": "sourceKeys", "dataType": ["text[]"]},
            {"name": "isRoot", "dataType": ["boolean"]},
            {"name": "parentText", "dataType": ["text"]},
            {"name": "sourceCoordinates", "dataType": ["text"]},
            {"name": "aclCohort", "dataType": ["text"]},
            {"name": "summaryScope", "dataType": ["text"]},
            {"name": "indexVersion", "dataType": ["text"]},
        ]

    def _object(
        self,
        document: Document,
        key: str,
        text: str,
        node_type: str,
        vector: list[float],
        chunk: Chunk | None = None,
        *,
        page: int | None = None,
        section: str | None = None,
        level: int | None = None,
        child_ids: list[str] | None = None,
        source_ids: list[str] | None = None,
        is_root: bool = False,
    ) -> dict:
        identity = f"{self.index_version}:{document.id}:{document.version}"
        node_id = str(uuid5(NAMESPACE_URL, f"{identity}:{key}"))
        resolved_sources = [
            str(uuid5(NAMESPACE_URL, f"{identity}:{source}")) for source in (source_ids or [])
        ]
        resolved_children = [
            str(uuid5(NAMESPACE_URL, f"{identity}:{child}")) for child in (child_ids or [])
        ]
        return {
            "class": self.collection,
            "id": node_id,
            "vector": vector,
            "properties": {
                "nodeId": node_id,
                "tenantId": document.tenant_id,
                "documentId": str(document.id),
                "documentTitle": document.title,
                "text": text,
                "page": chunk.page if chunk else page,
                "section": chunk.section if chunk else section,
                "aclGroups": sorted(document.acl_groups) or ["__public__"],
                "nodeType": node_type,
                "version": document.version,
                "summaryLevel": level,
                "childIds": resolved_children,
                "sourceKeys": resolved_sources or ([node_id] if node_type == "chunk" else []),
                "isRoot": is_root,
                "parentText": chunk.parent_text if chunk else None,
                "sourceCoordinates": json.dumps(chunk.bbox) if chunk and chunk.bbox else None,
                "aclCohort": acl_cohort(document.acl_groups),
                "summaryScope": "document" if node_type == "summary" else "source",
                "indexVersion": self.index_version,
            },
        }

    def _corpus_object(self, document: Document, cohort: str, summary) -> dict:
        namespace = f"{self.index_version}:{document.tenant_id}:{cohort}"
        node_id = str(uuid5(NAMESPACE_URL, f"{namespace}:{summary.key}"))
        child_ids = [
            child
            if len(child) == 36 and child.count("-") == 4
            else str(uuid5(NAMESPACE_URL, f"{namespace}:{child}"))
            for child in summary.child_ids
        ]
        return {
            "class": self.collection,
            "id": node_id,
            "vector": summary.vector,
            "properties": {
                "nodeId": node_id,
                "tenantId": document.tenant_id,
                "documentId": str(uuid5(NAMESPACE_URL, namespace)),
                "documentTitle": "Corpus overview",
                "text": summary.text,
                "aclGroups": sorted(document.acl_groups) or ["__public__"],
                "nodeType": "summary",
                "version": 1,
                "summaryLevel": summary.level,
                "childIds": child_ids,
                # Corpus nodes navigate through childIds to avoid unbounded arrays.
                "sourceKeys": [],
                "isRoot": summary.is_root,
                "aclCohort": cohort,
                "summaryScope": "corpus",
                "indexVersion": self.index_version,
            },
        }


def acl_cohort(groups: set[str]) -> str:
    value = "\0".join(sorted(groups)) or "__public__"
    return hashlib.sha256(value.encode()).hexdigest()


def parse_pdf(path: str | Path) -> ParsedDocument:
    """Extract native PDF text in visual reading order; OCR is intentionally unavailable."""
    try:
        document = pymupdf.open(path)
    except (pymupdf.FileDataError, OSError) as exc:
        raise AppError(422, "INVALID_PDF", "The PDF could not be opened") from exc

    blocks: list[TextBlock] = []
    artifacts: list[LayoutArtifact] = []
    pages_without_text: list[int] = []
    page_count = document.page_count
    try:
        for page_index, page in enumerate(document):
            page_blocks, page_artifacts = _page_layout(page, page_index + 1)
            artifacts.extend(page_artifacts)
            if page_blocks:
                blocks.extend(page_blocks)
            else:
                pages_without_text.append(page_index + 1)
    finally:
        document.close()

    if not blocks:
        raise AppError(
            422,
            "OCR_REQUIRED",
            "The PDF contains no extractable text; scanned PDFs require OCR",
        )
    return ParsedDocument(tuple(blocks), page_count, tuple(pages_without_text), tuple(artifacts))


def _page_layout(
    page: pymupdf.Page, page_number: int
) -> tuple[list[TextBlock], list[LayoutArtifact]]:
    raw = page.get_text("dict", sort=True).get("blocks", [])
    return _page_text_blocks(raw, page_number), _page_artifacts(page, raw, page_number)


def _page_artifacts(
    page: pymupdf.Page, raw: list[dict], page_number: int
) -> list[LayoutArtifact]:
    artifacts = [
        LayoutArtifact("figure", page_number, tuple(float(value) for value in block["bbox"]))
        for block in raw
        if block.get("type") == 1
    ]
    try:
        artifacts.extend(
            LayoutArtifact(
                "table",
                page_number,
                tuple(float(value) for value in table.bbox),
                "\n".join(" | ".join(cell or "" for cell in row) for row in table.extract()),
            )
            for table in page.find_tables().tables
        )
    except (AttributeError, ValueError):
        pass
    return artifacts


def _body_font_size(raw: list[dict]) -> float:
    sizes = [
        span["size"]
        for block in raw
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("text", "").strip()
    ]
    return median(sizes) if sizes else 0


def _page_text_blocks(raw: list[dict], page_number: int) -> list[TextBlock]:
    body_size = _body_font_size(raw)
    result: list[TextBlock] = []
    for block in raw:
        if block.get("type") != 0:
            continue
        spans = [span for line in block.get("lines", []) for span in line.get("spans", [])]
        text = " ".join(span.get("text", "").strip() for span in spans).strip()
        if not text:
            continue
        max_size = max((span.get("size", 0) for span in spans), default=0)
        result.append(
            TextBlock(
                page=page_number,
                text=text,
                bbox=tuple(float(value) for value in block["bbox"]),
                heading=max_size >= body_size * 1.25 and len(text) <= 200,
            )
        )
    return result
