from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from neo4j import AsyncDriver, AsyncGraphDatabase
from openai import OpenAIError
from pydantic import BaseModel, Field

from ..components.llm import ModelGateway
from ..components.voyage import Embedder, cosine_similarity
from ..prompts.registry import prompt
from ..schemas.domain import Document
from .groundedness import is_grounded
from .ingestion import Chunk

logger = logging.getLogger(__name__)


class OpenStatement(BaseModel):
    subject: str = Field(min_length=1, max_length=300)
    predicate: str = Field(min_length=1, max_length=200)
    object: str = Field(min_length=1, max_length=500)
    date: str | None = Field(default=None, max_length=40)


class StatementBatch(BaseModel):
    statements: list[OpenStatement] = Field(default_factory=list, max_length=20)


@dataclass(frozen=True, slots=True)
class ProvenancedStatement:
    statement: OpenStatement
    page: int
    section: str | None
    source_text: str


class Neo4jIndexer:
    """Creates rebuildable, provenance-first graph indexes in Neo4j Aura."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        models: ModelGateway,
        embedder: Embedder,
        index_version: str = "v1",
        *,
        driver: AsyncDriver | None = None,
    ) -> None:
        self.driver = driver or AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.models = models
        self.embedder = embedder
        self.index_version = index_version

    async def close(self) -> None:
        await self.driver.close()

    async def index(self, document: Document, chunks: list[Chunk]) -> int:
        statements = await self._extract(chunks)
        await self._replace_statements(document, statements)
        await self._index_synonyms(document, statements)
        return len(statements)

    async def delete(self, document: Document) -> None:
        await self.driver.execute_query(
            """
            MATCH ()-[synonym:SYNONYM_OF {
              tenant_id: $tenant, document_id: $document_id, index_version: $index_version
            }]->()
            DELETE synonym
            """,
            tenant=document.tenant_id,
            document_id=str(document.id),
            index_version=self.index_version,
        )
        await self.driver.execute_query(
            """
            MATCH (statement:Statement {
              tenant_id: $tenant, document_id: $document_id, index_version: $index_version
            })
            DETACH DELETE statement
            """,
            tenant=document.tenant_id,
            document_id=str(document.id),
            index_version=self.index_version,
        )
        await self.driver.execute_query(
            """
            MATCH (document:Document {tenant_id: $tenant, id: $document_id})
            WHERE NOT (document)<-[:SUPPORTED_BY]-(:Statement)
            DETACH DELETE document
            """,
            tenant=document.tenant_id,
            document_id=str(document.id),
        )

    async def _extract(self, chunks: list[Chunk]) -> list[ProvenancedStatement]:
        parents: dict[str, Chunk] = {}
        for chunk in chunks:
            parents.setdefault(chunk.parent_text, chunk)
        result: list[ProvenancedStatement] = []
        for text, chunk in parents.items():
            try:
                raw = await self.models.complete(
                    [
                        {"role": "system", "content": prompt("open-triples:v1")},
                        {"role": "user", "content": f"<evidence>\n{text}\n</evidence>"},
                    ],
                    json_output=True,
                )
                batch = StatementBatch.model_validate_json(raw)
            except (ValueError, json.JSONDecodeError, OpenAIError) as exc:
                # One chunk's extraction failing (token-limit truncation, an
                # over-length statement batch) must not sacrifice the whole
                # document -- vector indexing already succeeded by this stage,
                # and other chunks' triples are still worth keeping.
                logger.warning(
                    "triple_extraction_call_failed", extra={"reason": type(exc).__name__}
                )
                continue
            for item in batch.statements:
                if is_grounded(item.subject, text) and is_grounded(item.object, text):
                    result.append(ProvenancedStatement(item, chunk.page, chunk.section, text))
                else:
                    # Untrusted document content drives this prompt; an extraction with
                    # no vocabulary overlap in its own source chunk is more likely
                    # injected than a real fact, so it never reaches the graph.
                    logger.warning(
                        "ungrounded_statement_rejected",
                        extra={"subject": item.subject, "object": item.object},
                    )
        return result

    async def _replace_statements(
        self, document: Document, statements: list[ProvenancedStatement]
    ) -> None:
        rows = []
        for item in statements:
            triple = item.statement
            identity = ":".join(
                [
                    self.index_version,
                    str(document.id),
                    str(document.version),
                    triple.subject,
                    triple.predicate,
                    triple.object,
                    str(item.page),
                ]
            )
            rows.append(
                {
                    "id": str(uuid5(NAMESPACE_URL, identity)),
                    "subject": triple.subject,
                    "subject_key": triple.subject.casefold().strip(),
                    "predicate": triple.predicate,
                    "object": triple.object,
                    "object_key": triple.object.casefold().strip(),
                    "text": f"{triple.subject} {triple.predicate} {triple.object}",
                    "date": triple.date,
                    "page": item.page,
                    "section": item.section,
                    "source_text": item.source_text,
                }
            )
        await self.driver.execute_query(
            """
            MATCH (old:Statement {
              tenant_id: $tenant, document_id: $document_id, index_version: $index_version
            })
            DETACH DELETE old
            """,
            tenant=document.tenant_id,
            document_id=str(document.id),
            index_version=self.index_version,
        )
        cypher = """
        MERGE (d:Document {tenant_id: $tenant, id: $document_id})
        SET d.title = $title, d.version = $version, d.acl_groups = $acl_groups
        WITH d
        UNWIND $rows AS row
        MERGE (subject:Entity {tenant_id: $tenant, key: row.subject_key})
        ON CREATE SET subject.name = row.subject
        MERGE (object:Entity {tenant_id: $tenant, key: row.object_key})
        ON CREATE SET object.name = row.object
        CREATE (s:Statement {
          id: row.id, tenant_id: $tenant, document_id: $document_id,
          index_version: $index_version,
          version: $version, acl_groups: $acl_groups, text: row.text,
          predicate: row.predicate, date: coalesce(row.date, ''), page: row.page,
          section: row.section, source_text: row.source_text
        })
        MERGE (s)-[:SUBJECT]->(subject)
        MERGE (s)-[:OBJECT]->(object)
        MERGE (s)-[:SUPPORTED_BY]->(d)
        """
        await self.driver.execute_query(
            cypher,
            tenant=document.tenant_id,
            document_id=str(document.id),
            title=document.title,
            version=document.version,
            acl_groups=sorted(document.acl_groups),
            rows=rows,
            index_version=self.index_version,
        )

    async def _index_synonyms(
        self, document: Document, statements: list[ProvenancedStatement]
    ) -> None:
        entities = sorted(
            {
                value
                for item in statements
                for value in (item.statement.subject, item.statement.object)
            }
        )
        if len(entities) < 2:
            return
        vectors = await self.embedder.embed(entities, "document")
        await self._persist_entity_embeddings(document.tenant_id, entities, vectors)
        pairs = []
        for left in range(len(entities)):
            for right in range(left + 1, len(entities)):
                score = cosine_similarity(vectors[left], vectors[right])
                if score >= 0.92 and entities[left].casefold() != entities[right].casefold():
                    pairs.append(
                        {
                            "left": entities[left].casefold().strip(),
                            "right": entities[right].casefold().strip(),
                            "score": score,
                        }
                    )
        if not pairs:
            return
        await self.driver.execute_query(
            """
            UNWIND $pairs AS pair
            MATCH (left:Entity {tenant_id: $tenant, key: pair.left}),
                  (right:Entity {tenant_id: $tenant, key: pair.right})
            MERGE (left)-[edge:SYNONYM_OF {
              tenant_id: $tenant, document_id: $document_id, index_version: $index_version
            }]->(right)
            SET edge.score = pair.score
            """,
            pairs=pairs,
            tenant=document.tenant_id,
            document_id=str(document.id),
            index_version=self.index_version,
        )

    async def _persist_entity_embeddings(
        self, tenant_id: str, entities: list[str], vectors: list[list[float]]
    ) -> None:
        # Enables embedding-similarity seeding for paraphrase/synonym-only queries
        # that share no substring with any Statement text (see Neo4jRetriever).
        rows = [
            {"key": entity.casefold().strip(), "embedding": vector}
            for entity, vector in zip(entities, vectors, strict=True)
        ]
        await self.driver.execute_query(
            """
            UNWIND $rows AS row
            MATCH (e:Entity {tenant_id: $tenant, key: row.key})
            SET e.embedding = row.embedding
            """,
            rows=rows,
            tenant=tenant_id,
        )
