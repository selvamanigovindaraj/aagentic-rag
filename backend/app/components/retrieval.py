from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from time import perf_counter
from typing import Protocol

import httpx
from neo4j import AsyncDriver, AsyncGraphDatabase

from ..repositories.store import Store
from ..schemas.domain import AuthContext, Evidence, Route
from ..services.semantic_cache import SemanticCache
from .voyage import Embedder, Reranker, cosine_similarity

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    async def retrieve(
        self, query: str, route: Route, auth: AuthContext, limit: int
    ) -> list[Evidence]: ...


class EmptyRetriever:
    async def retrieve(
        self, query: str, route: Route, auth: AuthContext, limit: int
    ) -> list[Evidence]:
        return []


class CachedRetriever:
    def __init__(self, retriever: Retriever, cache: SemanticCache, index_version: str) -> None:
        self.retriever = retriever
        self.cache = cache
        self.index_version = index_version

    def _cache_key(self, query: str, route: Route, auth: AuthContext, limit: int) -> str:
        identity = json.dumps(
            [self.index_version, auth.tenant_id, sorted(auth.groups), route, query, limit],
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode()).hexdigest()

    async def retrieve(
        self, query: str, route: Route, auth: AuthContext, limit: int
    ) -> list[Evidence]:
        key = self._cache_key(query, route, auth, limit)
        cached = await self.cache.get(key)
        if cached:
            try:
                return [Evidence.model_validate(item) for item in cached["evidence"]]
            except (KeyError, ValueError):
                pass
        evidence = await self.retriever.retrieve(query, route, auth, limit)
        await self.cache.set(key, {"evidence": [item.model_dump(mode="json") for item in evidence]})
        return evidence


class ActiveDocumentRetriever:
    """Final canonical-store gate for stale, deleted, or revoked index entries."""

    def __init__(self, retriever: Retriever, store: Store, *, backfill_multiplier: int = 2) -> None:
        self.retriever = retriever
        self.store = store
        self.backfill_multiplier = backfill_multiplier

    async def retrieve(
        self, query: str, route: Route, auth: AuthContext, limit: int
    ) -> list[Evidence]:
        evidence = await self.retriever.retrieve(query, route, auth, limit)
        filtered = await self._filter_active(evidence, auth)
        if len(filtered) < limit and len(evidence) >= limit:
            # The upstream fetch was truncated at `limit`, so stale/deleted/revoked
            # hits dropped here may have pushed genuinely available evidence out of
            # the fetched window. Re-fetch once, wider, instead of silently starving
            # the leaf of evidence the corpus still has.
            wider = await self.retriever.retrieve(
                query, route, auth, limit * self.backfill_multiplier
            )
            filtered = (await self._filter_active(wider, auth))[:limit]
        return filtered

    async def _filter_active(self, evidence: list[Evidence], auth: AuthContext) -> list[Evidence]:
        active = await self.store.active_document_ids(
            auth.tenant_id, auth.groups, {item.document_id for item in evidence}
        )
        return [item for item in evidence if item.document_id in active]


def weaviate_acl_filter(auth: AuthContext) -> dict:
    tenant = {"path": ["tenantId"], "operator": "Equal", "valueText": auth.tenant_id}
    groups = {
        "path": ["aclGroups"],
        "operator": "ContainsAny",
        "valueText": ["__public__", *sorted(auth.groups)],
    }
    return {"operator": "And", "operands": [tenant, groups]}


class WeaviateRetriever:
    """Thin GraphQL adapter; Weaviate remains a rebuildable index."""

    def __init__(
        self,
        url: str,
        api_key: str,
        embedder: Embedder | None = None,
        collection: str = "RagNode",
        index_version: str = "v1",
    ) -> None:
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.embedder = embedder
        self.collection = collection
        self.index_version = index_version

    async def retrieve(
        self, query: str, route: Route, auth: AuthContext, limit: int
    ) -> list[Evidence]:
        # GraphQL string-literal escaping is JSON's: json.dumps also escapes
        # newlines/control characters that a bare backslash+quote replace
        # missed (a query with an embedded newline broke the raw GraphQL
        # request with "Unterminated string"). Already includes the
        # surrounding quotes, so the query templates below embed it bare.
        escaped = json.dumps(query)
        vector = (await self.embedder.embed([query], "query"))[0] if self.embedder else None
        groups = json.dumps(["__public__", *sorted(auth.groups)])
        tenant = json.dumps(auth.tenant_id)
        if route == Route.SYNTHESIS:
            source_keys = await self._summary_sources(escaped, vector, tenant, groups, limit)
            # Summary text is navigation-only.  If tree navigation is weak, a
            # normal collapsed chunk search is the safe fallback.
            if source_keys:
                source_filter = (
                    ',{path:["sourceKeys"],operator:ContainsAny,valueText:'
                    + json.dumps(source_keys)
                    + "}"
                )
                return await self._search_chunks(
                    escaped, vector, tenant, groups, limit, source_filter
                )
        return await self._search_chunks(escaped, vector, tenant, groups, limit)

    async def _post_graphql(self, gql: str) -> dict:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{self.url}/v1/graphql", headers=self.headers, json={"query": gql}
            )
            response.raise_for_status()
        return response.json()

    def _where(self, tenant: str, groups: str, node_type: str, extra_filter: str = "") -> str:
        return (
            "{operator:And,operands:["
            f'{{path:["tenantId"],operator:Equal,valueText:{tenant}}},'
            f'{{path:["indexVersion"],operator:Equal,valueText:{json.dumps(self.index_version)}}},'
            f'{{path:["aclGroups"],operator:ContainsAny,valueText:{groups}}},'
            f'{{path:["nodeType"],operator:Equal,valueText:"{node_type}"}}{extra_filter}'
            "]}"
        )

    def _chunk_query(
        self,
        escaped: str,
        vector: list[float] | None,
        tenant: str,
        groups: str,
        limit: int,
        extra_filter: str,
        include_parent: bool,
    ) -> str:
        return (
            """query Search {
          Get { %s(
            where: %s,
            limit: %d,
            hybrid: {query: %s, vector: %s, alpha: 0.5}
          ) {
            nodeId documentId documentTitle text %spage section aclGroups nodeType
            _additional { score }
          }}
        }"""  # noqa: UP031
            % (
                self.collection,
                self._where(tenant, groups, "chunk", extra_filter),
                limit,
                escaped,
                json.dumps(vector),
                "parentText " if include_parent else "",
            )
        )

    def _log_search(self, tenant: str, rows: list[dict], started: float) -> None:
        # Single shared collection across tenants (CLAUDE.md constraint): logging
        # per-query result count and latency here is what lets an operator later
        # notice a noisy tenant diluting another tenant's candidate pool.
        logger.info(
            "weaviate_search",
            extra={
                "tenant_id": tenant.strip('"'),
                "node_type": "chunk",
                "result_count": len(rows),
                "duration_ms": (perf_counter() - started) * 1000,
            },
        )

    async def _search_chunks(
        self,
        escaped: str,
        vector: list[float] | None,
        tenant: str,
        groups: str,
        limit: int,
        extra_filter: str = "",
        include_parent: bool = True,
    ) -> list[Evidence]:
        started = perf_counter()
        payload = await self._post_graphql(
            self._chunk_query(escaped, vector, tenant, groups, limit, extra_filter, include_parent)
        )
        if payload.get("errors") and include_parent and "parentText" in str(payload["errors"]):
            # Rolling-upgrade safety: retry without the optional property until
            # ingestion's schema reconciliation adds it.
            return await self._search_chunks(
                escaped, vector, tenant, groups, limit, extra_filter, include_parent=False
            )
        if payload.get("errors"):
            raise RuntimeError(payload["errors"][0]["message"])
        rows = payload.get("data", {}).get("Get", {}).get(self.collection, [])
        self._log_search(tenant, rows, started)
        return [_chunk_evidence(row) for row in rows]

    async def _summary_sources(
        self,
        escaped: str,
        vector: list[float] | None,
        tenant: str,
        groups: str,
        limit: int,
    ) -> list[str]:
        """Flat (collapsed-tree) RAPTOR retrieval: rank every summary node -- any
        level, any scope -- by similarity in ONE query, instead of walking
        isRoot -> childIds top-down (matches the RAPTOR paper's own finding that
        collapsed-tree retrieval performs as well as layer-by-layer traversal).
        Corpus-scope nodes store childIds only, not sourceKeys (see ingestion.py's
        _corpus_object, "avoid unbounded arrays"), so any corpus-scope hits among
        the ranked results still need one bounded resolution hop; document-scope
        hits already carry sourceKeys directly and skip it entirely."""
        rows = await self._summary_rows(escaped, vector, tenant, groups, min(limit, 20))
        return await self._resolve_source_keys(rows, escaped, vector, tenant, groups)

    async def _resolve_source_keys(
        self, rows: list[dict], escaped: str, vector: list[float] | None, tenant: str, groups: str
    ) -> list[str]:
        resolved: dict[str, None] = {}
        # ponytail: bounded hop limit for childIds-only corpus chains -- raise this
        # or give corpus nodes real sourceKeys directly if 6 hops isn't enough.
        for _ in range(6):
            for key in _unique_keys(rows, "sourceKeys"):
                resolved.setdefault(key)
            children = _unique_keys([row for row in rows if not row.get("sourceKeys")], "childIds")
            if not children:
                break
            rows = await self._summary_rows(
                escaped, vector, tenant, groups, min(len(children), 20), _children_filter(children)
            )
        return list(resolved)

    async def _summary_rows(
        self,
        escaped: str,
        vector: list[float] | None,
        tenant: str,
        groups: str,
        limit: int,
        extra_filter: str = "",
    ) -> list[dict]:
        gql = """query Navigate {
          Get { %s(where:%s,limit:%d,hybrid:{query:%s,vector:%s,alpha:0.5}) {
            nodeId childIds sourceKeys _additional { score }
          }}
        }""" % (  # noqa: UP031
            self.collection,
            self._where(tenant, groups, "summary", extra_filter),
            min(limit, 20),
            escaped,
            json.dumps(vector),
        )
        payload = await self._post_graphql(gql)
        if payload.get("errors"):
            raise RuntimeError(payload["errors"][0]["message"])
        return payload.get("data", {}).get("Get", {}).get(self.collection, [])


def _chunk_evidence(row: dict) -> Evidence:
    return Evidence(
        id=row["nodeId"],
        document_id=row["documentId"],
        document_title=row["documentTitle"],
        text=row["text"],
        page=row.get("page"),
        section=row.get("section"),
        score=float(row.get("_additional", {}).get("score") or 0),
        acl_groups=set(row.get("aclGroups") or []) - {"__public__"},
        context_text=row.get("parentText") or row["text"],
    )


def _unique_keys(rows: list[dict], field: str) -> list[str]:
    return list(dict.fromkeys(key for row in rows for key in row.get(field, [])))


def _children_filter(children: list[str]) -> str:
    return ',{path:["nodeId"],operator:ContainsAny,valueText:' + json.dumps(children) + "}"


_TRAVERSAL_CYPHER = """
CALL () {
  MATCH (seed:Statement)-[:SUBJECT]->(seed_subject:Entity),
        (seed)-[:OBJECT]->(seed_object:Entity)
  WHERE seed.tenant_id = $tenant
    AND coalesce(seed.index_version, 'v1') = $index_version
    AND (seed.acl_groups IS NULL OR size(seed.acl_groups) = 0
         OR any(g IN seed.acl_groups WHERE g IN $groups))
    AND (
      (size($terms) > 0 AND any(term IN $terms WHERE toLower(seed.text) CONTAINS term
              OR seed_subject.key CONTAINS term OR seed_object.key CONTAINS term))
      OR (size($seed_keys) > 0
          AND (seed_subject.key IN $seed_keys OR seed_object.key IN $seed_keys))
    )
  WITH seed_subject, seed_object LIMIT 25
  RETURN collect(DISTINCT seed_subject) + collect(DISTINCT seed_object) AS seeds
}
MATCH (s:Statement)-[:SUPPORTED_BY]->(d:Document),
      (s)-[:SUBJECT]->(subject:Entity), (s)-[:OBJECT]->(object:Entity)
WHERE s.tenant_id = $tenant
  AND coalesce(s.index_version, 'v1') = $index_version
  AND (s.acl_groups IS NULL OR size(s.acl_groups) = 0
       OR any(g IN s.acl_groups WHERE g IN $groups))
  AND (
    subject IN seeds OR object IN seeds
    OR any(seed IN seeds WHERE EXISTS {
      MATCH (seed)-[synonym:SYNONYM_OF]-(alias:Entity)
      WHERE alias IN [subject, object] AND synonym.tenant_id = $tenant
        AND coalesce(synonym.index_version, 'v1') = $index_version
        AND EXISTS {
          MATCH (authorized:Statement {tenant_id: $tenant,
                                        document_id: synonym.document_id})
          WHERE (authorized.acl_groups IS NULL OR size(authorized.acl_groups) = 0
            OR any(g IN authorized.acl_groups WHERE g IN $groups))
            AND coalesce(authorized.index_version, 'v1') = $index_version
        }
    })
  )
RETURN s.id AS id, d.id AS document_id, d.title AS document_title,
       s.source_text AS text, s.page AS page, null AS section,
       s.acl_groups AS acl_groups,
       subject.key AS subject_key, object.key AS object_key,
       size([term IN $terms WHERE toLower(s.text) CONTAINS term]) AS lexical_score
ORDER BY lexical_score DESC, s.date DESC
LIMIT $candidate_limit
"""

_ENTITY_CANDIDATES_CYPHER = """
MATCH (e:Entity {tenant_id: $tenant})
WHERE e.embedding IS NOT NULL
  AND EXISTS {
    MATCH (e)<-[:SUBJECT|OBJECT]-(s:Statement)
    WHERE s.tenant_id = $tenant
      AND coalesce(s.index_version, 'v1') = $index_version
      AND (s.acl_groups IS NULL OR size(s.acl_groups) = 0
           OR any(g IN s.acl_groups WHERE g IN $groups))
  }
RETURN e.key AS key, e.embedding AS embedding
LIMIT 500
"""


class Neo4jRetriever:
    """ACL filtering happens on Statement nodes before graph expansion."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        index_version: str = "v1",
        *,
        driver: AsyncDriver | None = None,
        embedder: Embedder | None = None,
        embedding_seed_count: int = 10,
    ) -> None:
        self.driver = driver or AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.index_version = index_version
        self.embedder = embedder
        self.embedding_seed_count = embedding_seed_count

    async def retrieve(
        self, query: str, route: Route, auth: AuthContext, limit: int
    ) -> list[Evidence]:
        terms = _query_terms(query)
        rows = await self._traverse(terms, [], auth, limit) if terms else []
        seed_keys: list[str] = []
        if not rows and self.embedder:
            # A paraphrase/synonym-only query shares no substring with any statement,
            # so lexical seeding finds nothing and the SYNONYM_OF hop-expansion this
            # graph exists for never triggers. Fall back to embedding-similarity
            # seeds against Entity surface strings instead of dropping the query.
            seed_keys = await self._embedding_seed_keys(query, auth)
            if seed_keys:
                rows = await self._traverse([], seed_keys, auth, limit)
        if not rows:
            return []
        ranked = _personalized_pagerank(rows, terms, tuple(seed_keys))
        return [_statement_evidence(row, score) for row, score in ranked[:limit]]

    async def _traverse(
        self, terms: list[str], seed_keys: list[str], auth: AuthContext, limit: int
    ) -> list[dict]:
        records, _, _ = await self.driver.execute_query(
            _TRAVERSAL_CYPHER,
            tenant=auth.tenant_id,
            groups=list(auth.groups),
            terms=terms,
            seed_keys=seed_keys,
            candidate_limit=max(200, limit * 20),
            index_version=self.index_version,
        )
        return [record.data() for record in records]

    async def _embedding_seed_keys(self, query: str, auth: AuthContext) -> list[str]:
        records, _, _ = await self.driver.execute_query(
            _ENTITY_CANDIDATES_CYPHER,
            tenant=auth.tenant_id,
            groups=list(auth.groups),
            index_version=self.index_version,
        )
        candidates = [record.data() for record in records]
        if not candidates:
            return []
        query_vector = (await self.embedder.embed([query], "query"))[0]
        ranked = sorted(
            candidates,
            key=lambda row: cosine_similarity(query_vector, row["embedding"]),
            reverse=True,
        )
        return [row["key"] for row in ranked[: self.embedding_seed_count]]


_STOP_WORDS = {"and", "are", "for", "from", "how", "the", "this", "was", "what", "why"}


def _statement_evidence(row: dict, score: float) -> Evidence:
    return Evidence(
        id=row["id"],
        document_id=row["document_id"],
        document_title=row["document_title"],
        text=row["text"],
        page=row.get("page"),
        section=row.get("section"),
        score=score,
        acl_groups=set(row.get("acl_groups") or []),
        context_text=row["text"],
    )


def _query_terms(query: str) -> list[str]:
    terms = (term.strip(".,?!:;()[]{}\"'").casefold() for term in query.split())
    return sorted({term for term in terms if len(term) > 2 and term not in _STOP_WORDS})


def _statement_entity_graph(
    rows: list[dict], terms: list[str], seed_keys: set[str]
) -> tuple[dict[str, set[str]], dict[str, float]]:
    """Builds the bipartite statement-entity adjacency and the personalization seeds."""
    adjacency: dict[str, set[str]] = {}
    seeds: dict[str, float] = {}
    for row in rows:
        statement = f"s:{row['id']}"
        keys = (row["subject_key"], row["object_key"])
        entities = tuple(f"e:{key}" for key in keys)
        adjacency.setdefault(statement, set()).update(entities)
        for entity in entities:
            adjacency.setdefault(entity, set()).add(statement)
        lexical_score = float(row.get("lexical_score") or 0)
        if lexical_score:
            seeds[statement] = lexical_score
        for key, entity in zip(keys, entities, strict=True):
            matches = sum(term in entity.casefold() for term in terms)
            if matches:
                seeds[entity] = float(matches)
            if key in seed_keys:
                # Embedding-seeded fallback: the entity itself is the seed signal
                # since there's no lexical term match to score it by.
                seeds[entity] = max(seeds.get(entity, 0.0), 1.0)
    return adjacency, seeds


def _personalized_pagerank(
    rows: list[dict],
    terms: list[str],
    seed_keys: tuple[str, ...] = (),
    *,
    damping: float = 0.85,
    iterations: int = 20,
) -> list[tuple[dict, float]]:
    adjacency, seeds = _statement_entity_graph(rows, terms, set(seed_keys))
    if not seeds:
        return []
    seed_total = sum(seeds.values())
    personalization = {node: score / seed_total for node, score in seeds.items()}
    scores = {node: personalization.get(node, 0.0) for node in adjacency}
    for _ in range(iterations):
        updated = {node: (1 - damping) * personalization.get(node, 0.0) for node in adjacency}
        for node, neighbors in adjacency.items():
            if not neighbors:
                continue
            contribution = damping * scores[node] / len(neighbors)
            for neighbor in neighbors:
                updated[neighbor] += contribution
        scores = updated
    ranked = [(row, scores.get(f"s:{row['id']}", 0.0)) for row in rows]
    return sorted(ranked, key=lambda item: item[1], reverse=True)


class CompositeRetriever:
    def __init__(self, vector: Retriever, graph: Retriever) -> None:
        self.vector = vector
        self.graph = graph

    async def retrieve(
        self, query: str, route: Route, auth: AuthContext, limit: int
    ) -> list[Evidence]:
        if route in {Route.MULTIHOP, Route.TEMPORAL}:
            graph, vector = await asyncio.gather(
                self.graph.retrieve(query, route, auth, limit // 2),
                self.vector.retrieve(query, route, auth, limit // 2),
            )
            return sorted(graph + vector, key=lambda item: item.score, reverse=True)[:limit]
        return await self.vector.retrieve(query, route, auth, limit)


class RerankingRetriever:
    def __init__(self, retriever: Retriever, reranker: Reranker, max_results: int = 20) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.max_results = max_results

    async def retrieve(
        self, query: str, route: Route, auth: AuthContext, limit: int
    ) -> list[Evidence]:
        evidence = await self.retriever.retrieve(query, route, auth, limit)
        return await self.reranker.rerank(query, evidence, min(limit, self.max_results))
