from redis.asyncio import Redis

from ..core.config import Settings
from ..repositories.store import Store
from ..services.semantic_cache import SemanticCache
from .retrieval import (
    ActiveDocumentRetriever,
    CachedRetriever,
    CompositeRetriever,
    EmptyRetriever,
    Neo4jRetriever,
    RerankingRetriever,
    Retriever,
    WeaviateRetriever,
)
from .voyage import VoyageGateway

__all__ = [
    "ActiveDocumentRetriever",
    "CachedRetriever",
    "CompositeRetriever",
    "EmptyRetriever",
    "Neo4jRetriever",
    "RerankingRetriever",
    "Retriever",
    "WeaviateRetriever",
    "build_retrieval_stack",
]


def build_retrieval_stack(
    settings: Settings,
    embedder: VoyageGateway,
    graph_retriever: Neo4jRetriever,
    redis: Redis,
    store: Store,
) -> Retriever:
    """The one production retriever stack, shared by the api and the worker."""
    return ActiveDocumentRetriever(
        CachedRetriever(
            RerankingRetriever(
                CompositeRetriever(
                    WeaviateRetriever(
                        settings.weaviate_url,
                        settings.weaviate_api_key,
                        embedder,
                        settings.weaviate_collection,
                        settings.index_version,
                    ),
                    graph_retriever,
                ),
                embedder,
                settings.max_reranked_candidates,
            ),
            SemanticCache(redis),
            settings.index_version,
        ),
        store,
    )
