from __future__ import annotations

import math
from typing import Literal, Protocol

from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank

from ..core.config import Settings
from ..schemas.domain import Evidence


def cosine_similarity(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    if not denominator:
        return 0.0
    return sum(x * y for x, y in zip(left, right, strict=True)) / denominator


class Embedder(Protocol):
    async def embed(
        self, texts: list[str], input_type: Literal["query", "document"]
    ) -> list[list[float]]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, evidence: list[Evidence], limit: int) -> list[Evidence]: ...


class VoyageGateway:
    """LangChain-only Voyage embedding and reranking models."""

    def __init__(self, settings: Settings) -> None:
        if not settings.voyage_api_key:
            raise ValueError("VOYAGE_API_KEY is required")
        common = {"api_key": settings.voyage_api_key, "base_url": settings.voyage_base_url}
        self.embeddings = VoyageAIEmbeddings(
            model=settings.voyage_embedding_model,
            **common,
        )
        self.reranker = VoyageAIRerank(
            model=settings.voyage_rerank_model,
            **common,
        )

    async def embed(
        self, texts: list[str], input_type: Literal["query", "document"]
    ) -> list[list[float]]:
        if input_type == "query":
            return [await self.embeddings.aembed_query(texts[0])]
        return await self.embeddings.aembed_documents(texts)

    async def rerank(self, query: str, evidence: list[Evidence], limit: int) -> list[Evidence]:
        if not evidence:
            return []
        documents = [
            Document(page_content=item.text, metadata={"index": index})
            for index, item in enumerate(evidence)
        ]
        ranked = await self.reranker.acompress_documents(documents, query)
        return [
            evidence[item.metadata["index"]].model_copy(
                update={"score": item.metadata.get("relevance_score", 0.0)}
            )
            for item in ranked[:limit]
        ]
