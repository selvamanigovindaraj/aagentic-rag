from __future__ import annotations

import asyncio
import math
from typing import Literal, Protocol

import voyageai.error
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from ..core.config import Settings
from ..schemas.domain import Evidence

# Concurrent retrieval/eval traffic can burst past Voyage's per-minute token
# rate limit (observed live: rerank-2.5-lite's 4M TPM ceiling under
# concurrency=8) -- retry past it rather than letting a single burst crash
# the whole retrieve() call for real user traffic, not just eval scripts.
_retry_on_rate_limit = retry(
    retry=retry_if_exception_type(voyageai.error.RateLimitError),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(6),
)


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

    def __init__(self, settings: Settings, max_concurrent_calls: int = 2) -> None:
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
        # Caps concurrent Voyage calls regardless of how many concurrent
        # callers there are (retrieve() calls from N concurrent retrieval_eval
        # cases, or N concurrent chat runs) -- retrying a rate limit under
        # high concurrency just re-submits more token volume into the same
        # per-minute window and never lets it recover (same failure mode
        # ingest()'s upload pacing works around); this throttles at the
        # source instead.
        self._semaphore = asyncio.Semaphore(max_concurrent_calls)

    @_retry_on_rate_limit
    async def embed(
        self, texts: list[str], input_type: Literal["query", "document"]
    ) -> list[list[float]]:
        async with self._semaphore:
            if input_type == "query":
                return [await self.embeddings.aembed_query(texts[0])]
            return await self.embeddings.aembed_documents(texts)

    @_retry_on_rate_limit
    async def rerank(self, query: str, evidence: list[Evidence], limit: int) -> list[Evidence]:
        if not evidence:
            return []
        documents = [
            Document(page_content=item.text, metadata={"index": index})
            for index, item in enumerate(evidence)
        ]
        async with self._semaphore:
            ranked = await self.reranker.acompress_documents(documents, query)
        return [
            evidence[item.metadata["index"]].model_copy(
                update={"score": item.metadata.get("relevance_score", 0.0)}
            )
            for item in ranked[:limit]
        ]
