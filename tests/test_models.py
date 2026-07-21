from uuid import uuid4

import pytest
from app.components.llm import DeepSeekGateway
from app.components.voyage import VoyageGateway
from app.core.config import Settings
from app.schemas.domain import Evidence
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank


def test_weaviate_hostname_is_normalized_to_https():
    assert Settings(weaviate_url="cluster.weaviate.network").weaviate_url == (
        "https://cluster.weaviate.network"
    )
    assert Settings().weaviate_collection == "FilingSection"


@pytest.mark.asyncio
async def test_deepseek_uses_langchain_flash_and_pro(monkeypatch):
    calls: list[str] = []
    configs = []

    async def invoke(model, messages, config=None, **kwargs):
        calls.append(model.model_name)
        configs.append(config)
        return AIMessage(
            content='{"route":"direct"}',
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(ChatDeepSeek, "ainvoke", invoke)
    gateway = DeepSeekGateway(Settings(deepseek_api_key="test-key"))

    assert gateway.pro.extra_body == {"thinking": {"type": "disabled"}}
    assert gateway.pro.max_tokens == 2000

    await gateway.complete([{"role": "user", "content": "route"}], json_output=True)
    await gateway.complete([{"role": "user", "content": "solve"}], use_pro=True)

    assert calls == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert gateway.input_tokens == 200
    assert gateway.output_tokens == 40
    assert gateway.estimated_cost_usd > 0
    assert configs[0]["metadata"]["index_version"] == "v1"


@pytest.mark.asyncio
async def test_voyage_uses_langchain_embedding_and_reranker(monkeypatch):
    async def embed_query(model, text):
        assert model.model == "voyage-4-lite"
        return [0.1, 0.2]

    async def compress(reranker, documents, query, callbacks=None):
        assert reranker.model == "rerank-2.5-lite"
        return [
            Document(
                page_content=documents[0].page_content,
                metadata={**documents[0].metadata, "relevance_score": 0.99},
            )
        ]

    monkeypatch.setattr(VoyageAIEmbeddings, "aembed_query", embed_query)
    monkeypatch.setattr(VoyageAIRerank, "acompress_documents", compress)
    gateway = VoyageGateway(Settings(voyage_api_key="test-key"))
    evidence = Evidence(
        id="e1",
        document_id=uuid4(),
        document_title="Doc",
        text="Evidence",
        score=0.1,
    )

    assert await gateway.embed(["query"], "query") == [[0.1, 0.2]]
    ranked = await gateway.rerank("query", [evidence], 1)
    assert ranked[0].score == 0.99
