import logging
from uuid import uuid4

import litellm
import pytest
from app.components.llm import LiteLLMGateway
from app.components.voyage import VoyageGateway
from app.core.config import Settings
from app.schemas.domain import Evidence
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_litellm import ChatLiteLLM
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank


def test_weaviate_hostname_is_normalized_to_https():
    assert Settings(weaviate_url="cluster.weaviate.network").weaviate_url == (
        "https://cluster.weaviate.network"
    )
    assert Settings().weaviate_collection == "FilingSection"


def test_neo4j_uri_accepts_a_local_docker_uri_not_just_aura():
    """Neo4j runs as a local Docker container, not Aura -- any valid scheme
    (neo4j://, bolt://, neo4j+s://) is accepted; there is no longer an
    Aura-only enforcement rule."""
    assert Settings(neo4j_uri="neo4j://neo4j:7687").neo4j_uri == "neo4j://neo4j:7687"


def test_litellm_gateway_drops_unsupported_model_kwargs_instead_of_raising():
    """Some providers (e.g. MiniMax-M2.7-highspeed) reject the "thinking"
    model_kwarg with litellm.exceptions.UnsupportedParamsError instead of
    ignoring it -- without drop_params=True, every structured-output call
    would silently degrade to the keyword-only fallback, undoing the whole
    point of LLM-augmented routing without any visible error."""
    litellm.drop_params = False
    LiteLLMGateway(Settings(llm_api_key="test-key"))
    assert litellm.drop_params is True


@pytest.mark.asyncio
async def test_complete_strips_a_markdown_json_fence_when_json_output(monkeypatch):
    """MiniMax-M2.7-highspeed sometimes wraps JSON-mode output in a ```json
    fence despite response_format={"type": "json_object"} -- observed live,
    not hypothetical. Without stripping it, every downstream
    schema.model_validate_json(raw) call fails and silently degrades to that
    node's keyword/heuristic fallback."""

    async def invoke(model, messages, config=None, **kwargs):
        return AIMessage(
            content='\n\n```json\n{"route":"direct"}\n```',
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(ChatLiteLLM, "ainvoke", invoke)
    gateway = LiteLLMGateway(Settings(llm_api_key="test-key"))

    raw = await gateway.complete([{"role": "user", "content": "route"}], json_output=True)

    assert raw == '{"route":"direct"}'


@pytest.mark.asyncio
async def test_complete_leaves_plain_text_untouched_when_not_json_output(monkeypatch):
    async def invoke(model, messages, config=None, **kwargs):
        return AIMessage(
            content="```not json, just an answer```",
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(ChatLiteLLM, "ainvoke", invoke)
    gateway = LiteLLMGateway(Settings(llm_api_key="test-key"))

    raw = await gateway.complete([{"role": "user", "content": "answer"}])

    assert raw == "```not json, just an answer```"


@pytest.mark.asyncio
async def test_litellm_gateway_uses_flash_and_pro_from_settings(monkeypatch):
    calls: list[str] = []
    configs = []

    async def invoke(model, messages, config=None, **kwargs):
        calls.append(model.model)
        configs.append(config)
        return AIMessage(
            content='{"route":"direct"}',
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(ChatLiteLLM, "ainvoke", invoke)
    gateway = LiteLLMGateway(Settings(llm_api_key="test-key"))

    assert gateway.pro.model_kwargs == {"thinking": {"type": "disabled"}}
    assert gateway.pro.max_tokens == 2000

    await gateway.complete([{"role": "user", "content": "route"}], json_output=True)
    await gateway.complete([{"role": "user", "content": "solve"}], use_pro=True)

    assert calls == ["minimax/MiniMax-M2.7-highspeed", "minimax/MiniMax-M3"]
    assert gateway.input_tokens == 200
    assert gateway.output_tokens == 40
    assert gateway.estimated_cost_usd > 0
    assert configs[0]["metadata"]["index_version"] == "v1"


@pytest.mark.asyncio
async def test_provider_swap_via_config_only_no_code_change(monkeypatch):
    """Same LiteLLMGateway class, only Settings change -- proves a provider swap
    away from the MiniMax default (here, to DeepSeek) needs no branching or
    provider-specific code anywhere."""
    calls: list[str] = []

    async def invoke(model, messages, config=None, **kwargs):
        calls.append(model.model)
        return AIMessage(
            content='{"ok":true}',
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

    monkeypatch.setattr(ChatLiteLLM, "ainvoke", invoke)
    settings = Settings(
        llm_flash_model="deepseek/deepseek-v4-flash",
        llm_pro_model="deepseek/deepseek-v4-pro",
        llm_api_key="fake-deepseek-key",
    )
    gateway = LiteLLMGateway(settings)

    await gateway.complete([{"role": "user", "content": "x"}], json_output=True)

    assert calls == ["deepseek/deepseek-v4-flash"]
    assert gateway.input_tokens == 10


def test_litellm_gateway_warns_when_no_credentials_configured(caplog):
    with caplog.at_level(logging.WARNING):
        LiteLLMGateway(Settings())

    assert any("llm_api_key" in record.message for record in caplog.records)


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
