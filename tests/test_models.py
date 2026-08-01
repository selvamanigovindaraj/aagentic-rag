import asyncio
import logging
from uuid import uuid4

import litellm
import pytest
from app.components import llm as llm_module
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
async def test_complete_strips_a_think_block_before_the_json_when_json_output(monkeypatch):
    """MiniMax-M3 (the pro model, used only for multihop/temporal grounded-answer
    calls) always wraps json_output=True completions in a <think>...</think>
    reasoning block despite model_kwargs={"thinking": {"type": "disabled"}} being
    silently dropped for this provider -- observed live via a direct gateway call,
    not hypothetical. Without stripping it, GroundedAnswer.model_validate_json(raw)
    fails deterministically on every multihop/temporal call, and _generate falls
    through to _numbered_fallback (dumping raw evidence) instead of the intended
    refusal/citation paths."""

    async def invoke(model, messages, config=None, **kwargs):
        return AIMessage(
            content='<think>\nThe user wants JSON.\n</think>\n{"route":"direct"}',
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(ChatLiteLLM, "ainvoke", invoke)
    gateway = LiteLLMGateway(Settings(llm_api_key="test-key"))

    raw = await gateway.complete([{"role": "user", "content": "route"}], json_output=True)

    assert raw == '{"route":"direct"}'


@pytest.mark.asyncio
async def test_complete_strips_a_think_block_stacked_with_a_markdown_fence(monkeypatch):
    """A model could plausibly wrap output in both quirks at once -- the think
    strip and the fence strip must chain, not branch, or one wrapper alone gets
    handled and the other leaks into the "parsed" JSON."""

    async def invoke(model, messages, config=None, **kwargs):
        return AIMessage(
            content='<think>reasoning</think>\n```json\n{"route":"direct"}\n```',
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(ChatLiteLLM, "ainvoke", invoke)
    gateway = LiteLLMGateway(Settings(llm_api_key="test-key"))

    raw = await gateway.complete([{"role": "user", "content": "route"}], json_output=True)

    assert raw == '{"route":"direct"}'


@pytest.mark.asyncio
async def test_complete_strips_a_trailing_fence_with_no_matching_opening_fence(monkeypatch):
    """Observed live: even at temperature=0 the model isn't perfectly
    deterministic about wrapping -- one call for the exact same prompt
    produced plain JSON followed by a stray closing ``` with no opening
    fence at all. The old strip logic only handled a fence that STARTS the
    text, so this trailing garbage caused json.loads to fail with
    "Extra data" even though the JSON itself was perfectly well-formed."""

    async def invoke(model, messages, config=None, **kwargs):
        return AIMessage(
            content='{"route":"direct"}\n```',
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(ChatLiteLLM, "ainvoke", invoke)
    gateway = LiteLLMGateway(Settings(llm_api_key="test-key"))

    raw = await gateway.complete([{"role": "user", "content": "route"}], json_output=True)

    assert raw == '{"route":"direct"}'


@pytest.mark.asyncio
async def test_complete_leaves_an_unclosed_think_block_unchanged(monkeypatch):
    """A reasoning block truncated by max_tokens before its closing tag can't be
    safely stripped (no way to know where reasoning ends and content begins) --
    the text passes through unchanged and the caller's JSON parse fails as before,
    which is safe because _generate now refuses rather than dumping evidence."""

    async def invoke(model, messages, config=None, **kwargs):
        return AIMessage(
            content="<think>\nreasoning that got cut off before any closing tag",
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(ChatLiteLLM, "ainvoke", invoke)
    gateway = LiteLLMGateway(Settings(llm_api_key="test-key"))

    raw = await gateway.complete([{"role": "user", "content": "route"}], json_output=True)

    assert raw == "<think>\nreasoning that got cut off before any closing tag"


@pytest.mark.asyncio
async def test_complete_recovers_json_from_a_think_block_missing_its_closing_tag(monkeypatch):
    """Observed live: the model sometimes never emits a </think> closing tag
    at all, even though it goes on to produce a complete, well-formed JSON
    answer at the end -- not a truncation (finish_reason was "stop"), just an
    inconsistent formatting habit. A tag-based strip can't help here since
    there's no closing tag to find; scanning for the rightmost valid JSON
    object recovers the real answer regardless."""

    async def invoke(model, messages, config=None, **kwargs):
        return AIMessage(
            content=(
                '<think>\nLet me analyze the question. The company invests in {things}, '
                'so the answer is clear.\n{"claims":[{"text":"Answer.","evidence_ids":["a"]}],'
                '"unsupported":[]}'
            ),
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(ChatLiteLLM, "ainvoke", invoke)
    gateway = LiteLLMGateway(Settings(llm_api_key="test-key"))

    raw = await gateway.complete([{"role": "user", "content": "route"}], json_output=True)

    assert raw == '{"claims":[{"text":"Answer.","evidence_ids":["a"]}],"unsupported":[]}'


@pytest.mark.asyncio
async def test_complete_prefers_the_final_json_over_a_schema_quoted_in_reasoning(monkeypatch):
    """A model reasoning about the task can quote the target JSON shape
    inline ("I need to return {\"claims\": [], ...}") before producing the
    real answer -- a left-to-right search for the first '{' would grab that
    inline example instead of the actual answer that follows it."""

    async def invoke(model, messages, config=None, **kwargs):
        return AIMessage(
            content=(
                '<think>\nI need to return JSON like {"claims": [], "unsupported": []} '
                "as the schema.\n</think>\n"
                '{"claims":[{"text":"Real answer.","evidence_ids":["a"]}],"unsupported":[]}'
            ),
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(ChatLiteLLM, "ainvoke", invoke)
    gateway = LiteLLMGateway(Settings(llm_api_key="test-key"))

    raw = await gateway.complete([{"role": "user", "content": "route"}], json_output=True)

    assert raw == '{"claims":[{"text":"Real answer.","evidence_ids":["a"]}],"unsupported":[]}'


@pytest.mark.asyncio
async def test_complete_converts_recursion_error_from_deeply_nested_json_to_value_error(
    monkeypatch,
):
    """`JSONDecoder.raw_decode` recurses per nesting level and raises
    RecursionError (not JSONDecodeError) on pathologically deep model output.
    Left uncaught, that would escape `RagPipeline._structured()`'s
    `except (ValueError, json.JSONDecodeError)` fallback and crash the whole
    durable run instead of degrading gracefully like any other malformed
    structured response. The decoder is faked to raise directly rather than
    feeding it a real deeply-nested payload: the nesting depth needed to
    trigger CPython's C-accelerated `_json` scanner varies by build (far
    deeper than `sys.getrecursionlimit()` would suggest), so a depth
    calibrated against one interpreter isn't a reliable trigger on another."""

    class _RaisingDecoder:
        @staticmethod
        def raw_decode(_text, _start):
            raise RecursionError

    monkeypatch.setattr(llm_module, "_JSON_DECODER", _RaisingDecoder())

    async def invoke(model, messages, config=None, **kwargs):
        return AIMessage(
            content="{}",
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(ChatLiteLLM, "ainvoke", invoke)
    gateway = LiteLLMGateway(Settings(llm_api_key="test-key"))

    with pytest.raises(ValueError):
        await gateway.complete([{"role": "user", "content": "route"}], json_output=True)


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
    assert gateway.flash.max_tokens == Settings().llm_flash_max_tokens
    assert gateway.pro.max_tokens == Settings().llm_pro_max_tokens

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


@pytest.mark.asyncio
async def test_voyage_rerank_retries_past_a_rate_limit(monkeypatch):
    import voyageai.error

    attempts = {"count": 0}

    async def compress(reranker, documents, query, callbacks=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise voyageai.error.RateLimitError("rate limited")
        return [
            Document(
                page_content=documents[0].page_content,
                metadata={**documents[0].metadata, "relevance_score": 0.5},
            )
        ]

    monkeypatch.setattr(VoyageAIRerank, "acompress_documents", compress)
    gateway = VoyageGateway(Settings(voyage_api_key="test-key"))
    evidence = Evidence(
        id="e1", document_id=uuid4(), document_title="Doc", text="Evidence", score=0.1
    )

    ranked = await gateway.rerank("query", [evidence], 1)

    assert attempts["count"] == 2
    assert ranked[0].score == 0.5


@pytest.mark.asyncio
async def test_voyage_bounds_concurrent_calls_regardless_of_caller_concurrency(monkeypatch):
    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def compress(reranker, documents, query, callbacks=None):
        nonlocal concurrent, max_concurrent
        async with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.01)
        async with lock:
            concurrent -= 1
        return [
            Document(
                page_content=documents[0].page_content,
                metadata={**documents[0].metadata, "relevance_score": 0.5},
            )
        ]

    monkeypatch.setattr(VoyageAIRerank, "acompress_documents", compress)
    gateway = VoyageGateway(Settings(voyage_api_key="test-key"), max_concurrent_calls=2)
    evidence = Evidence(
        id="e1", document_id=uuid4(), document_title="Doc", text="Evidence", score=0.1
    )

    await asyncio.gather(*(gateway.rerank("q", [evidence], 1) for _ in range(6)))

    assert max_concurrent <= 2
