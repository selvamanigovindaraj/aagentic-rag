from __future__ import annotations

import logging
import os
from typing import Protocol

import litellm
from langchain_core.messages import AIMessage
from langchain_litellm import ChatLiteLLM

from ..core.config import Settings

logger = logging.getLogger(__name__)


class ModelGateway(Protocol):
    async def complete(
        self, messages: list[dict[str, str]], *, use_pro: bool = False, json_output: bool = False
    ) -> str: ...


def _strip_json_fence(text: str) -> str:
    """Some models wrap JSON-mode output in a ```json fence despite
    response_format={"type": "json_object"} -- observed live on MiniMax-M2.7-
    highspeed. Unwrap it so schema.model_validate_json(raw) still parses."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    _, _, body = stripped.partition("\n")
    return body.removesuffix("```").strip()


class LiteLLMGateway:
    """Provider-agnostic completion models via LiteLLM (a LangChain integration,
    per this project's "always go through LangChain provider integrations"
    convention -- never a hand-written provider HTTP client). Swapping providers
    (DeepSeek, MiniMax, or any other LiteLLM-supported one) is a config/env-var
    change only: the model strings and the provider's own credential env var."""

    def __init__(self, settings: Settings) -> None:
        # Some providers reject an optional model_kwarg outright (e.g.
        # MiniMax-M2.7-highspeed raises UnsupportedParamsError for "thinking")
        # instead of ignoring it like most do. Without this, every such call
        # would raise, get caught by each node's own fallback handler, and
        # silently degrade to the keyword-only path -- invisibly undoing
        # LLM-augmented routing/generation with no visible error.
        litellm.drop_params = True
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        if not settings.allow_sensitive_tracing:
            os.environ["LANGCHAIN_HIDE_INPUTS"] = "true"
            os.environ["LANGCHAIN_HIDE_OUTPUTS"] = "true"
        if not settings.llm_api_key and not settings.llm_base_url:
            logger.warning(
                "Neither llm_api_key nor llm_base_url is set; relying on LiteLLM's "
                "provider-specific env var auto-discovery (e.g. DEEPSEEK_API_KEY)"
            )
        self.flash = self._build_model(settings.llm_flash_model, settings)
        self.pro = self._build_model(settings.llm_pro_model, settings)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_cost_usd = 0.0
        self.settings = settings

    @staticmethod
    def _build_model(model: str, settings: Settings) -> ChatLiteLLM:
        kwargs = {
            "model": model,
            "temperature": 0,
            "max_tokens": 2000,
            "request_timeout": 120,
            "max_retries": 2,
            # ponytail: DeepSeek honors this and returns a clean answer; other
            # providers (e.g. MiniMax) silently ignore it and keep reasoning --
            # harmless (response.text still extracts just the answer text), just
            # not cost/latency-optimal on non-DeepSeek providers.
            "model_kwargs": {"thinking": {"type": "disabled"}},
        }
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        if settings.llm_base_url:
            kwargs["api_base"] = settings.llm_base_url
        return ChatLiteLLM(**kwargs)

    def _trace_config(self, model_name: str) -> dict:
        return {
            "metadata": {
                "index_version": self.settings.index_version,
                "prompt_version": "v1",
                "content_tracing": self.settings.allow_sensitive_tracing,
            },
            "tags": [model_name, self.settings.index_version],
        }

    async def complete(
        self, messages: list[dict[str, str]], *, use_pro: bool = False, json_output: bool = False
    ) -> str:
        self.calls += 1
        model = self.pro if use_pro else self.flash
        model_name = self.settings.llm_pro_model if use_pro else self.settings.llm_flash_model
        runnable = model.bind(response_format={"type": "json_object"}) if json_output else model
        config = self._trace_config(model_name)
        response: AIMessage = await runnable.ainvoke(messages, config=config)
        self._track_usage(response, use_pro)
        if not response.text:
            raise ValueError("LLM returned empty content")
        return _strip_json_fence(response.text) if json_output else response.text

    def _track_usage(self, response: AIMessage, use_pro: bool) -> None:
        usage = response.usage_metadata or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if use_pro:
            input_rate = self.settings.llm_pro_input_usd_per_million
            output_rate = self.settings.llm_pro_output_usd_per_million
        else:
            input_rate = self.settings.llm_flash_input_usd_per_million
            output_rate = self.settings.llm_flash_output_usd_per_million
        self.estimated_cost_usd += (
            input_tokens * input_rate + output_tokens * output_rate
        ) / 1_000_000
