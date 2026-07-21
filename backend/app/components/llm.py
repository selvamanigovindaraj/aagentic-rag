from __future__ import annotations

import os
from typing import Protocol

from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from ..core.config import Settings


class ModelGateway(Protocol):
    async def complete(
        self, messages: list[dict[str, str]], *, use_pro: bool = False, json_output: bool = False
    ) -> str: ...


class DeepSeekGateway:
    """LangChain-only DeepSeek completion models."""

    def __init__(self, settings: Settings) -> None:
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        if not settings.allow_sensitive_tracing:
            os.environ["LANGCHAIN_HIDE_INPUTS"] = "true"
            os.environ["LANGCHAIN_HIDE_OUTPUTS"] = "true"
        common = {
            "api_key": settings.deepseek_api_key,
            "base_url": settings.deepseek_base_url,
            "temperature": 0,
            "timeout": 120,
            "max_retries": 2,
        }
        self.flash = ChatDeepSeek(
            model=settings.deepseek_flash_model,
            max_tokens=2000,
            extra_body={"thinking": {"type": "disabled"}},
            **common,
        )
        self.pro = ChatDeepSeek(
            model=settings.deepseek_pro_model,
            max_tokens=2000,
            extra_body={"thinking": {"type": "disabled"}},
            **common,
        )
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_cost_usd = 0.0
        self.settings = settings

    async def complete(
        self, messages: list[dict[str, str]], *, use_pro: bool = False, json_output: bool = False
    ) -> str:
        self.calls += 1
        model = self.pro if use_pro else self.flash
        model_name = (
            self.settings.deepseek_pro_model if use_pro else self.settings.deepseek_flash_model
        )
        runnable = model.bind(response_format={"type": "json_object"}) if json_output else model
        response: AIMessage = await runnable.ainvoke(
            messages,
            config={
                "metadata": {
                    "index_version": self.settings.index_version,
                    "prompt_version": "v1",
                    "content_tracing": self.settings.allow_sensitive_tracing,
                },
                "tags": [model_name, self.settings.index_version],
            },
        )
        usage = response.usage_metadata or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if use_pro:
            input_rate = self.settings.deepseek_pro_input_usd_per_million
            output_rate = self.settings.deepseek_pro_output_usd_per_million
        else:
            input_rate = self.settings.deepseek_flash_input_usd_per_million
            output_rate = self.settings.deepseek_flash_output_usd_per_million
        self.estimated_cost_usd += (
            input_tokens * input_rate + output_tokens * output_rate
        ) / 1_000_000
        if not response.text:
            raise ValueError("DeepSeek returned empty content")
        return response.text
