from __future__ import annotations

import os

from .config import get_settings


def configure_tracing() -> None:
    """Auto-instruments every LangChain/LangGraph run via OpenInference's
    LangChainInstrumentor -- no call-site changes needed, traces flow to
    Phoenix's OTLP collector the same way LiteLLM calls already flow to
    LangSmith when LANGCHAIN_TRACING_V2 is set.

    Prompts and responses are captured in full (hide_inputs/hide_outputs=False
    below): unlike the LANGCHAIN_HIDE_INPUTS/OUTPUTS gate on LangSmith tracing
    (allow_sensitive_tracing in llm.py), OpenInference has no equivalent
    default-hidden behavior, but this is set explicitly rather than relying on
    that default staying unhidden."""
    settings = get_settings()
    if not settings.phoenix_enabled:
        return
    if "PYTEST_CURRENT_TEST" in os.environ:
        # main.py calls this at import time; tests importing `app.main`
        # (e.g. test_api.py's TestClient) would otherwise silently start
        # exporting real spans -- including from other tests in the same
        # pytest process that build a live RagPipeline -- to Phoenix.
        return
    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    tracer_provider = register(
        project_name=settings.phoenix_project_name,
        endpoint=settings.phoenix_collector_endpoint,
        auto_instrument=False,
        batch=True,
        # register()'s default banner print goes to stdout -- eval scripts
        # (retrieval_eval.py etc.) print their JSON report to stdout too, and
        # a script invoked as `... > report.json` would get the banner
        # prepended, corrupting the file as JSON.
        verbose=False,
    )
    trace_config = TraceConfig(
        hide_inputs=False,
        hide_outputs=False,
        hide_input_messages=False,
        hide_output_messages=False,
        hide_input_text=False,
        hide_output_text=False,
        hide_prompts=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider, config=trace_config)
