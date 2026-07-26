from unittest.mock import MagicMock

from app.core.config import Settings
from app.core.tracing import configure_tracing


def test_configure_tracing_is_a_noop_when_phoenix_disabled(monkeypatch):
    monkeypatch.setattr(
        "app.core.tracing.get_settings", lambda: Settings(phoenix_enabled=False)
    )
    import openinference.instrumentation.langchain as oi_langchain

    instrument = MagicMock()
    monkeypatch.setattr(oi_langchain.LangChainInstrumentor, "instrument", instrument)

    configure_tracing()

    instrument.assert_not_called()


def test_configure_tracing_is_a_noop_under_pytest_even_when_enabled(monkeypatch):
    monkeypatch.setattr(
        "app.core.tracing.get_settings", lambda: Settings(phoenix_enabled=True)
    )
    import openinference.instrumentation.langchain as oi_langchain

    instrument = MagicMock()
    monkeypatch.setattr(oi_langchain.LangChainInstrumentor, "instrument", instrument)

    configure_tracing()  # PYTEST_CURRENT_TEST is set by pytest itself here

    instrument.assert_not_called()


def test_configure_tracing_does_not_hide_prompts_or_responses(monkeypatch):
    monkeypatch.setattr(
        "app.core.tracing.get_settings",
        lambda: Settings(phoenix_enabled=True, phoenix_project_name="test-project"),
    )
    # Simulate a real (non-pytest) invocation -- pytest itself sets this env
    # var for the duration of every test, which configure_tracing() checks to
    # avoid exporting other tests' real RagPipeline runs to Phoenix.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    register = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("phoenix.otel.register", register)
    import openinference.instrumentation.langchain as oi_langchain

    instrument = MagicMock()
    monkeypatch.setattr(oi_langchain.LangChainInstrumentor, "instrument", instrument)

    configure_tracing()

    _, kwargs = instrument.call_args
    trace_config = kwargs["config"]
    assert trace_config.hide_inputs is False
    assert trace_config.hide_outputs is False
    assert trace_config.hide_input_messages is False
    assert trace_config.hide_output_messages is False


def test_configure_tracing_silences_the_stdout_banner(monkeypatch):
    # register()'s banner print pollutes eval scripts that redirect stdout
    # to a JSON report file (e.g. `generation_eval.py > report.json`).
    monkeypatch.setattr(
        "app.core.tracing.get_settings",
        lambda: Settings(phoenix_enabled=True, phoenix_project_name="test-project"),
    )
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    register = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("phoenix.otel.register", register)
    import openinference.instrumentation.langchain as oi_langchain

    monkeypatch.setattr(oi_langchain.LangChainInstrumentor, "instrument", MagicMock())

    configure_tracing()

    assert register.call_args.kwargs["verbose"] is False
