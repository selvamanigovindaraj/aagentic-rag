import logging

from app.core.logging import ExtraFieldsFormatter


def test_extra_fields_are_appended_to_the_log_line():
    formatter = ExtraFieldsFormatter("%(levelname)s:%(name)s:%(message)s")
    record = logging.LogRecord(
        name="app.components.retrieval",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="weaviate_search",
        args=None,
        exc_info=None,
    )
    record.tenant_id = "tenant-a"
    record.result_count = 5

    line = formatter.format(record)

    assert line.startswith("INFO:app.components.retrieval:weaviate_search")
    assert "tenant_id='tenant-a'" in line
    assert "result_count=5" in line


def test_record_without_extras_is_unchanged():
    formatter = ExtraFieldsFormatter("%(message)s")
    record = logging.LogRecord(
        name="app",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="plain",
        args=None,
        exc_info=None,
    )

    assert formatter.format(record) == "plain"
