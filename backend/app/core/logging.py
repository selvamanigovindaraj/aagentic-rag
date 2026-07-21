from __future__ import annotations

import logging

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",  # set by Formatter.format() itself via record.getMessage()
    "asctime",  # set when the format string includes %(asctime)s
}


class ExtraFieldsFormatter(logging.Formatter):
    """Appends any `extra={...}` fields to the base log line so structured
    context (tenant_id, reason, duration_ms, ...) isn't silently dropped by
    the default formatter."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_ATTRS and not key.startswith("_")
        }
        if not extras:
            return base
        fields = " ".join(f"{key}={value!r}" for key, value in sorted(extras.items()))
        return f"{base} {fields}"


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(ExtraFieldsFormatter("%(levelname)s:%(name)s:%(message)s"))
    logging.basicConfig(level=level, handlers=[handler])
