"""Structured JSON logging with contextvars propagation for run/shot/stage metadata."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_LOG_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "gtm_videogen_log_context", default=None
)

_STANDARD_FIELDS = ("run_id", "version_id", "shot_id", "stage", "duration_ms")


def get_log_context() -> dict[str, Any]:
    """Return a copy of the current structured logging context."""
    return dict(_LOG_CONTEXT.get() or {})


def bind_log_context(**kwargs: Any) -> dict[str, Any]:
    """Merge non-None context fields into the current ContextVar."""
    current = dict(_LOG_CONTEXT.get() or {})
    for key, value in kwargs.items():
        if value is not None:
            current[key] = value
    _LOG_CONTEXT.set(current)
    return current


def clear_log_context() -> None:
    """Reset the structured logging context."""
    _LOG_CONTEXT.set(None)


@contextmanager
def log_context(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Temporarily bind structured logging fields for the duration of a block."""
    prev = _LOG_CONTEXT.get()
    merged = dict(prev or {})
    for key, value in kwargs.items():
        if value is not None:
            merged[key] = value
    token = _LOG_CONTEXT.set(merged)
    try:
        yield merged
    finally:
        _LOG_CONTEXT.reset(token)


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON with run/version/shot/stage/duration_ms."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        ctx = _LOG_CONTEXT.get() or {}
        for key, value in ctx.items():
            if value is not None:
                payload[key] = value

        for field_name in _STANDARD_FIELDS:
            if hasattr(record, field_name):
                val = getattr(record, field_name)
                if val is not None:
                    payload[field_name] = val

        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            for key, value in extra_fields.items():
                if value is not None:
                    payload[key] = value

        if record.exc_info:
            payload["err"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def get_logger(name: str, level: str | int = logging.INFO) -> logging.Logger:
    """Create or return a structured JSON logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
