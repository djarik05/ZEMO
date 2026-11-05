"""
Zemo structured, privacy-aware logging.

Log formatters redact any value that looks like a secret or PII
before writing the record to the output stream.
"""

from __future__ import annotations

import logging
import re
import sys
from app.core.config import get_settings

# Patterns that MUST be redacted from log output
_REDACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(api[_-]?key|secret|password|token|authorization|bearer)\s*[:=]\s*\S+"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    re.compile(r"\b(?:sk|pk|rk|ak|AKIA)[A-Za-z0-9_\-]{16,}\b"),
]

_REPLACEMENT = "[REDACTED]"


def _scrub(message: str) -> str:
    """Remove secrets and PII patterns from a log string."""
    for pat in _REDACT_PATTERNS:
        message = pat.sub(_REPLACEMENT, message)
    return message


class SafeFormatter(logging.Formatter):
    """Formatter that scrubs sensitive data before output."""

    def format(self, record: logging.LogRecord) -> str:
        return _scrub(super().format(record))


def setup_logging() -> None:
    """Configure root logger with a privacy-safe formatter."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = SafeFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger("zemo")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False

    # Quiet noisy third-party loggers
    for name in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``zemo`` namespace."""
    return logging.getLogger(f"zemo.{name}")
