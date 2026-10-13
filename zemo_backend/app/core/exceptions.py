"""
Zemo custom exception hierarchy.

All exceptions carry a stable `code` for API error serialisation
and an optional safe `detail` dict that must never contain raw secrets.
"""

from __future__ import annotations
from typing import Any


class ZemoError(Exception):
    """Base for all Zemo application errors."""

    code: str = "ZEMO_ERROR"
    status_code: int = 500

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


# ── Ingestion ────────────────────────────────────────────────────
class UnsupportedFormatError(ZemoError):
    code = "UNSUPPORTED_FORMAT"
    status_code = 400


class FileTooLargeError(ZemoError):
    code = "FILE_TOO_LARGE"
    status_code = 413


class DatasetParsingError(ZemoError):
    code = "PARSING_ERROR"
    status_code = 422


class RecordLimitExceededError(ZemoError):
    code = "RECORD_LIMIT_EXCEEDED"
    status_code = 422


# ── Lookup ───────────────────────────────────────────────────────
class DatasetNotFoundError(ZemoError):
    code = "DATASET_NOT_FOUND"
    status_code = 404


class ScanNotFoundError(ZemoError):
    code = "SCAN_NOT_FOUND"
    status_code = 404


class FindingNotFoundError(ZemoError):
    code = "FINDING_NOT_FOUND"
    status_code = 404


class OperationNotFoundError(ZemoError):
    code = "OPERATION_NOT_FOUND"
    status_code = 404


# ── Processing ───────────────────────────────────────────────────
class ScanAlreadyRunningError(ZemoError):
    code = "SCAN_ALREADY_RUNNING"
    status_code = 409


class SanitizationError(ZemoError):
    code = "SANITIZATION_ERROR"
    status_code = 500


class ValidationError(ZemoError):
    code = "VALIDATION_ERROR"
    status_code = 422


# ── Security ─────────────────────────────────────────────────────
class PathTraversalError(ZemoError):
    code = "PATH_TRAVERSAL"
    status_code = 400


class ResourceExhaustedError(ZemoError):
    code = "RESOURCE_EXHAUSTED"
    status_code = 429


# ── ML ───────────────────────────────────────────────────────────
class ModelNotAvailableError(ZemoError):
    code = "MODEL_NOT_AVAILABLE"
    status_code = 503
