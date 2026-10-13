"""
Zemo domain enumerations and lightweight internal models.

These are not Pydantic schemas — they are plain Python types
used throughout the service layer.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Enumerations ─────────────────────────────────────────────────

class DatasetFormat(str, enum.Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"


class DatasetStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    PROFILING = "profiling"
    PROFILED = "profiled"
    SCANNING = "scanning"
    SCANNED = "scanned"
    SANITIZING = "sanitizing"
    SANITIZED = "sanitized"
    VALIDATED = "validated"
    READY = "ready"
    FAILED = "failed"


class ScanStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Severity(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class FindingCategory(str, enum.Enum):
    # Privacy
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    PERSONAL_ID = "personal_id"
    # Financial
    CREDIT_CARD = "credit_card"
    FINANCIAL = "financial"
    # Secrets
    API_KEY = "api_key"
    PASSWORD = "password"
    ACCESS_TOKEN = "access_token"
    PRIVATE_KEY = "private_key"
    DB_CREDENTIAL = "db_credential"
    # Security
    PROMPT_INJECTION = "prompt_injection"
    SUSPICIOUS_CONTENT = "suspicious_content"
    MALFORMED_RECORD = "malformed_record"
    # Quality
    MISSING_VALUE = "missing_value"
    DUPLICATE_RECORD = "duplicate_record"
    SCHEMA_INCONSISTENCY = "schema_inconsistency"
    FORMAT_ERROR = "format_error"
    ENCODING_ERROR = "encoding_error"
    EMPTY_RECORD = "empty_record"
    EXCESSIVE_REPETITION = "excessive_repetition"


class PolicyDecision(str, enum.Enum):
    ALLOW = "allow"
    REVIEW = "review"
    SANITIZE = "sanitize"
    BLOCK = "block"


class ReviewStatus(str, enum.Enum):
    PENDING = "pending"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class RemediationStatus(str, enum.Enum):
    PENDING = "pending"
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


class ReadinessStatus(str, enum.Enum):
    READY = "ready"
    NOT_READY = "not_ready"
    BLOCKED = "blocked"
    INCOMPLETE = "incomplete"


class IntendedOperation(str, enum.Enum):
    TRAINING = "training"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"
    INFERENCE = "inference"


class DetectionMethod(str, enum.Enum):
    REGEX = "regex"
    HEURISTIC = "heuristic"
    ENTROPY = "entropy"
    FIELD_NAME = "field_name"
    VALIDATION = "validation"
    ML_CLASSIFIER = "ml_classifier"
    STRUCTURAL = "structural"
    STATISTICAL = "statistical"


# ── Utility ──────────────────────────────────────────────────────

def utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)
