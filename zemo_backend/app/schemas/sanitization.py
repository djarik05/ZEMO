"""Pydantic schemas for sanitization and post-sanitization validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import PolicyDecision, RemediationStatus


class SanitizationAction(BaseModel):
    finding_id: str
    action: str            # redact | mask | remove_record | hash | replace
    target_field: str | None = None
    replacement: str = ""  # only for 'replace' action


class SanitizationRequest(BaseModel):
    dataset_id: str
    actions: list[SanitizationAction] = Field(default_factory=list)
    auto_apply_policy: bool = True   # auto-sanitize per policy decision


class SanitizationRecord(BaseModel):
    finding_id: str
    action: str
    status: RemediationStatus
    field_name: str | None = None
    message: str = ""


class SanitizationResult(BaseModel):
    dataset_id: str
    sanitized_dataset_id: str        # ID of the new, clean dataset version
    original_record_count: int
    sanitized_record_count: int
    records_removed: int = 0
    fields_redacted: int = 0
    fields_masked: int = 0
    fields_hashed: int = 0
    actions_applied: list[SanitizationRecord] = Field(default_factory=list)
    sanitized_at: datetime | None = None
    message: str = ""


class PostValidationResult(BaseModel):
    sanitized_dataset_id: str
    remaining_findings: int = 0
    passed: bool = True
    checks: list[str] = Field(default_factory=list)
    validated_at: datetime | None = None
