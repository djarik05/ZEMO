"""Pydantic schemas for findings across all detection engines."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import (
    FindingCategory,
    Severity,
    DetectionMethod,
    PolicyDecision,
    ReviewStatus,
    RemediationStatus,
)


class FindingBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    finding_id: str
    category: FindingCategory
    subcategory: str = ""
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    dataset_id: str
    record_index: int | None = None
    field_name: str | None = None
    location: str = ""
    evidence: str = ""            # always masked / redacted
    explanation: str = ""
    detection_method: DetectionMethod = DetectionMethod.REGEX
    recommended_action: PolicyDecision = PolicyDecision.REVIEW
    review_status: ReviewStatus = ReviewStatus.PENDING
    remediation_status: RemediationStatus = RemediationStatus.PENDING
    created_at: datetime


class FindingDetail(FindingBase):
    """Extended view returned when fetching a single finding."""
    raw_value_masked: str = ""    # additional masked evidence
    impact: str = ""
    remediation_guidance: str = ""


class FindingSummary(BaseModel):
    total: int = 0
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    informational: int = 0
    by_category: dict[str, int] = Field(default_factory=dict)


class FindingListResponse(BaseModel):
    dataset_id: str
    findings: list[FindingBase]
    summary: FindingSummary
