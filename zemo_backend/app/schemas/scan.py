"""Pydantic schemas for scan orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import ScanStatus
from app.schemas.finding import FindingSummary


class ScanRequest(BaseModel):
    dataset_id: str
    enable_privacy: bool = True
    enable_secrets: bool = True
    enable_security: bool = True
    enable_quality: bool = True
    enable_ml: bool = False


class StageResult(BaseModel):
    name: str
    status: str           # successful | failed | skipped
    finding_count: int = 0
    duration_ms: int = 0
    message: str = ""


class ScanResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    scan_id: str
    dataset_id: str
    status: ScanStatus
    stages: list[StageResult] = Field(default_factory=list)
    summary: FindingSummary = Field(default_factory=FindingSummary)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int = 0
    error_message: str | None = None


class ScanListItem(BaseModel):
    scan_id: str
    dataset_id: str
    status: ScanStatus
    finding_count: int = 0
    started_at: datetime | None = None
