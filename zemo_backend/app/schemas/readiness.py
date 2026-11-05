"""Pydantic schemas for training readiness assessment."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import ReadinessStatus, IntendedOperation


class ReadinessCheck(BaseModel):
    check_name: str
    passed: bool
    message: str
    weight: float = 1.0


class ReadinessReport(BaseModel):
    dataset_id: str
    intended_operation: IntendedOperation = IntendedOperation.TRAINING
    overall_status: ReadinessStatus
    overall_score: float = Field(ge=0.0, le=100.0, default=0.0)
    checks: list[ReadinessCheck] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    assessed_at: datetime | None = None
    message: str = ""
