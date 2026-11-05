"""Pydantic schemas for risk assessment and policy engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import PolicyDecision, Severity


class RiskBreakdown(BaseModel):
    privacy_score: float = Field(ge=0.0, le=100.0, default=0.0)
    secret_score: float = Field(ge=0.0, le=100.0, default=0.0)
    security_score: float = Field(ge=0.0, le=100.0, default=0.0)
    quality_score: float = Field(ge=0.0, le=100.0, default=0.0)


class RiskAssessment(BaseModel):
    dataset_id: str
    overall_score: float = Field(ge=0.0, le=100.0, default=0.0)
    breakdown: RiskBreakdown = Field(default_factory=RiskBreakdown)
    risk_level: Severity = Severity.LOW
    assessed_at: datetime | None = None


class PolicyRule(BaseModel):
    rule_id: str
    name: str
    description: str
    category: str
    severity_trigger: Severity
    decision: PolicyDecision
    mandatory: bool = False
    enabled: bool = True


class PolicyEvaluation(BaseModel):
    dataset_id: str
    overall_decision: PolicyDecision
    triggered_rules: list[PolicyRule] = Field(default_factory=list)
    mandatory_blocks: int = 0
    explanation: str = ""
    evaluated_at: datetime | None = None


class PolicyConfig(BaseModel):
    """User-configurable policy overrides."""
    custom_rules: list[PolicyRule] = Field(default_factory=list)
    risk_threshold_block: float = Field(ge=0, le=100, default=80.0)
    risk_threshold_review: float = Field(ge=0, le=100, default=50.0)
    auto_sanitize_medium: bool = False
