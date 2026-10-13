"""
Unit and integration tests for RiskEngine and PolicyEngine.
"""

from __future__ import annotations

import pytest

from app.models.enums import FindingCategory, PolicyDecision, Severity
from app.services.risk_engine import PolicyEngine, RiskEngine


def test_risk_no_findings() -> None:
    """Empty findings should give 0 score, LOW risk level."""
    engine = RiskEngine()
    assessment = engine.assess("ds-risk-1", [])

    assert assessment["overall_score"] == 0.0
    assert assessment["risk_level"] == Severity.LOW.value
    assert assessment["breakdown"]["privacy_score"] == 0.0
    assert assessment["breakdown"]["secret_score"] == 0.0
    assert assessment["breakdown"]["security_score"] == 0.0
    assert assessment["breakdown"]["quality_score"] == 0.0


def test_risk_critical_findings() -> None:
    """Critical findings should give high risk score and CRITICAL level."""
    engine = RiskEngine()
    findings = [
        {"category": FindingCategory.API_KEY.value, "severity": Severity.CRITICAL.value},
        {"category": FindingCategory.PASSWORD.value, "severity": Severity.CRITICAL.value},
        {"category": FindingCategory.DB_CREDENTIAL.value, "severity": Severity.CRITICAL.value},
        {"category": FindingCategory.ACCESS_TOKEN.value, "severity": Severity.CRITICAL.value},
        {"category": FindingCategory.CREDIT_CARD.value, "severity": Severity.CRITICAL.value},
        {"category": FindingCategory.PERSONAL_ID.value, "severity": Severity.CRITICAL.value},
        {"category": FindingCategory.EMAIL.value, "severity": Severity.HIGH.value},
        {"category": FindingCategory.PHONE.value, "severity": Severity.HIGH.value},
        {"category": FindingCategory.PROMPT_INJECTION.value, "severity": Severity.HIGH.value},
        {"category": FindingCategory.SUSPICIOUS_CONTENT.value, "severity": Severity.HIGH.value},
    ]
    assessment = engine.assess("ds-risk-2", findings)

    assert assessment["overall_score"] >= 50.0
    assert assessment["risk_level"] in (Severity.HIGH.value, Severity.CRITICAL.value)


def test_risk_category_breakdown() -> None:
    """Verify privacy/secret/security scores are computed separately."""
    engine = RiskEngine()
    findings = [
        {"category": FindingCategory.EMAIL.value, "severity": Severity.HIGH.value},     # Privacy: 15
        {"category": FindingCategory.API_KEY.value, "severity": Severity.CRITICAL.value}, # Secret: 25
        {"category": FindingCategory.PROMPT_INJECTION.value, "severity": Severity.HIGH.value}, # Security: 15
    ]
    assessment = engine.assess("ds-risk-3", findings)
    breakdown = assessment["breakdown"]

    assert breakdown["privacy_score"] == 15.0
    assert breakdown["secret_score"] == 25.0
    assert breakdown["security_score"] == 15.0
    assert breakdown["quality_score"] == 0.0


def test_policy_block_on_critical_secret() -> None:
    """Mandatory BLOCK for critical secrets."""
    risk_engine = RiskEngine()
    policy_engine = PolicyEngine()

    findings = [
        {"category": FindingCategory.API_KEY.value, "severity": Severity.CRITICAL.value, "confidence": 0.99}
    ]
    risk = risk_engine.assess("ds-pol-1", findings)
    evaluation = policy_engine.evaluate("ds-pol-1", findings, risk)

    assert evaluation["overall_decision"] == PolicyDecision.BLOCK.value
    assert evaluation["mandatory_blocks"] >= 1


def test_policy_sanitize_on_pii() -> None:
    """SANITIZE decision for privacy findings when no mandatory block triggers."""
    risk_engine = RiskEngine()
    policy_engine = PolicyEngine()

    findings = [
        {"category": FindingCategory.EMAIL.value, "severity": Severity.HIGH.value, "confidence": 0.95},
        {"category": FindingCategory.PHONE.value, "severity": Severity.HIGH.value, "confidence": 0.90},
    ]
    risk = risk_engine.assess("ds-pol-2", findings)
    evaluation = policy_engine.evaluate("ds-pol-2", findings, risk)

    assert evaluation["overall_decision"] == PolicyDecision.SANITIZE.value
    assert evaluation["mandatory_blocks"] == 0


def test_policy_allow_clean() -> None:
    """ALLOW decision for no findings."""
    risk_engine = RiskEngine()
    policy_engine = PolicyEngine()

    findings: list[dict] = []
    risk = risk_engine.assess("ds-pol-3", findings)
    evaluation = policy_engine.evaluate("ds-pol-3", findings, risk)

    assert evaluation["overall_decision"] == PolicyDecision.ALLOW.value
    assert evaluation["mandatory_blocks"] == 0


def test_mandatory_block_not_overridable() -> None:
    """Even with a single finding and low average score, a mandatory BLOCK stays BLOCK."""
    risk_engine = RiskEngine()
    policy_engine = PolicyEngine()

    # Single critical secret generates 25 points * 0.30 = 7.5 overall score (normally LOW risk)
    findings = [
        {"category": FindingCategory.PASSWORD.value, "severity": Severity.CRITICAL.value, "confidence": 0.99}
    ]
    risk = risk_engine.assess("ds-pol-4", findings)
    assert risk["overall_score"] < 25.0  # Overall numeric score is low

    evaluation = policy_engine.evaluate("ds-pol-4", findings, risk)
    assert evaluation["overall_decision"] == PolicyDecision.BLOCK.value
    assert evaluation["mandatory_blocks"] >= 1
