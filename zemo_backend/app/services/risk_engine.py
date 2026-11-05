"""
Risk assessment and policy evaluation engines.

RiskEngine:
- Calculates risk scores across privacy, secrets, security, and quality categories
- Computes overall composite risk score and maps to Severity

PolicyEngine:
- Evaluates dataset safety against organizational policies
- Mandatory BLOCK rules for critical secrets, high-confidence critical PII, and prompt injections
- Strict escalation hierarchy: ALLOW < REVIEW < SANITIZE < BLOCK
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.models.enums import FindingCategory, PolicyDecision, Severity, utcnow
from app.schemas.risk import (
    PolicyEvaluation,
    PolicyRule,
    RiskAssessment,
    RiskBreakdown,
)

logger = get_logger("risk_engine")

# Weightings for finding severities
SEVERITY_POINTS = {
    Severity.CRITICAL.value: 25.0,
    Severity.HIGH.value: 15.0,
    Severity.MEDIUM.value: 8.0,
    Severity.LOW.value: 3.0,
    Severity.INFORMATIONAL.value: 1.0,
}

# Quality findings use lower weights
QUALITY_SEVERITY_POINTS = {
    Severity.CRITICAL.value: 15.0,
    Severity.HIGH.value: 10.0,
    Severity.MEDIUM.value: 5.0,
    Severity.LOW.value: 2.0,
    Severity.INFORMATIONAL.value: 0.5,
}

# Categorization sets
PRIVACY_CATEGORIES = {
    FindingCategory.EMAIL.value,
    FindingCategory.PHONE.value,
    FindingCategory.ADDRESS.value,
    FindingCategory.PERSONAL_ID.value,
    FindingCategory.CREDIT_CARD.value,
    FindingCategory.FINANCIAL.value,
}

SECRET_CATEGORIES = {
    FindingCategory.API_KEY.value,
    FindingCategory.PASSWORD.value,
    FindingCategory.ACCESS_TOKEN.value,
    FindingCategory.PRIVATE_KEY.value,
    FindingCategory.DB_CREDENTIAL.value,
}

SECURITY_CATEGORIES = {
    FindingCategory.PROMPT_INJECTION.value,
    FindingCategory.SUSPICIOUS_CONTENT.value,
    FindingCategory.MALFORMED_RECORD.value,
}

QUALITY_CATEGORIES = {
    FindingCategory.MISSING_VALUE.value,
    FindingCategory.DUPLICATE_RECORD.value,
    FindingCategory.SCHEMA_INCONSISTENCY.value,
    FindingCategory.FORMAT_ERROR.value,
    FindingCategory.ENCODING_ERROR.value,
    FindingCategory.EMPTY_RECORD.value,
    FindingCategory.EXCESSIVE_REPETITION.value,
}

# Decision escalation order
DECISION_RANK = {
    PolicyDecision.ALLOW: 0,
    PolicyDecision.REVIEW: 1,
    PolicyDecision.SANITIZE: 2,
    PolicyDecision.BLOCK: 3,
}


class RiskEngine:
    """Calculates granular risk scores per category and composite risk level."""

    def assess(self, dataset_id: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Assess risk across findings.
        Returns a dictionary representation conforming to the RiskAssessment schema.
        """
        privacy_points = 0.0
        secret_points = 0.0
        security_points = 0.0
        quality_points = 0.0

        for f in findings:
            cat = f.get("category", "")
            sev = f.get("severity", Severity.LOW.value)

            if cat in PRIVACY_CATEGORIES:
                privacy_points += SEVERITY_POINTS.get(sev, 0.0)
            elif cat in SECRET_CATEGORIES:
                secret_points += SEVERITY_POINTS.get(sev, 0.0)
            elif cat in SECURITY_CATEGORIES:
                security_points += SEVERITY_POINTS.get(sev, 0.0)
            elif cat in QUALITY_CATEGORIES:
                quality_points += QUALITY_SEVERITY_POINTS.get(sev, 0.0)
            else:
                quality_points += QUALITY_SEVERITY_POINTS.get(sev, 0.0)

        # Cap individual category scores at 100.0
        privacy_score = min(100.0, privacy_points)
        secret_score = min(100.0, secret_points)
        security_score = min(100.0, security_points)
        quality_score = min(100.0, quality_points)

        # Weighted composite score: Privacy 30%, Secrets 30%, Security 25%, Quality 15%
        overall_score = round(
            (privacy_score * 0.30)
            + (secret_score * 0.30)
            + (security_score * 0.25)
            + (quality_score * 0.15),
            2,
        )

        # Determine risk level based on overall score thresholds
        if overall_score >= 75.0:
            risk_level = Severity.CRITICAL
        elif overall_score >= 50.0:
            risk_level = Severity.HIGH
        elif overall_score >= 25.0:
            risk_level = Severity.MEDIUM
        else:
            risk_level = Severity.LOW

        now = utcnow()
        assessment = RiskAssessment(
            dataset_id=dataset_id,
            overall_score=overall_score,
            breakdown=RiskBreakdown(
                privacy_score=round(privacy_score, 2),
                secret_score=round(secret_score, 2),
                security_score=round(security_score, 2),
                quality_score=round(quality_score, 2),
            ),
            risk_level=risk_level,
            assessed_at=now,
        )

        logger.info(
            "Risk assessment for %s: overall=%.2f, level=%s (priv=%.1f, sec=%.1f, sct=%.1f, qual=%.1f)",
            dataset_id,
            overall_score,
            risk_level.value,
            privacy_score,
            security_score,
            secret_score,
            quality_score,
        )

        return assessment.model_dump(mode="json")


class PolicyEngine:
    """Evaluates mandatory and organizational rules to determine policy decisions."""

    def evaluate(
        self,
        dataset_id: str,
        findings: list[dict[str, Any]],
        risk: dict[str, Any] | RiskAssessment,
    ) -> dict[str, Any]:
        """
        Evaluate dataset against safety policies.
        Mandatory rules:
        1. BLOCK if any CRITICAL secret found.
        2. BLOCK if any CRITICAL PII with confidence > 0.9.
        3. BLOCK if prompt injection count > 5.
        4. SANITIZE if privacy findings exist with severity >= MEDIUM.
        5. REVIEW if risk overall_score > 50.
        6. ALLOW if no findings or all INFORMATIONAL.

        Mandatory BLOCK can never be overridden. Escalation order: ALLOW < REVIEW < SANITIZE < BLOCK.
        """
        overall_score = (
            risk.overall_score
            if isinstance(risk, RiskAssessment)
            else float(risk.get("overall_score", 0.0))
        )

        triggered_rules: list[PolicyRule] = []
        mandatory_blocks = 0
        current_decision = PolicyDecision.ALLOW
        reasons: list[str] = []

        def escalate(rule: PolicyRule, reason: str) -> None:
            nonlocal current_decision
            triggered_rules.append(rule)
            reasons.append(reason)
            if DECISION_RANK[rule.decision] > DECISION_RANK[current_decision]:
                current_decision = rule.decision

        # Count prompt injections & inspect findings
        prompt_injection_count = 0
        has_critical_secret = False
        has_critical_pii_high_conf = False
        has_privacy_medium_plus = False
        non_info_finding_count = 0

        for f in findings:
            cat = f.get("category", "")
            sev = f.get("severity", Severity.LOW.value)
            conf = float(f.get("confidence", 0.0))

            if sev != Severity.INFORMATIONAL.value:
                non_info_finding_count += 1

            if cat == FindingCategory.PROMPT_INJECTION.value:
                prompt_injection_count += 1

            if cat in SECRET_CATEGORIES and sev == Severity.CRITICAL.value:
                has_critical_secret = True

            if cat in PRIVACY_CATEGORIES:
                if sev == Severity.CRITICAL.value and conf > 0.9:
                    has_critical_pii_high_conf = True
                if sev in (Severity.CRITICAL.value, Severity.HIGH.value, Severity.MEDIUM.value):
                    has_privacy_medium_plus = True

        # Rule 1: CRITICAL secret found -> BLOCK (Mandatory)
        if has_critical_secret:
            mandatory_blocks += 1
            escalate(
                PolicyRule(
                    rule_id="RULE-SEC-001",
                    name="Block Critical Secrets",
                    description="Immediate mandatory block when any critical secret or credential is found.",
                    category="secrets",
                    severity_trigger=Severity.CRITICAL,
                    decision=PolicyDecision.BLOCK,
                    mandatory=True,
                    enabled=True,
                ),
                "Critical secrets or credentials detected in dataset.",
            )

        # Rule 2: CRITICAL PII with confidence > 0.9 -> BLOCK (Mandatory)
        if has_critical_pii_high_conf:
            mandatory_blocks += 1
            escalate(
                PolicyRule(
                    rule_id="RULE-PRIV-001",
                    name="Block High Confidence Critical PII",
                    description="Mandatory block when high confidence critical PII (SSN, payment cards) is detected.",
                    category="privacy",
                    severity_trigger=Severity.CRITICAL,
                    decision=PolicyDecision.BLOCK,
                    mandatory=True,
                    enabled=True,
                ),
                "Critical PII with >90% confidence detected in dataset.",
            )

        # Rule 3: Prompt injection count > 5 -> BLOCK (Mandatory)
        if prompt_injection_count > 5:
            mandatory_blocks += 1
            escalate(
                PolicyRule(
                    rule_id="RULE-SEC-002",
                    name="Block Excessive Prompt Injections",
                    description="Mandatory block when dataset contains more than 5 prompt injection patterns.",
                    category="security",
                    severity_trigger=Severity.HIGH,
                    decision=PolicyDecision.BLOCK,
                    mandatory=True,
                    enabled=True,
                ),
                f"Excessive prompt injection patterns detected ({prompt_injection_count} occurrences).",
            )

        # Rule 4: Privacy findings with severity >= MEDIUM -> SANITIZE
        if has_privacy_medium_plus:
            escalate(
                PolicyRule(
                    rule_id="RULE-PRIV-002",
                    name="Sanitize Privacy Findings",
                    description="Require sanitization when personal information of medium or higher severity is discovered.",
                    category="privacy",
                    severity_trigger=Severity.MEDIUM,
                    decision=PolicyDecision.SANITIZE,
                    mandatory=False,
                    enabled=True,
                ),
                "Privacy-sensitive information found that requires data masking or redaction.",
            )

        # Rule 5: Overall risk score > 50 -> REVIEW
        if overall_score > 50.0:
            escalate(
                PolicyRule(
                    rule_id="RULE-RISK-001",
                    name="Review Elevated Risk",
                    description="Flag dataset for human review when overall composite risk score exceeds 50.",
                    category="risk",
                    severity_trigger=Severity.HIGH,
                    decision=PolicyDecision.REVIEW,
                    mandatory=False,
                    enabled=True,
                ),
                f"Overall risk score of {overall_score:.1f} exceeds the threshold of 50.0.",
            )

        # Rule 6: If no findings or all informational -> ALLOW
        if not findings or non_info_finding_count == 0:
            current_decision = PolicyDecision.ALLOW
            reasons = ["No actionable safety or quality findings discovered. Dataset is safe."]

        # CRITICAL RULE: Mandatory block can NEVER be overridden
        if mandatory_blocks > 0:
            current_decision = PolicyDecision.BLOCK

        explanation = " | ".join(reasons) if reasons else "Policy evaluated successfully."

        evaluation = PolicyEvaluation(
            dataset_id=dataset_id,
            overall_decision=current_decision,
            triggered_rules=triggered_rules,
            mandatory_blocks=mandatory_blocks,
            explanation=explanation,
            evaluated_at=utcnow(),
        )

        logger.info(
            "Policy evaluation for %s: decision=%s, mandatory_blocks=%d, triggered_rules=%d",
            dataset_id,
            current_decision.value,
            mandatory_blocks,
            len(triggered_rules),
        )

        return evaluation.model_dump(mode="json")
