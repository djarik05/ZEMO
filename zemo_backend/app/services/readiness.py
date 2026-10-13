"""
Training readiness assessment service.

Evaluates whether a dataset meets privacy, security, and quality criteria
for AI training, fine-tuning, evaluation, or inference.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import DatasetNotFoundError
from app.core.logging import get_logger
from app.models.enums import (
    FindingCategory,
    IntendedOperation,
    PolicyDecision,
    ReadinessStatus,
    Severity,
    utcnow,
)
from app.repositories.dataset_repo import DatasetRepository
from app.schemas.readiness import ReadinessCheck, ReadinessReport
from app.services.risk_engine import PolicyEngine, RiskEngine

logger = get_logger("readiness")


class ReadinessService:
    """Service to evaluate dataset readiness for AI model lifecycle operations."""

    def __init__(self, repo: DatasetRepository | None = None) -> None:
        self.repo = repo or DatasetRepository()
        self.risk_engine = RiskEngine()
        self.policy_engine = PolicyEngine()

    def assess(
        self,
        dataset_id: str,
        intended_operation: IntendedOperation = IntendedOperation.TRAINING,
    ) -> ReadinessReport:
        """
        Assess dataset training readiness.
        - Checks for critical secrets (blocks)
        - Checks for unresolved PII
        - Checks for prompt injection & security risks
        - Computes weighted overall readiness score
        """
        if not self.repo.exists(dataset_id):
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' not found.",
                detail={"dataset_id": dataset_id},
            )

        findings = self.repo.get_findings(dataset_id)
        risk = self.repo.get_risk(dataset_id)
        if not risk:
            risk = self.risk_engine.assess(dataset_id, findings)

        policy = self.repo.get_policy(dataset_id)
        if not policy:
            policy = self.policy_engine.evaluate(dataset_id, findings, risk)

        checks: list[ReadinessCheck] = []
        blocking_issues: list[str] = []
        warnings: list[str] = []
        recommendations: list[str] = []

        # Analyze findings
        critical_secrets = [
            f for f in findings
            if f.get("severity") == Severity.CRITICAL.value
            and f.get("category") in {
                FindingCategory.API_KEY.value,
                FindingCategory.PASSWORD.value,
                FindingCategory.ACCESS_TOKEN.value,
                FindingCategory.PRIVATE_KEY.value,
                FindingCategory.DB_CREDENTIAL.value,
            }
        ]
        critical_pii = [
            f for f in findings
            if f.get("severity") == Severity.CRITICAL.value
            and f.get("category") in {
                FindingCategory.CREDIT_CARD.value,
                FindingCategory.PERSONAL_ID.value,
            }
        ]
        prompt_injections = [
            f for f in findings
            if f.get("category") == FindingCategory.PROMPT_INJECTION.value
        ]
        other_pii = [
            f for f in findings
            if f.get("category") in {
                FindingCategory.EMAIL.value,
                FindingCategory.PHONE.value,
                FindingCategory.ADDRESS.value,
            }
        ]
        exploits = [
            f for f in findings
            if f.get("category") in {
                FindingCategory.SUSPICIOUS_CONTENT.value,
                FindingCategory.MALFORMED_RECORD.value,
            }
        ]

        # Check 1: No Critical Secrets (Weight 2.0)
        c1_pass = len(critical_secrets) == 0
        if not c1_pass:
            blocking_issues.append(f"Found {len(critical_secrets)} critical secrets/credentials.")
            recommendations.append("Redact all exposed API keys, passwords, and tokens before use.")
        checks.append(ReadinessCheck(
            check_name="No Critical Secrets",
            passed=c1_pass,
            message="No critical credentials or secrets detected." if c1_pass else f"{len(critical_secrets)} critical secrets detected.",
            weight=2.0,
        ))

        # Check 2: No Critical PII (Weight 2.0)
        c2_pass = len(critical_pii) == 0
        if not c2_pass:
            blocking_issues.append(f"Found {len(critical_pii)} critical PII instances (cards/SSNs).")
            recommendations.append("Apply sanitization to mask credit cards and national IDs.")
        checks.append(ReadinessCheck(
            check_name="No Critical PII",
            passed=c2_pass,
            message="No unmasked critical PII detected." if c2_pass else f"{len(critical_pii)} critical PII items detected.",
            weight=2.0,
        ))

        # Check 3: No Prompt Injections (Weight 1.5)
        c3_pass = len(prompt_injections) == 0
        if not c3_pass:
            warnings.append(f"Detected {len(prompt_injections)} potential prompt injection patterns.")
            recommendations.append("Remove records with prompt injections to prevent model jailbreak training.")
        checks.append(ReadinessCheck(
            check_name="Prompt Injection Safety",
            passed=c3_pass,
            message="Zero prompt injection triggers found." if c3_pass else f"{len(prompt_injections)} injection patterns found.",
            weight=1.5,
        ))

        # Check 4: General Privacy Compliance (Weight 1.5)
        c4_pass = len(other_pii) == 0
        if not c4_pass:
            warnings.append(f"Dataset contains {len(other_pii)} privacy findings (emails, phones, addresses).")
            recommendations.append("Apply pseudonymization or masking to all personal data.")
        checks.append(ReadinessCheck(
            check_name="Privacy Protection",
            passed=c4_pass,
            message="Dataset is clean of common PII." if c4_pass else f"{len(other_pii)} personal identifiers found.",
            weight=1.5,
        ))

        # Check 5: Exploit and Integrity Safety (Weight 1.0)
        c5_pass = len(exploits) == 0
        if not c5_pass:
            warnings.append(f"Dataset contains {len(exploits)} exploit payloads or malformed records.")
        checks.append(ReadinessCheck(
            check_name="Payload Safety",
            passed=c5_pass,
            message="No malicious payloads detected." if c5_pass else f"{len(exploits)} suspicious payloads detected.",
            weight=1.0,
        ))

        # Calculate weighted score
        total_weight = sum(c.weight for c in checks)
        passed_weight = sum(c.weight for c in checks if c.passed)
        overall_score = round((passed_weight / total_weight) * 100.0, 2) if total_weight > 0 else 100.0

        # Determine overall readiness status
        if blocking_issues or policy.get("overall_decision") == PolicyDecision.BLOCK.value or len(critical_secrets) > 0:
            overall_status = ReadinessStatus.BLOCKED
            msg = "Dataset is BLOCKED from AI training due to critical security or privacy violations."
        elif not findings or overall_score >= 99.9:
            overall_status = ReadinessStatus.READY
            msg = "Dataset is READY for intended AI model operations."
        else:
            overall_status = ReadinessStatus.NOT_READY
            msg = "Dataset is NOT READY. Remediation or sanitization required."

        report = ReadinessReport(
            dataset_id=dataset_id,
            intended_operation=intended_operation,
            overall_status=overall_status,
            overall_score=overall_score,
            checks=checks,
            blocking_issues=blocking_issues,
            warnings=warnings,
            recommendations=recommendations,
            assessed_at=utcnow(),
            message=msg,
        )

        self.repo.save_readiness(dataset_id, report.model_dump(mode="json"))
        logger.info(
            "Readiness for %s: status=%s, score=%.1f",
            dataset_id,
            overall_status.value,
            overall_score,
        )

        return report
