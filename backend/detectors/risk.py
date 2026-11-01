from dataclasses import dataclass
from typing import Any
from enum import Enum

from core.database.models.dataset import (
    FindingCategory,
    FindingSeverity,
    FindingConfidence,
)


class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


@dataclass
class RiskAssessment:
    overall_risk: RiskLevel
    risk_score: float
    finding_counts: dict[str, int]
    severity_counts: dict[str, int]
    category_counts: dict[str, int]
    confidence_distribution: dict[str, int]
    detector_coverage: dict[str, bool]
    incomplete_detectors: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_risk": self.overall_risk.value,
            "risk_score": self.risk_score,
            "finding_counts": self.finding_counts,
            "severity_counts": self.severity_counts,
            "category_counts": self.category_counts,
            "confidence_distribution": self.confidence_distribution,
            "detector_coverage": self.detector_coverage,
            "incomplete_detectors": self.incomplete_detectors,
            "recommendations": self.recommendations,
        }


class RiskAssessmentService:
    SEVERITY_WEIGHTS = {
        FindingSeverity.CRITICAL: 100,
        FindingSeverity.HIGH: 50,
        FindingSeverity.MEDIUM: 20,
        FindingSeverity.LOW: 5,
        FindingSeverity.INFORMATIONAL: 1,
    }

    CONFIDENCE_MULTIPLIERS = {
        FindingConfidence.HIGH: 1.0,
        FindingConfidence.MEDIUM: 0.7,
        FindingConfidence.LOW: 0.3,
    }

    CATEGORY_RISK_MODIFIERS = {
        FindingCategory.SECRETS: 1.5,
        FindingCategory.PRIVACY: 1.2,
        FindingCategory.PROMPT_INJECTION: 1.3,
        FindingCategory.DATA_QUALITY: 0.8,
    }

    MAX_RISK_SCORE = 1000

    def __init__(self):
        self.detectors = [
            "privacy",
            "secrets",
            "prompt_injection",
            "data_quality",
        ]

    def assess(self, findings: list[Any], detectors_run: list[str] | None = None) -> RiskAssessment:
        if not findings:
            return RiskAssessment(
                overall_risk=RiskLevel.INFORMATIONAL,
                risk_score=0.0,
                finding_counts={"total": 0},
                severity_counts={},
                category_counts={},
                confidence_distribution={},
                detector_coverage={d: d in (detectors_run or []) for d in self.detectors},
                incomplete_detectors=[d for d in self.detectors if d not in (detectors_run or [])],
                recommendations=["No findings detected. Dataset appears clean based on enabled detectors."],
            )

        finding_counts = {"total": len(findings)}
        severity_counts = {}
        category_counts = {}
        confidence_distribution = {}

        total_weighted_score = 0.0

        for finding in findings:
            severity = finding.severity if isinstance(finding.severity, FindingSeverity) else FindingSeverity(finding.severity)
            category = finding.category if isinstance(finding.category, FindingCategory) else FindingCategory(finding.category)
            confidence = finding.confidence if isinstance(finding.confidence, FindingConfidence) else FindingConfidence(finding.confidence)

            severity_counts[severity.value] = severity_counts.get(severity.value, 0) + 1
            category_counts[category.value] = category_counts.get(category.value, 0) + 1
            confidence_distribution[confidence.value] = confidence_distribution.get(confidence.value, 0) + 1

            base_weight = self.SEVERITY_WEIGHTS.get(severity, 1)
            confidence_mult = self.CONFIDENCE_MULTIPLIERS.get(confidence, 0.5)
            category_mod = self.CATEGORY_RISK_MODIFIERS.get(category, 1.0)

            weighted_score = base_weight * confidence_mult * category_mod
            total_weighted_score += weighted_score

        normalized_score = min(total_weighted_score / self.MAX_RISK_SCORE * 100, 100)

        if normalized_score >= 80:
            overall_risk = RiskLevel.CRITICAL
        elif normalized_score >= 60:
            overall_risk = RiskLevel.HIGH
        elif normalized_score >= 40:
            overall_risk = RiskLevel.MEDIUM
        elif normalized_score >= 20:
            overall_risk = RiskLevel.LOW
        else:
            overall_risk = RiskLevel.INFORMATIONAL

        detectors_run = detectors_run or []
        detector_coverage = {d: d in detectors_run for d in self.detectors}
        incomplete_detectors = [d for d in self.detectors if d not in detectors_run]

        recommendations = self._generate_recommendations(findings, overall_risk, incomplete_detectors)

        return RiskAssessment(
            overall_risk=overall_risk,
            risk_score=normalized_score,
            finding_counts=finding_counts,
            severity_counts=severity_counts,
            category_counts=category_counts,
            confidence_distribution=confidence_distribution,
            detector_coverage=detector_coverage,
            incomplete_detectors=incomplete_detectors,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        findings: list[Any],
        overall_risk: RiskLevel,
        incomplete_detectors: list[str],
    ) -> list[str]:
        recommendations = []

        if overall_risk in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            recommendations.append(
                "HIGH PRIORITY: Dataset contains critical/high severity findings. "
                "Review and remediate before using for training or sharing."
            )

        secrets_findings = [f for f in findings if f.category == FindingCategory.SECRETS]
        if secrets_findings:
            recommendations.append(
                f"SECRETS: {len(secrets_findings)} secret(s) detected. "
                "Immediately rotate exposed credentials and remove from dataset."
            )

        privacy_findings = [f for f in findings if f.category == FindingCategory.PRIVACY]
        if privacy_findings:
            recommendations.append(
                f"PRIVACY: {len(privacy_findings)} privacy finding(s). "
                "Apply pseudonymization or remove PII before use."
            )

        injection_findings = [f for f in findings if f.category == FindingCategory.PROMPT_INJECTION]
        if injection_findings:
            recommendations.append(
                f"PROMPT INJECTION: {len(injection_findings)} indicator(s) found. "
                "Sanitize input data before model training."
            )

        quality_findings = [f for f in findings if f.category == FindingCategory.DATA_QUALITY]
        if quality_findings:
            recommendations.append(
                f"DATA QUALITY: {len(quality_findings)} quality issue(s). "
                "Review data pipeline for systematic issues."
            )

        if incomplete_detectors:
            recommendations.append(
                f"COVERAGE: Detectors not run: {', '.join(incomplete_detectors)}. "
                "Risk assessment may be incomplete."
            )

        low_confidence = [f for f in findings if f.confidence == FindingConfidence.LOW]
        if len(low_confidence) > len(findings) * 0.5:
            recommendations.append(
                "Many findings have low confidence. Manual review recommended."
            )

        if not recommendations:
            recommendations.append("Dataset appears clean based on current scan configuration.")

        return recommendations