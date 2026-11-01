from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from core.database.models.dataset import (
    FindingCategory,
    FindingSeverity,
    FindingConfidence,
)


class DetectorName(str, Enum):
    PRIVACY = "privacy"
    SECRETS = "secrets"
    PROMPT_INJECTION = "prompt_injection"
    DATA_QUALITY = "data_quality"


class DetectorName(str, Enum):
    PRIVACY = "privacy"
    SECRETS = "secrets"
    PROMPT_INJECTION = "prompt_injection"
    DATA_QUALITY = "data_quality"


@dataclass
class DetectionContext:
    """Context passed to detectors during scanning."""
    dataset_id: str
    version_id: str
    records: list[dict[str, Any]]
    columns: list[str]
    schema_metadata: dict[str, Any] | None = None
    profile_metadata: dict[str, Any] | None = None
    file_format: str = ""
    max_records: int = 1000000
    max_columns: int = 1000


@dataclass
class Finding:
    """Normalized finding from a detector."""
    category: FindingCategory
    rule_id: str
    severity: FindingSeverity
    confidence: FindingConfidence
    title: str
    description: str
    row_number: int | None = None
    column_name: str | None = None
    evidence: str | None = None
    masked_evidence: str | None = None
    remediation_guidance: str | None = None
    detector_name: str = ""
    detector_version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "title": self.title,
            "description": self.description,
            "row_number": self.row_number,
            "column_name": self.column_name,
            "evidence": self.evidence,
            "masked_evidence": self.masked_evidence,
            "remediation_guidance": self.remediation_guidance,
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "metadata": self.metadata,
        }


class BaseDetector(ABC):
    """Base class for all detectors."""

    name: str = ""
    version: str = "1.0.0"
    category: FindingCategory = FindingCategory.DATA_QUALITY

    @abstractmethod
    def detect(self, context: DetectionContext) -> list[Finding]:
        """Run detection on the given context and return findings."""
        pass

    def _create_finding(
        self,
        rule_id: str,
        severity: FindingSeverity,
        confidence: FindingConfidence,
        title: str,
        description: str,
        row_number: int | None = None,
        column_name: str | None = None,
        evidence: str | None = None,
        masked_evidence: str | None = None,
        remediation_guidance: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Finding:
        return Finding(
            category=self.category,
            rule_id=rule_id,
            severity=severity,
            confidence=confidence,
            title=title,
            description=description,
            row_number=row_number,
            column_name=column_name,
            evidence=evidence,
            masked_evidence=masked_evidence,
            remediation_guidance=remediation_guidance,
            detector_name=self.name,
            detector_version=self.version,
            metadata=metadata or {},
        )

    def _mask_sensitive_value(self, value: str, visible_chars: int = 4) -> str:
        """Mask a sensitive value for safe logging/storage."""
        if len(value) <= visible_chars * 2:
            return "*" * len(value)
        return value[:visible_chars] + "*" * (len(value) - visible_chars * 2) + value[-visible_chars:]

    def _mask_email(self, email: str) -> str:
        """Mask email address for safe storage."""
        if "@" not in email:
            return self._mask_sensitive_value(email)
        local, domain = email.split("@", 1)
        if len(local) <= 2:
            masked_local = "*" * len(local)
        else:
            masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
        return f"{masked_local}@{domain}"

    def _mask_phone(self, phone: str) -> str:
        """Mask phone number for safe storage."""
        digits = "".join(c for c in phone if c.isdigit())
        if len(digits) <= 4:
            return "*" * len(phone)
        masked = "*" * (len(digits) - 4) + digits[-4:]
        result = ""
        digit_idx = 0
        for c in phone:
            if c.isdigit():
                result += masked[digit_idx]
                digit_idx += 1
            else:
                result += c
        return result