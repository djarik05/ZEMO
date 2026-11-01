from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.sqlite import BLOB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base

if TYPE_CHECKING:
    from core.database.models.dataset_version import DatasetVersion


# =============================================================================
# ENUMS
# =============================================================================

class DatasetStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    INVALID = "INVALID"
    PROFILING = "PROFILING"
    PROFILED = "PROFILED"
    FAILED = "FAILED"


class DatasetFormat(str, enum.Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    XLSX = "xlsx"
    PARQUET = "parquet"


class ValidationStatus(str, enum.Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ProfilingStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ScanStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class FindingCategory(str, enum.Enum):
    PRIVACY = "PRIVACY"
    SECRETS = "SECRETS"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    DATA_QUALITY = "DATA_QUALITY"


class FindingSeverity(str, enum.Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class FindingConfidence(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class PrivacyRuleType(str, enum.Enum):
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    GOVERNMENT_ID = "GOVERNMENT_ID"
    FINANCIAL = "FINANCIAL"
    SENSITIVE_FIELD_NAME = "SENSITIVE_FIELD_NAME"
    PERSONAL_INFO_COMBO = "PERSONAL_INFO_COMBO"


class SecretsRuleType(str, enum.Enum):
    API_KEY = "API_KEY"
    ACCESS_TOKEN = "ACCESS_TOKEN"
    PASSWORD = "PASSWORD"
    PRIVATE_KEY = "PRIVATE_KEY"
    CLOUD_CREDENTIAL = "CLOUD_CREDENTIAL"
    DATABASE_CONNECTION_STRING = "DATABASE_CONNECTION_STRING"
    HIGH_ENTROPY_STRING = "HIGH_ENTROPY_STRING"


class PromptInjectionRuleType(str, enum.Enum):
    OVERRIDE_INSTRUCTION = "OVERRIDE_INSTRUCTION"
    REVEAL_PROMPT = "REVEAL_PROMPT"
    BYPASS_SAFETY = "BYPASS_SAFETY"
    REDIRECT_TASK = "REDIRECT_TASK"
    TOOL_EXFILTRATION = "TOOL_EXFILTRATION"
    CONFLICTING_INSTRUCTION = "CONFLICTING_INSTRUCTION"


class DataQualityRuleType(str, enum.Enum):
    MISSING_VALUES = "MISSING_VALUES"
    DUPLICATE_RECORDS = "DUPLICATE_RECORDS"
    UNEXPECTED_DATA_TYPE = "UNEXPECTED_DATA_TYPE"
    STRUCTURAL_INCONSISTENCY = "STRUCTURAL_INCONSISTENCY"
    MALFORMED_RECORDS = "MALFORMED_RECORDS"
    REPETITIVE_VALUES = "REPETITIVE_VALUES"
    SCHEMA_INCONSISTENCY = "SCHEMA_INCONSISTENCY"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


# =============================================================================
# DATASET MODELS (Original Phase 2 models)
# =============================================================================

class DatasetFormat(str, enum.Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"


class ValidationStatus(str, enum.Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ProfilingStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[DatasetFormat] = mapped_column(Enum(DatasetFormat), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    current_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[DatasetStatus] = mapped_column(Enum(DatasetStatus), default=DatasetStatus.UPLOADED, nullable=False, index=True)
    validation_status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    profiling_status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    record_count: Mapped["int | None"] = mapped_column(Integer, nullable=True)
    column_count: Mapped["int | None"] = mapped_column(Integer, nullable=True)
    schema_metadata: Mapped["dict | None"] = mapped_column(JSON, nullable=True)
    error_summary: Mapped["str | None"] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    versions: Mapped[list["DatasetVersion"]] = relationship(
        "DatasetVersion",
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="DatasetVersion.version_number",
    )

    __table_args__ = (
        Index("ix_datasets_created_at", "created_at"),
        Index("ix_datasets_status_created_at", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name={self.name}, format={self.format}, status={self.status})>"


class DatasetProfile(Base):
    __tablename__ = "dataset_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    columns: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    schema: Mapped[dict] = mapped_column(JSON, nullable=False)
    statistics: Mapped[dict] = mapped_column(JSON, nullable=False)
    duplicate_records: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    empty_columns: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default=ProfilingStatus.PENDING, nullable=False)
    error_summary: Mapped["str | None"] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<DatasetProfile(id={self.id}, dataset_id={self.dataset_id}, version_id={self.version_id}, records={self.record_count})>"


class DatasetValidationResult(Base):
    __tablename__ = "dataset_validation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), default=ValidationStatus.PENDING, nullable=False)
    errors: Mapped[list[dict]] = mapped_column(JSON, default=list, nullable=False)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    valid_records: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    invalid_records: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_records: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    partial_ingestion: Mapped[bool] = mapped_column(default=False, nullable=False)
    error_summary: Mapped["str | None"] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<DatasetValidationResult(id={self.id}, dataset_id={self.dataset_id}, version_id={self.version_id}, status={self.status})>"


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    status: Mapped[ScanStatus] = mapped_column(Enum(ScanStatus), default=ScanStatus.QUEUED, nullable=False, index=True)
    detectors_enabled: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    detectors_completed: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    finding_counts: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    risk_assessment: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_summary: Mapped["str | None"] = mapped_column(Text, nullable=True)
    started_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    findings: Mapped[list["Finding"]] = relationship(
        "Finding",
        back_populates="scan",
        cascade="all, delete-orphan",
        order_by="Finding.created_at",
    )

    __table_args__ = (
        Index("ix_scans_dataset_id_created_at", "dataset_id", "created_at"),
        Index("ix_scans_status_created_at", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Scan(id={self.id}, dataset_id={self.dataset_id}, version_id={self.version_id}, status={self.status})>"


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_id: Mapped[str] = mapped_column(String(36), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    category: Mapped[FindingCategory] = mapped_column(Enum(FindingCategory), nullable=False, index=True)
    rule_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    severity: Mapped[FindingSeverity] = mapped_column(Enum(FindingSeverity), nullable=False, index=True)
    confidence: Mapped[FindingConfidence] = mapped_column(Enum(FindingConfidence), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    row_number: Mapped["int | None"] = mapped_column(Integer, nullable=True)
    column_name: Mapped["str | None"] = mapped_column(String(255), nullable=True, index=True)
    evidence: Mapped["str | None"] = mapped_column(Text, nullable=True)
    masked_evidence: Mapped["str | None"] = mapped_column(Text, nullable=True)
    remediation_guidance: Mapped["str | None"] = mapped_column(Text, nullable=True)
    detector_name: Mapped[str] = mapped_column(String(100), nullable=False)
    detector_version: Mapped[str] = mapped_column(String(50), default="1.0.0", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    scan: Mapped["Scan"] = relationship("Scan", back_populates="findings")

    __table_args__ = (
        Index("ix_findings_scan_id_created_at", "scan_id", "created_at"),
        Index("ix_findings_category_severity", "category", "severity"),
        Index("ix_findings_dataset_id_created_at", "dataset_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Finding(id={self.id}, scan_id={self.scan_id}, category={self.category}, rule_id={self.rule_id}, severity={self.severity})>"


# =============================================================================
# PHASE 4 ENUMS (Policy, Sanitization, Readiness)
# =============================================================================

class ValidationStatus(str, enum.Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ProfilingStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DatasetFormat(str, enum.Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"


# --- Phase 4 Models (Policy, Sanitization, Readiness) ---

class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


# =============================================================================
# PHASE 4 MODELS (Policy, Sanitization, Readiness)
# =============================================================================

class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


# =============================================================================
# PHASE 4 ENUMS (Policy, Sanitization, Readiness)
# =============================================================================

class ValidationStatus(str, enum.Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ProfilingStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DatasetFormat(str, enum.Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"


# --- Phase 4 Models (Policy, Sanitization, Readiness) ---

class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


# =============================================================================
# PHASE 4 MODELS (Policy, Sanitization, Readiness)
# =============================================================================

class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


# =============================================================================
# PHASE 4 MODELS (Policy, Sanitization, Readiness)
# =============================================================================

class PolicyDecision(Base):
    __tablename__ = "policy_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    policy_id: Mapped[str] = mapped_column(String(36), ForeignKey("policies.id", ondelete="CASCADE"), nullable=False, index=True)
    evaluation_id: Mapped[str] = mapped_column(String(36), ForeignKey("policy_evaluations.id", ondelete="CASCADE"), nullable=False, index=True)
    decision: Mapped[PolicyDecisionType] = mapped_column(Enum(PolicyDecisionType), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<PolicyDecision(id={self.id}, policy_id={self.policy_id}, evaluation_id={self.evaluation_id}, decision={self.decision})>"


class PolicyRule(Base):
    __tablename__ = "policy_rules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    policy_id: Mapped[str] = mapped_column(String(36), ForeignKey("policies.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    category: Mapped[PolicyRuleCategory] = mapped_column(Enum(PolicyRuleCategory), nullable=False)
    condition: Mapped[dict] = mapped_column(JSON, nullable=False)
    action: Mapped[PolicyDecisionType] = mapped_column(Enum(PolicyDecisionType), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<PolicyRule(id={self.id}, policy_id={self.policy_id}, name={self.name}, action={self.action})>"


class Policy(Base):
    __tablename__ = "policies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    intended_uses: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    default_action: Mapped[PolicyDecisionType] = mapped_column(Enum(PolicyDecisionType), default=PolicyDecisionType.REVIEW, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_by: Mapped["str | None"] = mapped_column(String(100), nullable=True)

    rules: Mapped[list["PolicyRule"]] = relationship(
        "PolicyRule",
        backref="policy",
        cascade="all, delete-orphan",
        order_by="PolicyRule.priority.desc()",
    )

    def __repr__(self) -> str:
        return f"<Policy(id={self.id}, name={self.name}, version={self.version}, is_active={self.is_active})>"


class PolicyEvaluation(Base):
    __tablename__ = "policy_evaluations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    policy_id: Mapped[str] = mapped_column(String(36), ForeignKey("policies.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    scan_id: Mapped[str] = mapped_column(String(36), ForeignKey("scans.id", ondelete="CASCADE"), nullable=False, index=True)
    policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    decision: Mapped[PolicyDecisionType] = mapped_column(Enum(PolicyDecisionType), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    matched_rules: Mapped[list[dict]] = mapped_column(JSON, default=list, nullable=False)
    requires_approval: Mapped[bool] = mapped_column(default=False, nullable=False)
    approved_by: Mapped["str | None"] = mapped_column(String(100), nullable=True)
    approved_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)

    decisions: Mapped[list["PolicyDecision"]] = relationship(
        "PolicyDecision",
        backref="evaluation",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<PolicyEvaluation(id={self.id}, policy_id={self.policy_id}, dataset_id={self.dataset_id}, decision={self.decision})>"


class ApprovalRecord(Base):
    __tablename__ = "approval_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    evaluation_id: Mapped[str] = mapped_column(String(36), ForeignKey("policy_evaluations.id", ondelete="CASCADE"), nullable=False, index=True)
    approved_by: Mapped[str] = mapped_column(String(100), nullable=False)
    decision: Mapped[PolicyDecisionType] = mapped_column(Enum(PolicyDecisionType), nullable=False)
    comment: Mapped["str | None"] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<ApprovalRecord(id={self.id}, evaluation_id={self.evaluation_id}, approved_by={self.approved_by}, decision={self.decision})>"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


class SanitizationJob(Base):
    __tablename__ = "sanitization_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    scan_id: Mapped["str | None"] = mapped_column(String(36), ForeignKey("scans.id", ondelete="SET NULL"), nullable=True, index=True)
    evaluation_id: Mapped["str | None"] = mapped_column(String(36), ForeignKey("policy_evaluations.id", ondelete="SET NULL"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(50), default="PENDING", nullable=False, index=True)
    rules_applied: Mapped[list[dict]] = mapped_column(JSON, default=list, nullable=False)
    records_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_removed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    columns_removed: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    values_masked: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    values_redacted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_summary: Mapped["str | None"] = mapped_column(Text, nullable=True)
    output_version_id: Mapped["str | None"] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True, index=True)
    started_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<SanitizationJob(id={self.id}, dataset_id={self.dataset_id}, version_id={self.version_id}, status={self.status})>"


class SanitizationRule(Base):
    __tablename__ = "sanitization_rules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("sanitization_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    rule_type: Mapped[SanitizationRuleType] = mapped_column(Enum(SanitizationRuleType), nullable=False)
    action_type: Mapped[SanitizationActionType] = mapped_column(Enum(SanitizationActionType), nullable=False)
    target_column: Mapped["str | None"] = mapped_column(String(255), nullable=True)
    target_finding_rule_id: Mapped["str | None"] = mapped_column(String(100), nullable=True)
    parameters: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    records_affected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<SanitizationRule(id={self.id}, job_id={self.job_id}, type={self.rule_type}, action={self.action_type})>"


class SanitizationResult(Base):
    __tablename__ = "sanitization_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("sanitization_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    rule_id: Mapped[str] = mapped_column(String(36), ForeignKey("sanitization_rules.id", ondelete="CASCADE"), nullable=False, index=True)
    records_affected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<SanitizationResult(id={self.id}, job_id={self.job_id}, rule_id={self.rule_id}, records_affected={self.records_affected})>"


class ReadinessAssessment(Base):
    __tablename__ = "readiness_assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id: Mapped[str] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    scan_id: Mapped["str | None"] = mapped_column(String(36), ForeignKey("scans.id", ondelete="SET NULL"), nullable=True, index=True)
    evaluation_id: Mapped["str | None"] = mapped_column(String(36), ForeignKey("policy_evaluations.id", ondelete="SET NULL"), nullable=True, index=True)
    intended_use: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    policy_decision: Mapped["PolicyDecisionType | None"] = mapped_column(Enum(PolicyDecisionType), nullable=True)
    requires_sanitization: Mapped[bool] = mapped_column(default=False, nullable=False)
    residual_findings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    residual_risk: Mapped["str | None"] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at: Mapped["datetime | None"] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<ReadinessAssessment(id={self.id}, dataset_id={self.dataset_id}, intended_use={self.intended_use}, status={self.status})>"


# =============================================================================
# DATASET MODELS (Original Phase 2 models - DEFINITIVE DEFINITIONS)
# =============================================================================

class DatasetFormat(str, enum.Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"


class ValidationStatus(str, enum.Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ProfilingStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Phase 4 Models (Policy, Sanitization, Readiness) ---

class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"


class PolicyDecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class SanitizationActionType(str, enum.Enum):
    MASK = "MASK"
    REDACT = "REDACT"
    REMOVE_COLUMN = "REMOVE_COLUMN"
    REMOVE_ROW = "REMOVE_ROW"
    QUARANTINE = "QUARANTINE"


class SanitizationRuleType(str, enum.Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    SECRETS_REDACT = "SECRETS_REDACT"
    PROMPT_INJECTION_QUARANTINE = "PROMPT_INJECTION_QUARANTINE"
    DATA_QUALITY_REMOVE_DUPLICATES = "DATA_QUALITY_REMOVE_DUPLICATES"
    DATA_QUALITY_REMOVE_COLUMN = "DATA_QUALITY_REMOVE_COLUMN"
    CUSTOM_REDACT = "CUSTOM_REDACT"