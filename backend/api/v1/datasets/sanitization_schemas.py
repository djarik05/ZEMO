import enum
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field


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


class SanitizationRuleCreate(BaseModel):
    rule_type: str = Field(..., pattern="^(PRIVACY_MASK|SECRETS_REDACT|PROMPT_INJECTION_QUARANTINE|DATA_QUALITY_REMOVE_DUPLICATES|DATA_QUALITY_REMOVE_COLUMN|CUSTOM_REDACT)$")
    action_type: str = Field(..., pattern="^(MASK|REDACT|REMOVE_COLUMN|REMOVE_ROW|QUARANTINE)$")
    target_column: Optional[str] = None
    target_finding_rule_id: Optional[str] = None
    parameters: dict = Field(default_factory=dict)


class SanitizationRuleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    rule_type: str
    action_type: str
    target_column: Optional[str]
    target_finding_rule_id: Optional[str]
    parameters: dict
    records_affected: int
    created_at: datetime


class SanitizationJobCreate(BaseModel):
    dataset_id: str
    version_id: Optional[str] = None
    scan_id: Optional[str] = None
    evaluation_id: Optional[str] = None
    rules: List[SanitizationRuleCreate] = Field(default_factory=list)


class SanitizationJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    version_id: str
    scan_id: Optional[str]
    evaluation_id: Optional[str]
    status: str
    rules_applied: List[dict]
    records_processed: int
    records_removed: int
    columns_removed: List[str]
    values_masked: int
    values_redacted: int
    error_summary: Optional[str]
    output_version_id: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime


class SanitizationJobListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[dict]
    total: int
    page: int
    page_size: int
    total_pages: int


class SanitizationJobStatusResponse(BaseModel):
    job: dict
    rules: List[dict]
    results: List[dict]


class SanitizationJobReportResponse(BaseModel):
    job: dict
    rules: List[dict]
    results: List[dict]