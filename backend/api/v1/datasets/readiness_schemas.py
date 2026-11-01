from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field


class ReadinessAssessmentCreate(BaseModel):
    dataset_id: str
    intended_use: str = Field(..., min_length=1, max_length=50)
    version_id: Optional[str] = None
    scan_id: Optional[str] = None
    evaluation_id: Optional[str] = None


class ReadinessAssessmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    version_id: str
    scan_id: Optional[str]
    evaluation_id: Optional[str]
    intended_use: str
    status: str
    reason: str
    details: dict
    policy_decision: Optional[str]
    requires_sanitization: bool
    residual_findings: int
    residual_risk: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]


class ReadinessAssessmentListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[dict]
    total: int
    page: int
    page_size: int
    total_pages: int


class ReadinessAssessmentReport(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    assessment: dict
    findings_by_category: dict
    total_findings: int