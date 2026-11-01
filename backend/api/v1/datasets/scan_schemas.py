from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field

from core.database.models.dataset import ScanStatus


class ScanCreateRequest(BaseModel):
    detectors: Optional[List[str]] = None
    version_id: Optional[str] = None


class ScanResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    version_id: str
    status: ScanStatus
    detectors_enabled: List[str]
    detectors_completed: List[str]
    finding_counts: dict
    risk_assessment: dict
    error_summary: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ScanListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[ScanResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class FindingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    scan_id: str
    dataset_id: str
    version_id: str
    category: str
    rule_id: str
    severity: str
    confidence: str
    title: str
    description: str
    row_number: Optional[int] = None
    column_name: Optional[str] = None
    evidence: Optional[str] = None
    masked_evidence: Optional[str] = None
    remediation_guidance: Optional[str] = None
    detector_name: str
    detector_version: str
    created_at: datetime


class FindingListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[FindingResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ScanReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    scan: ScanResponse
    findings_by_category: dict[str, List[dict]]
    total_findings: int


class RetryScanResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    scan: ScanResponse