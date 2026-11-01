import enum
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field
from core.database.models.dataset import PolicyDecisionType


class PolicyRuleCategory(str, enum.Enum):
    FINDING_BASED = "FINDING_BASED"
    SCAN_BASED = "SCAN_BASED"
    DATASET_BASED = "DATASET_BASED"


class PolicyRuleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: PolicyRuleCategory
    condition: dict
    action: str = Field(..., pattern="^(ALLOW|REVIEW|SANITIZE|BLOCK)$")
    priority: int = Field(default=0, ge=0)


class PolicyRuleUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    condition: Optional[dict] = None
    action: Optional[str] = Field(None, pattern="^(ALLOW|REVIEW|SANITIZE|BLOCK)$")
    priority: Optional[int] = Field(None, ge=0)
    is_active: Optional[bool] = None


class PolicyRuleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    policy_id: str
    name: str
    description: Optional[str]
    category: str
    condition: dict
    action: str
    priority: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class PolicyRuleListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[PolicyRuleResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class PolicyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    intended_uses: List[str] = Field(default_factory=list)
    default_action: str = Field(default="REVIEW", pattern="^(ALLOW|REVIEW|SANITIZE|BLOCK)$")
    created_by: Optional[str] = None


class PolicyUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    intended_uses: Optional[List[str]] = None
    default_action: Optional[str] = Field(None, pattern="^(ALLOW|REVIEW|SANITIZE|BLOCK)$")
    is_active: Optional[bool] = None


class PolicyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: Optional[str]
    version: int
    is_active: bool
    intended_uses: List[str]
    default_action: str
    created_by: Optional[str]
    created_at: datetime
    updated_at: datetime


class PolicyListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[PolicyResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class PolicyEvaluationCreate(BaseModel):
    policy_id: str
    dataset_id: str
    version_id: Optional[str] = None
    scan_id: Optional[str] = None


class PolicyEvaluationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    policy_id: str
    dataset_id: str
    version_id: str
    scan_id: str
    policy_version: int
    decision: str
    reason: str
    details: dict
    matched_rules: List[dict]
    requires_approval: bool
    approved_by: Optional[str]
    approved_at: Optional[datetime]
    created_at: datetime
    completed_at: Optional[datetime]


class PolicyEvaluationListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[PolicyEvaluationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ApprovalRecordCreate(BaseModel):
    approved_by: str
    decision: str = Field(..., pattern="^(ALLOW|REVIEW|SANITIZE|BLOCK)$")
    comment: Optional[str] = None


class ApprovalRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    evaluation_id: str
    approved_by: str
    decision: str
    comment: Optional[str]
    created_at: datetime


class PolicyDecisionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    policy_id: str
    evaluation_id: str
    decision: str
    reason: str
    details: dict
    created_at: datetime