from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE_GEMINI = "google_gemini"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    CUSTOM = "custom"


class ProviderStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    TESTING = "testing"


class ProviderBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=255)
    provider_type: str = Field(..., pattern="^(openai|anthropic|google_gemini|mistral|ollama|openai_compatible|custom)$")
    base_url: HttpUrl
    auth_method: str = Field(..., pattern="^(api_key|bearer_token|basic_auth|oauth2|none)$")
    default_headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    extra_config: Optional[Dict[str, str]] = Field(default_factory=dict)


class ProviderCreate(ProviderBase):
    credential: Optional[str] = Field(None, min_length=1)


class ProviderUpdate(BaseModel):
    display_name: Optional[str] = Field(None, min_length=1, max_length=255)
    base_url: Optional[HttpUrl] = None
    auth_method: Optional[str] = Field(None, pattern="^(api_key|bearer_token|basic_auth|oauth2|none)$")
    credential: Optional[str] = None
    default_headers: Optional[Dict[str, str]] = None
    extra_config: Optional[Dict[str, str]] = None
    enabled: Optional[bool] = None


class ProviderResponse(ProviderBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    enabled: bool
    last_test_at: Optional[datetime] = None
    last_test_result: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ProviderListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[ProviderResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ProviderTestResponse(BaseModel):
    success: bool
    status: str
    diagnostic: str
    capabilities: Dict[str, str]


class ProviderSmokeTestRequest(BaseModel):
    model: Optional[str] = Field(None, min_length=1)


class ProviderSmokeTestResponse(BaseModel):
    success: bool
    model_tested: str
    status: str
    diagnostic: str
    capabilities: Dict[str, str]


class ModelCapabilityBase(BaseModel):
    model_identifier: str = Field(..., min_length=1, max_length=255)
    display_name: str = Field(..., min_length=1, max_length=255)
    capabilities: List[str] = Field(default_factory=list)
    max_context_tokens: Optional[int] = Field(None, ge=1)
    max_output_tokens: Optional[int] = Field(None, ge=1)
    supports_streaming: bool = False
    supports_system_prompt: bool = True
    supports_parallel_tool_calls: bool = False
    input_modalities: List[str] = Field(default_factory=list)
    output_modalities: List[str] = Field(default_factory=list)
    pricing_input_per_1k: Optional[float] = Field(None, ge=0)
    pricing_output_per_1k: Optional[float] = Field(None, ge=0)
    is_deprecated: bool = False
    metadata: Dict[str, str] = Field(default_factory=dict)


class ModelCapabilityCreate(ModelCapabilityBase):
    pass


class ModelCapabilityUpdate(BaseModel):
    display_name: Optional[str] = Field(None, min_length=1, max_length=255)
    capabilities: Optional[List[str]] = None
    max_context_tokens: Optional[int] = Field(None, ge=1)
    max_output_tokens: Optional[int] = Field(None, ge=1)
    supports_streaming: Optional[bool] = None
    supports_system_prompt: Optional[bool] = None
    supports_parallel_tool_calls: Optional[bool] = None
    input_modalities: Optional[List[str]] = None
    output_modalities: Optional[List[str]] = None
    pricing_input_per_1k: Optional[float] = Field(None, ge=0)
    pricing_output_per_1k: Optional[float] = Field(None, ge=0)
    is_deprecated: Optional[bool] = None
    metadata: Optional[Dict[str, str]] = None


class ModelCapabilityResponse(ModelCapabilityBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    provider_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def extract_metadata(cls, data: Any) -> Any:
        if hasattr(data, "model_metadata"):
            return {
                "id": getattr(data, "id", None),
                "provider_id": getattr(data, "provider_id", None),
                "model_identifier": getattr(data, "model_identifier", ""),
                "display_name": getattr(data, "display_name", ""),
                "capabilities": getattr(data, "capabilities", []),
                "max_context_tokens": getattr(data, "max_context_tokens", None),
                "max_output_tokens": getattr(data, "max_output_tokens", None),
                "supports_streaming": getattr(data, "supports_streaming", False),
                "supports_system_prompt": getattr(data, "supports_system_prompt", True),
                "supports_parallel_tool_calls": getattr(data, "supports_parallel_tool_calls", False),
                "input_modalities": getattr(data, "input_modalities", []),
                "output_modalities": getattr(data, "output_modalities", []),
                "pricing_input_per_1k": getattr(data, "pricing_input_per_1k", None),
                "pricing_output_per_1k": getattr(data, "pricing_output_per_1k", None),
                "is_deprecated": getattr(data, "is_deprecated", False),
                "metadata": getattr(data, "model_metadata", {}) or {},
                "created_at": getattr(data, "created_at", None),
                "updated_at": getattr(data, "updated_at", None),
            }
        return data


class ModelCapabilityListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[ModelCapabilityResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ModelIntegrationBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=255)
    provider_category: str = Field(..., pattern="^(cloud|local|custom)$")
    model_identifier: str = Field(..., min_length=1, max_length=255)
    adapter_type: str = Field(..., min_length=1, max_length=100)
    base_url: HttpUrl
    auth_method: str = Field(..., pattern="^(api_key|bearer_token|basic_auth|oauth2|none)$")
    declared_capabilities: List[str] = Field(default_factory=list)


class ModelIntegrationCreate(ModelIntegrationBase):
    credential: Optional[str] = Field(None, min_length=1)
    extra: Optional[Dict[str, Any]] = None


class ModelIntegrationUpdate(BaseModel):
    display_name: Optional[str] = Field(None, min_length=1, max_length=255)
    model_identifier: Optional[str] = Field(None, min_length=1, max_length=255)
    base_url: Optional[HttpUrl] = None
    auth_method: Optional[str] = Field(None, pattern="^(api_key|bearer_token|basic_auth|oauth2|none)$")
    declared_capabilities: Optional[List[str]] = None
    credential: Optional[str] = None
    enabled: Optional[bool] = None


class ModelIntegrationResponse(ModelIntegrationBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    enabled: bool
    verified_capabilities: Dict[str, str]
    verification_diagnostic: Optional[str] = None
    last_verification_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class ModelIntegrationListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[ModelIntegrationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class AgentIntegrationBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    framework: str = Field(..., pattern="^(custom_rest|openai_compatible|local_agent|webhook)$")
    protocol: str = Field(..., min_length=1, max_length=100)
    adapter_type: str = Field(..., min_length=1, max_length=100)
    endpoint: HttpUrl
    auth_method: str = Field(..., pattern="^(api_key|bearer_token|basic_auth|oauth2|none)$")
    declared_capabilities: List[str] = Field(default_factory=list)
    intended_operation: str = Field(..., min_length=1, max_length=100)


class AgentIntegrationCreate(AgentIntegrationBase):
    credential: Optional[str] = Field(None, min_length=1)


class AgentIntegrationUpdate(BaseModel):
    display_name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    endpoint: Optional[HttpUrl] = None
    auth_method: Optional[str] = Field(None, pattern="^(api_key|bearer_token|basic_auth|oauth2|none)$")
    declared_capabilities: Optional[List[str]] = None
    credential: Optional[str] = None
    enabled: Optional[bool] = None
    intended_operation: Optional[str] = Field(None, min_length=1, max_length=100)


class AgentIntegrationResponse(AgentIntegrationBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    enabled: bool
    verified_capabilities: Dict[str, str]
    verification_diagnostic: Optional[str] = None
    last_verification_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class AgentIntegrationListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[AgentIntegrationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class CapabilityCheckRequest(BaseModel):
    capability: str = Field(..., min_length=1)


class CapabilityCheckResponse(BaseModel):
    capability: str
    status: str
    verified: bool
    diagnostic: str
    metadata: Optional[Dict[str, Any]] = None


class VerificationResponse(BaseModel):
    success: bool
    status: str
    diagnostic: str
    capabilities: Dict[str, str]


class DatasetAccessGrantCreate(BaseModel):
    integration_type: str = Field(..., pattern="^(model|agent)$")
    integration_id: str
    dataset_id: str
    dataset_version_id: Optional[str] = None
    operation: str = Field(..., min_length=1, max_length=50)
    expires_at: Optional[datetime] = None


class DatasetAccessGrantResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    integration_type: str
    integration_id: str
    dataset_id: str
    dataset_version_id: Optional[str] = None
    operation: str
    granted_by: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime
    revoked_at: Optional[datetime] = None


class AdapterInfoResponse(BaseModel):
    adapter_type: str
    category: str
    description: str
    supported_capabilities: List[str]