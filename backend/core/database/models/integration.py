from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Union

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Boolean,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base


class ProviderCategory(str, enum.Enum):
    CLOUD = "cloud"
    LOCAL = "local"
    CUSTOM = "custom"


class ModelIntegrationStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    REGISTERED = "REGISTERED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


class AuthenticationMethod(str, enum.Enum):
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    NONE = "none"


class CapabilityStatus(str, enum.Enum):
    DECLARED = "DECLARED"
    VERIFIED = "VERIFIED"
    UNSUPPORTED = "UNSUPPORTED"
    UNKNOWN = "UNKNOWN"


class ModelIntegration(Base):
    __tablename__ = "model_integrations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_category: Mapped[ProviderCategory] = mapped_column(Enum(ProviderCategory), nullable=False)
    model_identifier: Mapped[str] = mapped_column(String(255), nullable=False)
    adapter_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    base_url: Mapped[str] = mapped_column(String(512), nullable=False)
    auth_method: Mapped[AuthenticationMethod] = mapped_column(Enum(AuthenticationMethod), default=AuthenticationMethod.API_KEY, nullable=False)
    declared_capabilities: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    verified_capabilities: Mapped[dict[str, str]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[ModelIntegrationStatus] = mapped_column(Enum(ModelIntegrationStatus), default=ModelIntegrationStatus.DRAFT, nullable=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    credential_ref: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    verification_diagnostic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_verification_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_model_integrations_status_enabled", "status", "enabled"),
        Index("ix_model_integrations_provider_category", "provider_category"),
        Index("ix_model_integrations_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<ModelIntegration(id={self.id}, name={self.name}, provider={self.provider_category}, status={self.status})>"


class AgentFramework(str, enum.Enum):
    CUSTOM_REST = "custom_rest"
    OPENAI_COMPATIBLE = "openai_compatible"
    LOCAL_AGENT = "local_agent"
    WEBHOOK = "webhook"


class AgentStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    REGISTERED = "REGISTERED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


class AgentIntegration(Base):
    __tablename__ = "agent_integrations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    framework: Mapped[AgentFramework] = mapped_column(Enum(AgentFramework), nullable=False)
    protocol: Mapped[str] = mapped_column(String(100), nullable=False)
    adapter_type: Mapped[str] = mapped_column(String(100), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(512), nullable=False)
    auth_method: Mapped[AuthenticationMethod] = mapped_column(Enum(AuthenticationMethod), default=AuthenticationMethod.API_KEY, nullable=False)
    declared_capabilities: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    verified_capabilities: Mapped[dict[str, str]] = mapped_column(JSON, default=dict, nullable=False)
    intended_operation: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[AgentStatus] = mapped_column(Enum(AgentStatus), default=AgentStatus.DRAFT, nullable=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    credential_ref: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    verification_diagnostic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_verification_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_agent_integrations_status_enabled", "status", "enabled"),
        Index("ix_agent_integrations_framework", "framework"),
        Index("ix_agent_integrations_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AgentIntegration(id={self.id}, name={self.name}, framework={self.framework}, status={self.status})>"


class DatasetAccessGrant(Base):
    __tablename__ = "dataset_access_grants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    integration_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    integration_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    operation: Mapped[str] = mapped_column(String(50), nullable=False)
    granted_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_dataset_access_grants_integration", "integration_type", "integration_id"),
        Index("ix_dataset_access_grants_dataset", "dataset_id", "dataset_version_id"),
    )

    def __repr__(self) -> str:
        return f"<DatasetAccessGrant(id={self.id}, integration={self.integration_type}:{self.integration_id}, dataset={self.dataset_id})>"


class ProviderType(str, enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE_GEMINI = "google_gemini"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    CUSTOM = "custom"


class ProviderStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    TESTING = "testing"


class Provider(Base):
    __tablename__ = "providers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_type: Mapped[ProviderType] = mapped_column(Enum(ProviderType), nullable=False, index=True)
    base_url: Mapped[str] = mapped_column(String(512), nullable=False)
    auth_method: Mapped[AuthenticationMethod] = mapped_column(Enum(AuthenticationMethod), default=AuthenticationMethod.API_KEY, nullable=False)
    credential_ref: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    default_headers: Mapped[dict[str, str]] = mapped_column(JSON, default=dict, nullable=False)
    extra_config: Mapped[dict[str, str]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[ProviderStatus] = mapped_column(Enum(ProviderStatus), default=ProviderStatus.INACTIVE, nullable=False)
    last_test_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_test_result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_providers_type_enabled", "provider_type", "enabled"),
        Index("ix_providers_status", "status"),
        Index("ix_providers_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Provider(id={self.id}, name={self.name}, type={self.provider_type}, status={self.status})>"


class ModelCapability(str, enum.Enum):
    TEXT_GENERATION = "text_generation"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"
    FINE_TUNING = "fine_tuning"
    VISION = "vision"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_GENERATION = "audio_generation"
    IMAGE_GENERATION = "image_generation"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    TOOL_USE = "tool_use"


class ModelCapabilityInfo(Base):
    __tablename__ = "model_capabilities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    provider_id: Mapped[str] = mapped_column(String(36), ForeignKey("providers.id", ondelete="CASCADE"), nullable=False, index=True)
    model_identifier: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    capabilities: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    max_context_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    max_output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    supports_streaming: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    supports_system_prompt: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    supports_parallel_tool_calls: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    input_modalities: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    output_modalities: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    pricing_input_per_1k: Mapped[Optional[float]] = mapped_column(JSON, default=None, nullable=True)
    pricing_output_per_1k: Mapped[Optional[float]] = mapped_column(JSON, default=None, nullable=True)
    is_deprecated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metadata: Mapped[dict[str, str]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_model_capabilities_provider_model", "provider_id", "model_identifier", unique=True),
        Index("ix_model_capabilities_deprecated", "is_deprecated"),
    )

    def __repr__(self) -> str:
        return f"<ModelCapabilityInfo(provider_id={self.provider_id}, model={self.model_identifier})>"