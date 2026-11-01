"""
Base provider classes and interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Dict, List
import os
from pydantic import BaseModel, Field, HttpUrl


class ProviderType(str, Enum):
    """Types of AI model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE_GEMINI = "google_gemini"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    CUSTOM = "custom"


class ProviderStatus(str, Enum):
    """Connection status of a provider."""
    UNKNOWN = "unknown"
    NOT_CONFIGURED = "not_configured"
    CONFIGURED = "configured"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    ERROR = "error"


class ModelCapability(str, Enum):
    """Supported model capabilities."""
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"
    TOOL_USE = "tool_use"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    FINE_TUNING = "fine_tuning"
    EMBEDDINGS = "embeddings"
    STRUCTURED_OUTPUT = "structured_output"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"


@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    name: str
    provider: str
    capabilities: List[str] = field(default_factory=list)
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    pricing: Optional[Dict[str, float]] = None
    description: Optional[str] = None
    deprecated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    name: str
    display_name: str
    description: str = ""
    api_key_env_var: Optional[str] = None
    base_url: Optional[str] = None
    default_models: List[str] = field(default_factory=list)
    supported_capabilities: List[str] = field(default_factory=list)
    requires_api_key: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    max_retries: int = 3
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderStatusInfo:
    """Current status of a provider."""
    provider_id: str
    status: str = "unknown"
    last_checked: Optional[datetime] = None
    last_error: Optional[str] = None
    connected_at: Optional[datetime] = None
    models_available: int = 0
    last_error: Optional[str] = None
    connection_latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelCapabilityInfo:
    """Model capability information."""
    def __init__(self, model_id: str, capabilities: List[str] = None, verified: bool = False, 
                 verified_at: Optional[datetime] = None, details: Dict[str, Any] = None):
        self.model_id = model_id
        self.capabilities = capabilities or []
        self.verified = verified
        self.verified_at = verified_at
        self.details = details or {}


class ConnectionTestResult:
    """Result of a connection test."""
    def __init__(self, success: bool, provider_id: str, latency_ms: float = None, 
                 error: str = None, models_found: List[str] = None, 
                 capabilities: Dict[str, bool] = None):
        self.success = success
        self.provider_id = provider_id
        self.latency_ms = latency_ms
        self.error = error
        self.models_found = models_found or []
        self.capabilities = capabilities or {}
        self.timestamp = datetime.utcnow()


class BaseProvider:
    """Abstract base class for AI model providers."""
    
    def __init__(self, config: dict, api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key
        self._client = None
        self._models_cache: Optional[List[dict]] = None
        self._models_cache_time: Optional[datetime] = None
        self._cache_ttl = 300  # 5 minutes
    
    @property
    def provider_type(self) -> str:
        """Return the provider type."""
        return self.config.get("provider_type", "unknown")
    
    @property
    def provider_name(self) -> str:
        """Return the provider display name."""
        return self.config.get("display_name", self.config.get("name", "unknown"))
    
    def validate_config(self) -> bool:
        """Validate the provider configuration."""
        if not self.config.get("name"):
            return False
        if self.config.get("requires_api_key", True):
            api_key = self.config.get("api_key_env_var")
            if not os.environ.get(api_key) and not self.config.get("api_key"):
                return False
        return True
    
    @abstractmethod
    async def test_connection(self) -> dict:
        """Test the connection to the provider."""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[dict]:
        """List available models from the provider."""
        pass
    
    @abstractmethod
    async def get_model_info(self, model_id: str) -> Optional[dict]:
        """Get information about a specific model."""
        pass
    
    @abstractmethod
    async def check_capabilities(self, model_id: str) -> Dict[str, bool]:
        """Check which capabilities are supported by a model."""
        pass
    
    @abstractmethod
    async def get_model_capabilities(self, model_id: str) -> dict:
        """Get detailed capability information for a model."""
        pass
    
    def _get_cached_models(self) -> Optional[List[dict]]:
        """Get cached models if still valid."""
        if self._models_cache and self._models_cache_time:
            if (datetime.utcnow() - self._models_cache_time).seconds < self._cache_ttl:
                return self._models_cache
        return None
    
    def _cache_models(self, models: List[dict]) -> None:
        """Cache models list."""
        self._models_cache = models
        self._models_cache_time = datetime.utcnow()
    
    def clear_cache(self) -> None:
        """Clear the models cache."""
        self._models_cache = None
        self._models_cache_time = None
    
    def _get_api_key(self) -> Optional[str]:
        """Get the API key from environment or config."""
        # Check config first
        if self.config.get("api_key"):
            return self.config.get("api_key")
        # Then check environment variable
        env_var = self.config.get("api_key_env_var")
        if env_var:
            return os.environ.get(env_var)
        return None


class ProviderRegistry:
    """Registry for managing AI model providers."""
    
    def __init__(self):
        self._providers = {}
        self._configs = {}
        self._status = {}
        self._initialized = False
    
    def register_provider(self, name: str, config: dict, provider_class: type) -> None:
        """Register a provider class with its configuration."""
        if name in self._configs:
            raise ValueError(f"Provider {name} already registered")
        
        self._configs[name] = config
        self._status[name] = {
            "provider_id": name,
            "status": "unknown",
            "last_checked": None,
            "last_error": None,
            "connected_at": None,
            "models_available": 0,
            "last_error": None,
            "connection_latency_ms": None,
            "metadata": {}
        }
        
        # Store the provider class for lazy instantiation
        if not hasattr(self, '_provider_classes'):
            self._provider_classes = {}
        self._provider_classes[name] = config.get("provider_class")
        self._initialized = True
    
    def get_provider(self, provider_id: str, api_key: Optional[str] = None):
        """Get or create a provider instance."""
        if provider_id not in self._configs:
            return None
        
        config = self._configs[provider_id]
        if not config.get("enabled", True):
            return None
        
        # Check if we have an API key
        api_key = None
        if config.get("api_key_env_var"):
            import os
            api_key = os.environ.get(config["api_key_env_var"])
        
        # Create provider instance (simplified - in production, use a factory)
        return None  # Placeholder - actual implementation would create provider instance
    
    def get_config(self, provider_id: str) -> Optional[dict]:
        """Get provider configuration."""
        return self._configs.get(provider_id)
    
    def get_all_configs(self) -> List[dict]:
        """Get all provider configurations."""
        return list(self._configs.values())
    
    def get_status(self, provider_id: str) -> Optional[dict]:
        """Get provider status."""
        return self._status.get(provider_id)
    
    def get_all_statuses(self) -> Dict[str, dict]:
        """Get all provider statuses."""
        return self._status.copy()
    
    def update_status(self, provider_id: str, status: dict) -> None:
        """Update provider status."""
        if provider_id in self._status:
            self._status[provider_id] = status
    
    def is_enabled(self, provider_id: str) -> bool:
        """Check if provider is enabled."""
        config = self._configs.get(provider_id)
        return config is not None and config.get("enabled", True)
    
    def list_enabled_providers(self) -> List[str]:
        """List all enabled provider IDs."""
        return [pid for pid, config in self._configs.items() if config.get("enabled", True)]


# Global provider registry instance
_provider_registry = None


def get_provider_registry():
    """Get the global provider registry instance."""
    global _provider_registry
    if _provider_registry is None:
        _provider_registry = ProviderRegistry()
    return _provider_registry


# Provider configuration registry
def register_default_providers(registry) -> None:
    """Register all default provider configurations."""
    # Import providers here to avoid circular imports
    from services.providers.openai_provider import OpenAIProvider
    from services.providers.anthropic_provider import AnthropicProvider
    from services.providers.gemini_provider import GoogleGeminiProvider
    from services.providers.mistral_provider import MistralProvider
    from services.providers.ollama_provider import OllamaProvider
    
    # OpenAI
    registry.register_provider(
        name="openai",
        config={
            "name": "openai",
            "display_name": "OpenAI",
            "description": "OpenAI API (GPT models)",
            "provider_type": "openai",
            "api_key_env_var": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "default_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "supported_capabilities": [
                "text_generation", "chat_completion", "function_calling",
                "vision", "structured_output", "reasoning"
            ],
            "enabled": True,
        }, "OpenAIProvider")
    
    # Anthropic
    registry.register_provider(
        name="anthropic",
        config={
            "name": "anthropic",
            "display_name": "Anthropic",
            "description": "Anthropic API (Claude models)",
            "provider_type": "anthropic",
            "api_key_env_var": "ANTHROPIC_API_KEY",
            "base_url": "https://api.anthropic.com/v1",
            "default_models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
            "supported_capabilities": [
                "text_generation", "chat_completion", "function_calling",
                "vision", "reasoning"
            ],
            "enabled": True,
        }, "AnthropicProvider")
    
    # Google Gemini
    registry.register_provider(
        name="google_gemini",
        config={
            "name": "google_gemini",
            "display_name": "Google Gemini",
            "description": "Google Gemini API",
            "provider_type": "google_gemini",
            "api_key_env_var": "GOOGLE_API_KEY",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "default_models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
            "supported_capabilities": [
                "text_generation", "chat_completion", "function_calling",
                "vision", "embeddings", "reasoning"
            ],
            "enabled": True,
        }, "GoogleGeminiProvider")
    
    # Mistral
    registry.register_provider(
        name="mistral",
        config={
            "name": "mistral",
            "display_name": "Mistral AI",
            "description": "Mistral AI API",
            "provider_type": "mistral",
            "api_key_env_var": "MISTRAL_API_KEY",
            "base_url": "https://api.mistral.ai/v1",
            "default_models": ["mistral-large-latest", "mistral-small-latest", "mistral-embed"],
            "supported_capabilities": [
                "text_generation", "chat_completion", "function_calling",
                "embeddings", "reasoning"
            ],
            "enabled": True,
        }, "MistralProvider")
    
    # Ollama (local)
    registry.register_provider(
        name="ollama",
        config={
            "name": "ollama",
            "display_name": "Ollama",
            "description": "Ollama local models",
            "provider_type": "ollama",
            "api_key_env_var": None,
            "base_url": "http://localhost:11434",
            "default_models": ["llama3.1", "llama3.2", "mistral", "codellama"],
            "supported_capabilities": [
                "text_generation", "chat_completion", "embeddings"
            ],
            "requires_api_key": False,
            "enabled": True,
        }, "OllamaProvider")


# Initialize default providers on module load
def initialize_providers() -> None:
    """Initialize the provider registry with default providers."""
    registry = get_provider_registry()
    register_default_providers(registry)


def get_provider_registry():
    """Get the global provider registry instance."""
    global _provider_registry
    if _provider_registry is None:
        _provider_registry = ProviderRegistry()
    return _provider_registry