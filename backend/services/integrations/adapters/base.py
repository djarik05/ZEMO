from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AdapterConfig:
    base_url: str
    model_identifier: str
    auth_method: str
    credential: Optional[str] = None
    extra: Optional[dict[str, Any]] = None


@dataclass
class VerificationResult:
    success: bool
    status: str
    diagnostic: str
    capabilities: dict[str, str]
    metadata: Optional[dict[str, Any]] = None


@dataclass
class CapabilityCheckResult:
    capability: str
    status: str
    verified: bool
    diagnostic: str
    metadata: Optional[dict[str, Any]] = None


class BaseAdapter(ABC):
    adapter_type: str = "base"

    def __init__(self, config: AdapterConfig, timeout_seconds: int = 30, max_response_size: int = 1048576):
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.max_response_size = max_response_size

    @abstractmethod
    async def verify_connection(self) -> VerificationResult:
        pass

    @abstractmethod
    async def check_capability(self, capability: str) -> CapabilityCheckResult:
        pass

    @abstractmethod
    def get_supported_capabilities(self) -> list[str]:
        pass

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "Zemo/1.0",
            "Accept": "application/json",
        }

        if self.config.auth_method == "api_key" and self.config.credential:
            headers["Authorization"] = f"Bearer {self.config.credential}"
        elif self.config.auth_method == "bearer_token" and self.config.credential:
            headers["Authorization"] = f"Bearer {self.config.credential}"

        return headers


class AdapterError(Exception):
    def __init__(self, message: str, adapter_type: str, details: dict | None = None):
        self.message = message
        self.adapter_type = adapter_type
        self.details = details or {}
        super().__init__(message)


class AdapterRegistry:
    _adapters: dict[str, type[BaseAdapter]] = {}

    @classmethod
    def register(cls, adapter_type: str, adapter_class: type[BaseAdapter]) -> None:
        cls._adapters[adapter_type] = adapter_class
        logger.info("adapter_registered", adapter_type=adapter_type)

    @classmethod
    def get(cls, adapter_type: str) -> type[BaseAdapter]:
        if adapter_type not in cls._adapters:
            raise AdapterError(
                f"Unknown adapter type: {adapter_type}",
                adapter_type,
                {"available": list(cls._adapters.keys())},
            )
        return cls._adapters[adapter_type]

    @classmethod
    def list_adapters(cls) -> list[str]:
        return list(cls._adapters.keys())