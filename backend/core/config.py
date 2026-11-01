from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Zemo", description="Application name")
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    api_prefix: str = Field(default="/api/v1", description="API version prefix")
    log_level: str = Field(default="INFO", description="Logging level")
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        description="Allowed CORS origins",
    )

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/zemo.db",
        description="Database connection URL",
    )

    # Dataset storage
    upload_dir: str = Field(
        default="./data/uploads",
        description="Directory for storing uploaded dataset files",
    )
    max_file_size_mb: int = Field(
        default=100,
        description="Maximum upload file size in MB",
        ge=1,
        le=1000,
    )
    max_records: int = Field(
        default=1000000,
        description="Maximum number of records to process",
        ge=1,
        le=10000000,
    )
    max_columns: int = Field(
        default=1000,
        description="Maximum number of columns per dataset",
        ge=1,
        le=10000,
    )
    max_field_size_bytes: int = Field(
        default=1048576,
        description="Maximum field size in bytes (1MB)",
        ge=1024,
        le=10485760,
    )
    max_json_nesting_depth: int = Field(
        default=10,
        description="Maximum JSON nesting depth",
        ge=1,
        le=50,
    )

    # Preview limits
    preview_max_rows: int = Field(
        default=100,
        description="Maximum rows in dataset preview",
        ge=1,
        le=1000,
    )
    preview_max_columns: int = Field(
        default=50,
        description="Maximum columns in dataset preview",
        ge=1,
        le=200,
    )

    # Pagination
    default_page_size: int = Field(
        default=20,
        description="Default page size for list endpoints",
        ge=1,
        le=100,
    )
    max_page_size: int = Field(
        default=100,
        description="Maximum page size for list endpoints",
        ge=1,
        le=500,
    )

    # Integration settings
    secret_key: str = Field(
        default="",
        description="Secret key for encrypting stored credentials",
    )
    integration_timeout_seconds: int = Field(
        default=30,
        description="Timeout for integration verification requests",
        ge=5,
        le=300,
    )
    integration_max_response_size: int = Field(
        default=1048576,
        description="Maximum response size for integration requests (1MB)",
        ge=1024,
        le=10485760,
    )
    allowed_integration_schemes: List[str] = Field(
        default=["https", "http"],
        description="Allowed URL schemes for integration endpoints",
    )
    blocked_integration_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1", "::1", "0.0.0.0", "169.254.169.254"],
        description="Blocked hosts for SSRF protection",
    )
    blocked_integration_networks: List[str] = Field(
        default=["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7", "fe80::/10"],
        description="Blocked CIDR networks for SSRF protection",
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v_upper

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("allowed_integration_schemes", mode="before")
    @classmethod
    def parse_allowed_integration_schemes(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [scheme.strip().lower() for scheme in v.split(",") if scheme.strip()]
        return [s.lower() for s in v]

    @field_validator("blocked_integration_hosts", mode="before")
    @classmethod
    def parse_blocked_integration_hosts(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [host.strip().lower() for host in v.split(",") if host.strip()]
        return [h.lower() for h in v]

    @field_validator("blocked_integration_networks", mode="before")
    @classmethod
    def parse_blocked_integration_networks(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [net.strip() for net in v.split(",") if net.strip()]
        return v

    @field_validator("upload_dir")
    @classmethod
    def validate_upload_dir(cls, v: str) -> str:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()