"""
Zemo application configuration.
Loaded from environment variables with Pydantic validation.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field

# Matches localhost, 127.0.0.1, RFC 1918 private LAN IPv4 ranges (10/8, 172.16/12, 192.168/16),
# and mDNS (.local) hostnames on any port, without allowing arbitrary public origins.
DEFAULT_LAN_ORIGIN_REGEX = (
    r"^https?://("
    r"localhost|"
    r"127\.0\.0\.1|"
    r"10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}|"
    r"[a-zA-Z0-9-]+\.local"
    r")(?::\d{1,5})?$"
)


@lru_cache(maxsize=1)
def get_lan_ips() -> list[str]:
    """
    Detect active local network (LAN/Wi-Fi) IPv4 addresses dynamically.
    Probes RFC 1918 private routes first so VPN tunnels (e.g., Cloudflare WARP 172.16.0.2)
    do not mask the physical Wi-Fi/Ethernet adapter.
    """
    candidates: list[str] = []

    for target in ("192.168.255.254", "10.255.255.254", "172.31.255.254"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(0.1)
                sock.connect((target, 80))
                ip = sock.getsockname()[0]
                if ip and ip not in candidates and ip != "172.16.0.2":
                    candidates.append(ip)
        except OSError:
            pass

    if not candidates:
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if ip and ip not in candidates:
                    candidates.append(ip)
        except OSError:
            pass

    valid_ips: list[str] = []
    fallback_ips: list[str] = []
    for ip in candidates:
        try:
            addr = ipaddress.ip_address(ip)
            if addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_unspecified:
                continue
            # Deprioritize Cloudflare WARP's /32 virtual tunnel address
            if ip == "172.16.0.2":
                fallback_ips.append(ip)
                continue
            if addr.is_private:
                valid_ips.append(ip)
            else:
                fallback_ips.append(ip)
        except ValueError:
            continue

    return valid_ips or fallback_ips or ["127.0.0.1"]


def get_primary_lan_ip() -> str:
    """Return the primary LAN IPv4 address of the host machine."""
    return get_lan_ips()[0]


class Settings(BaseSettings):
    """Validated application settings sourced from environment or .env file."""

    # ── Application ──────────────────────────────────────────────
    app_name: str = Field(default="Zemo", alias="ZEMO_APP_NAME")
    app_version: str = Field(default="0.1.0", alias="ZEMO_APP_VERSION")
    debug: bool = Field(default=False, alias="ZEMO_DEBUG")
    log_level: str = Field(default="INFO", alias="ZEMO_LOG_LEVEL")

    # ── Server ───────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="ZEMO_HOST")
    port: int = Field(default=8000, alias="ZEMO_PORT")

    # ── Storage ──────────────────────────────────────────────────
    storage_dir: str = Field(default="storage", alias="ZEMO_STORAGE_DIR")
    max_upload_size_mb: int = Field(default=50, alias="ZEMO_MAX_UPLOAD_SIZE_MB")
    max_record_count: int = Field(default=500_000, alias="ZEMO_MAX_RECORD_COUNT")
    max_column_count: int = Field(default=500, alias="ZEMO_MAX_COLUMN_COUNT")
    max_record_size_bytes: int = Field(default=1_048_576, alias="ZEMO_MAX_RECORD_SIZE_BYTES")

    # ── Security ─────────────────────────────────────────────────
    allowed_origins: str = Field(
        default=(
            "http://127.0.0.1:8000,http://localhost:8000,"
            "http://localhost:3000,http://127.0.0.1:3000,"
            "http://localhost:5500,http://127.0.0.1:5500,"
            "http://localhost:8080,http://127.0.0.1:8080"
        ),
        alias="ZEMO_ALLOWED_ORIGINS",
    )
    allowed_origin_regex: str = Field(
        default=DEFAULT_LAN_ORIGIN_REGEX,
        alias="ZEMO_ALLOWED_ORIGIN_REGEX",
    )

    # ── Derived ──────────────────────────────────────────────────
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def storage_path(self) -> Path:
        base = Path(self.storage_dir)
        if not base.is_absolute():
            backend_root = Path(__file__).resolve().parents[2]
            base = backend_root / base
        base.mkdir(parents=True, exist_ok=True)
        return base

    @property
    def cors_origins(self) -> list[str]:
        origins = [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
        ports = {self.port, 8000, 3000, 5500, 8080}
        for lan_ip in get_lan_ips():
            if lan_ip and lan_ip != "127.0.0.1":
                for p in sorted(ports):
                    candidate = f"http://{lan_ip}:{p}"
                    if candidate not in origins:
                        origins.append(candidate)
        return origins

    model_config = {
        "env_file": str(Path(__file__).resolve().parents[2] / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()

