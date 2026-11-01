import ipaddress
import socket
from typing import Optional
from urllib.parse import urlparse

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class SSRFProtectionError(Exception):
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class SSRFProtector:
    def __init__(self):
        settings = get_settings()
        self.allowed_schemes = set(settings.allowed_integration_schemes)
        self.blocked_hosts = set(settings.blocked_integration_hosts)
        self.blocked_networks = [ipaddress.ip_network(n) for n in settings.blocked_integration_networks]

    def validate_url(self, url: str) -> tuple[str, int]:
        parsed = urlparse(url)

        if parsed.scheme.lower() not in self.allowed_schemes:
            raise SSRFProtectionError(
                f"URL scheme '{parsed.scheme}' not allowed",
                {"scheme": parsed.scheme, "allowed_schemes": list(self.allowed_schemes)},
            )

        if parsed.username or parsed.password:
            raise SSRFProtectionError(
                "URL contains embedded credentials",
                {"has_credentials": True},
            )

        host = parsed.hostname
        if not host:
            raise SSRFProtectionError("URL missing hostname")

        host_lower = host.lower()
        if host_lower in self.blocked_hosts:
            raise SSRFProtectionError(
                f"Host '{host}' is blocked",
                {"host": host, "blocked": True},
            )

        try:
            ip = ipaddress.ip_address(host)
            if self._is_ip_blocked(ip):
                raise SSRFProtectionError(
                    f"IP address '{ip}' is in blocked network range",
                    {"ip": str(ip)},
                )
        except ValueError:
            resolved_ips = self._resolve_hostname(host)
            for ip in resolved_ips:
                if self._is_ip_blocked(ip):
                    raise SSRFProtectionError(
                        f"Hostname '{host}' resolves to blocked IP '{ip}'",
                        {"host": host, "resolved_ip": str(ip)},
                    )

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return host_lower, port

    def _resolve_hostname(self, hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        try:
            addrs = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            ips = set()
            for addr in addrs:
                ip_str = addr[4][0]
                ips.add(ipaddress.ip_address(ip_str))
            return list(ips)
        except socket.gaierror as e:
            logger.warning("dns_resolution_failed", host=hostname, error=str(e))
            return []

    def _is_ip_blocked(self, ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return True

        if isinstance(ip, ipaddress.IPv4Address):
            if ip in ipaddress.IPv4Network("169.254.0.0/16"):
                return True

        for network in self.blocked_networks:
            if ip in network:
                return True
        return False


def create_safe_client(
    timeout_seconds: int,
    max_response_size: int,
) -> "httpx.AsyncClient":
    import httpx

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(timeout_seconds, connect=10.0)

    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        follow_redirects=False,
        max_response_size=max_response_size,
    )


_ssrf_protector: Optional[SSRFProtector] = None


def get_ssrf_protector() -> SSRFProtector:
    global _ssrf_protector
    if _ssrf_protector is None:
        _ssrf_protector = SSRFProtector()
    return _ssrf_protector