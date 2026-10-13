"""
Data masking and redaction utilities.

All masking is irreversible by design — Zemo never stores
original sensitive values after redaction.
"""

from __future__ import annotations

import hashlib
import re


def mask_email(value: str) -> str:
    """Mask email: j***@example.com"""
    at = value.find("@")
    if at <= 0:
        return "***@***.***"
    return value[0] + "***" + value[at:]


def mask_phone(value: str) -> str:
    """Mask phone number keeping last 4 digits."""
    digits = re.sub(r"\D", "", value)
    if len(digits) < 4:
        return "****"
    return "***-***-" + digits[-4:]


def mask_credit_card(value: str) -> str:
    """Mask credit card showing last 4 digits."""
    digits = re.sub(r"\D", "", value)
    if len(digits) < 4:
        return "****-****-****-****"
    return "****-****-****-" + digits[-4:]


def mask_generic(value: str, visible_chars: int = 3) -> str:
    """Show first N characters then asterisks."""
    if len(value) <= visible_chars:
        return "*" * len(value)
    return value[:visible_chars] + "*" * min(10, len(value) - visible_chars)


def mask_api_key(value: str) -> str:
    """Show prefix and last 4 characters of an API key."""
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]


def redact_value(value: str) -> str:
    """Complete redaction — returns a fixed placeholder."""
    return "[REDACTED]"


def hash_value(value: str, *, salt: str = "zemo") -> str:
    """One-way SHA-256 hash with a fixed application salt."""
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]


def truncate_evidence(value: str, max_len: int = 60) -> str:
    """Safely truncate a value for evidence display."""
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."
