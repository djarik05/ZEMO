"""
Unit and integration tests for SecretDetector.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.models.enums import FindingCategory, Severity
from app.services.secret_detector import SecretDetector


def test_detect_api_key() -> None:
    """Feed data with 'sk-proj-abc123def456ghi789', verify API_KEY finding."""
    df = pd.DataFrame({
        "config": ["user_agent=default", "sk-proj-abc123def456ghi789", "timeout=30"]
    })
    detector = SecretDetector()
    findings = detector.detect("ds-secret-1", df)

    api_key_findings = [f for f in findings if f["category"] == FindingCategory.API_KEY.value]
    assert len(api_key_findings) >= 1
    assert api_key_findings[0]["severity"] == Severity.CRITICAL.value
    assert "..." in api_key_findings[0]["evidence"]  # Masked


def test_detect_password() -> None:
    """Feed 'password=MyS3cretP@ss', verify PASSWORD finding."""
    df = pd.DataFrame({
        "auth_info": ["role=admin", "password=MyS3cretP@ss", "status=active"]
    })
    detector = SecretDetector()
    findings = detector.detect("ds-secret-2", df)

    pwd_findings = [f for f in findings if f["category"] == FindingCategory.PASSWORD.value]
    assert len(pwd_findings) >= 1
    assert pwd_findings[0]["severity"] == Severity.CRITICAL.value


def test_detect_private_key() -> None:
    """Feed PEM key header, verify PRIVATE_KEY finding."""
    pem_key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0\n-----END RSA PRIVATE KEY-----"
    df = pd.DataFrame({
        "cert": ["valid", pem_key, "exp=2028"]
    })
    detector = SecretDetector()
    findings = detector.detect("ds-secret-3", df)

    key_findings = [f for f in findings if f["category"] == FindingCategory.PRIVATE_KEY.value]
    assert len(key_findings) >= 1
    assert key_findings[0]["severity"] == Severity.CRITICAL.value
    assert "PRIVATE KEY" in key_findings[0]["evidence"]


def test_detect_db_credential() -> None:
    """Feed connection string, verify DB_CREDENTIAL finding."""
    db_uri = "postgresql://admin:super_secret_pw@db.production.internal:5432/zemo"
    df = pd.DataFrame({
        "env_var": ["DEBUG=False", db_uri, "PORT=8000"]
    })
    detector = SecretDetector()
    findings = detector.detect("ds-secret-4", df)

    db_findings = [f for f in findings if f["category"] == FindingCategory.DB_CREDENTIAL.value]
    assert len(db_findings) >= 1
    assert db_findings[0]["severity"] == Severity.CRITICAL.value


def test_column_name_detection() -> None:
    """Feed DataFrame with column named 'api_secret', verify finding."""
    df = pd.DataFrame({
        "username": ["admin", "guest"],
        "api_secret": ["val1", "val2"],
    })
    detector = SecretDetector()
    findings = detector.detect("ds-secret-5", df)

    col_findings = [f for f in findings if f.get("field_name") == "api_secret"]
    assert len(col_findings) >= 1
    assert col_findings[0]["severity"] == Severity.CRITICAL.value
    assert col_findings[0]["subcategory"] == "suspicious_column_name"


def test_all_secrets_critical() -> None:
    """Verify all secret findings have CRITICAL severity."""
    df = pd.DataFrame({
        "token": ["test_token_val"],
        "api_keys": ["sk-proj-abc123def456ghi789"],
        "connection": ["mysql://user:pass1234@localhost:3306/prod"],
        "passwords": ["password=SuperSecretPassword123"],
    })
    detector = SecretDetector()
    findings = detector.detect("ds-secret-all", df)

    assert len(findings) > 0
    for f in findings:
        assert f["severity"] == Severity.CRITICAL.value
