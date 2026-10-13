"""
Unit and integration tests for PrivacyDetector.
"""

from __future__ import annotations

import io
import pandas as pd
import pytest

from app.models.enums import FindingCategory, Severity
from app.services.privacy_detector import PrivacyDetector


def test_detect_email(sample_csv_bytes: bytes) -> None:
    """Feed CSV with emails, verify EMAIL findings exist with severity HIGH."""
    df = pd.read_csv(io.BytesIO(sample_csv_bytes))
    detector = PrivacyDetector()
    findings = detector.detect("test-ds-1", df)

    email_findings = [f for f in findings if f["category"] == FindingCategory.EMAIL.value]
    assert len(email_findings) > 0

    for f in email_findings:
        assert f["severity"] == Severity.HIGH.value
        assert "@" in f["evidence"]
        assert "***" in f["evidence"]  # Masked format


def test_detect_phone(sample_csv_bytes: bytes) -> None:
    """Verify PHONE findings exist with severity HIGH."""
    df = pd.read_csv(io.BytesIO(sample_csv_bytes))
    detector = PrivacyDetector()
    findings = detector.detect("test-ds-2", df)

    phone_findings = [f for f in findings if f["category"] == FindingCategory.PHONE.value]
    assert len(phone_findings) > 0

    for f in phone_findings:
        assert f["severity"] == Severity.HIGH.value
        assert "***-***-" in f["evidence"]


def test_detect_credit_card(sample_csv_bytes: bytes) -> None:
    """Verify CREDIT_CARD findings with severity CRITICAL."""
    df = pd.read_csv(io.BytesIO(sample_csv_bytes))
    detector = PrivacyDetector()
    findings = detector.detect("test-ds-3", df)

    cc_findings = [f for f in findings if f["category"] == FindingCategory.CREDIT_CARD.value]
    assert len(cc_findings) >= 1

    for f in cc_findings:
        assert f["severity"] == Severity.CRITICAL.value
        assert "1111" in f["evidence"]
        assert "****" in f["evidence"]


def test_detect_ssn(sample_csv_bytes: bytes) -> None:
    """Verify PERSONAL_ID findings with severity CRITICAL."""
    df = pd.read_csv(io.BytesIO(sample_csv_bytes))
    detector = PrivacyDetector()
    findings = detector.detect("test-ds-4", df)

    ssn_findings = [f for f in findings if f["category"] == FindingCategory.PERSONAL_ID.value]
    assert len(ssn_findings) >= 1

    for f in ssn_findings:
        assert f["severity"] == Severity.CRITICAL.value
        assert "*" in f["evidence"]


def test_detect_address(sample_csv_bytes: bytes) -> None:
    """Verify ADDRESS findings with MEDIUM severity."""
    df = pd.read_csv(io.BytesIO(sample_csv_bytes))
    detector = PrivacyDetector()
    findings = detector.detect("test-ds-5", df)

    addr_findings = [f for f in findings if f["category"] == FindingCategory.ADDRESS.value]
    assert len(addr_findings) > 0

    for f in addr_findings:
        assert f["severity"] == Severity.MEDIUM.value


def test_no_false_positives_clean_data(sample_clean_csv_bytes: bytes) -> None:
    """Feed clean data, verify 0 privacy findings."""
    df = pd.read_csv(io.BytesIO(sample_clean_csv_bytes))
    detector = PrivacyDetector()
    findings = detector.detect("test-clean-ds", df)

    assert len(findings) == 0


def test_evidence_is_masked(sample_csv_bytes: bytes) -> None:
    """Verify that evidence field contains masked values (never raw PII)."""
    df = pd.read_csv(io.BytesIO(sample_csv_bytes))
    detector = PrivacyDetector()
    findings = detector.detect("test-ds-masked", df)

    raw_pii_samples = [
        "alice.smith@example.com",
        "bob.jones@example.com",
        "4111111111111111",
        "123-45-6789",
    ]

    for f in findings:
        evidence = f["evidence"]
        for raw in raw_pii_samples:
            assert raw not in evidence, f"Raw sensitive value '{raw}' found unmasked in evidence: {evidence}"
