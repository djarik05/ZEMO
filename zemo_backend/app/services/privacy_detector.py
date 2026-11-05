"""
Privacy & PII detection service.

Detects personally identifiable information (PII) including:
- Email addresses
- Phone numbers
- Credit card numbers
- Physical street addresses
- Personal identification numbers (SSN)
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from app.core.logging import get_logger
from app.core.security import generate_storage_id
from app.models.enums import (
    DetectionMethod,
    FindingCategory,
    PolicyDecision,
    RemediationStatus,
    ReviewStatus,
    Severity,
    utcnow,
)
from app.utils.masking import (
    mask_credit_card,
    mask_email,
    mask_generic,
    mask_phone,
    truncate_evidence,
)

logger = get_logger("privacy_detector")

# Compiled regular expressions for privacy detection
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_REGEX = re.compile(r"(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}")
CREDIT_CARD_REGEX = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
)
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
ADDRESS_REGEX = re.compile(
    r"\b\d{1,5}\s+[a-zA-Z0-9\s.,]{1,40}\s+(?:street|st\.?|avenue|ave\.?|blvd|boulevard|road|rd\.?|drive|dr\.?|lane|ln\.?|court|ct\.?|way|terrace|circle)\b",
    re.IGNORECASE,
)


class PrivacyDetector:
    """Service to scan datasets for Personally Identifiable Information (PII)."""

    def detect(self, dataset_id: str, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Scan every cell in every column of DataFrame for PII patterns.
        Returns a list of finding dictionaries conforming to the FindingBase schema.
        """
        findings: list[dict[str, Any]] = []
        now = utcnow()

        for col_name in df.columns:
            series = df[col_name]
            for row_idx, val in series.items():
                if val is None or pd.isna(val):
                    continue

                val_str = str(val).strip()
                if not val_str:
                    continue

                idx = int(row_idx) if isinstance(row_idx, (int, float)) else 0
                location = f"row:{idx},col:{col_name}"

                # 1. Credit Card (CRITICAL)
                cc_matches = CREDIT_CARD_REGEX.findall(val_str)
                for match in cc_matches:
                    clean_digits = re.sub(r"\D", "", match)
                    if 13 <= len(clean_digits) <= 19:
                        findings.append({
                            "finding_id": generate_storage_id(),
                            "category": FindingCategory.CREDIT_CARD.value,
                            "subcategory": "payment_card",
                            "severity": Severity.CRITICAL.value,
                            "confidence": 0.98,
                            "dataset_id": dataset_id,
                            "record_index": idx,
                            "field_name": str(col_name),
                            "location": location,
                            "evidence": mask_credit_card(match),
                            "explanation": "Valid payment card number format discovered in dataset cell.",
                            "detection_method": DetectionMethod.REGEX.value,
                            "recommended_action": PolicyDecision.SANITIZE.value,
                            "review_status": ReviewStatus.PENDING.value,
                            "remediation_status": RemediationStatus.PENDING.value,
                            "created_at": now.isoformat(),
                        })

                # 2. Personal ID / SSN (CRITICAL)
                ssn_matches = SSN_REGEX.findall(val_str)
                for match in ssn_matches:
                    findings.append({
                        "finding_id": generate_storage_id(),
                        "category": FindingCategory.PERSONAL_ID.value,
                        "subcategory": "us_ssn",
                        "severity": Severity.CRITICAL.value,
                        "confidence": 0.95,
                        "dataset_id": dataset_id,
                        "record_index": idx,
                        "field_name": str(col_name),
                        "location": location,
                        "evidence": mask_generic(match, visible_chars=0),
                        "explanation": "Social Security Number (SSN) pattern discovered.",
                        "detection_method": DetectionMethod.REGEX.value,
                        "recommended_action": PolicyDecision.SANITIZE.value,
                        "review_status": ReviewStatus.PENDING.value,
                        "remediation_status": RemediationStatus.PENDING.value,
                        "created_at": now.isoformat(),
                    })

                # 3. Email (HIGH)
                email_matches = EMAIL_REGEX.findall(val_str)
                for match in email_matches:
                    findings.append({
                        "finding_id": generate_storage_id(),
                        "category": FindingCategory.EMAIL.value,
                        "subcategory": "email_address",
                        "severity": Severity.HIGH.value,
                        "confidence": 0.95,
                        "dataset_id": dataset_id,
                        "record_index": idx,
                        "field_name": str(col_name),
                        "location": location,
                        "evidence": mask_email(match),
                        "explanation": "Personal email address discovered.",
                        "detection_method": DetectionMethod.REGEX.value,
                        "recommended_action": PolicyDecision.SANITIZE.value,
                        "review_status": ReviewStatus.PENDING.value,
                        "remediation_status": RemediationStatus.PENDING.value,
                        "created_at": now.isoformat(),
                    })

                # 4. Phone Number (HIGH)
                phone_matches = PHONE_REGEX.findall(val_str)
                for match in phone_matches:
                    # Ignore short number matches that aren't phone numbers
                    digits = re.sub(r"\D", "", match)
                    if 10 <= len(digits) <= 15:
                        findings.append({
                            "finding_id": generate_storage_id(),
                            "category": FindingCategory.PHONE.value,
                            "subcategory": "phone_number",
                            "severity": Severity.HIGH.value,
                            "confidence": 0.90,
                            "dataset_id": dataset_id,
                            "record_index": idx,
                            "field_name": str(col_name),
                            "location": location,
                            "evidence": mask_phone(match),
                            "explanation": "Telephone or mobile number pattern discovered.",
                            "detection_method": DetectionMethod.REGEX.value,
                            "recommended_action": PolicyDecision.SANITIZE.value,
                            "review_status": ReviewStatus.PENDING.value,
                            "remediation_status": RemediationStatus.PENDING.value,
                            "created_at": now.isoformat(),
                        })

                # 5. Address (MEDIUM)
                addr_matches = ADDRESS_REGEX.findall(val_str)
                for match in addr_matches:
                    # Ignore computer/hardware storage drives (e.g. "USB 3.2 solid state drive")
                    if re.search(r"(?:solid\s+state|hard|flash|pen|thumb|usb|disk|disc|dvd|cd)\s+drive", match, re.IGNORECASE):
                        continue
                    findings.append({
                        "finding_id": generate_storage_id(),
                        "category": FindingCategory.ADDRESS.value,
                        "subcategory": "physical_address",
                        "severity": Severity.MEDIUM.value,
                        "confidence": 0.85,
                        "dataset_id": dataset_id,
                        "record_index": idx,
                        "field_name": str(col_name),
                        "location": location,
                        "evidence": truncate_evidence(mask_generic(match, visible_chars=4)),
                        "explanation": "Physical street address pattern discovered.",
                        "detection_method": DetectionMethod.HEURISTIC.value,
                        "recommended_action": PolicyDecision.SANITIZE.value,
                        "review_status": ReviewStatus.PENDING.value,
                        "remediation_status": RemediationStatus.PENDING.value,
                        "created_at": now.isoformat(),
                    })

        logger.info("Privacy scan for %s completed with %d findings", dataset_id, len(findings))
        return findings
