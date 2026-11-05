"""
Secret and credential detection service.

Detects hardcoded secrets in dataset content and column headers:
- API Keys (OpenAI, AWS, generic api_key tokens)
- Passwords and credentials
- Bearer tokens and access tokens
- RSA/EC/DSA Private Keys
- Database connection strings with credentials
- Suspicious column names
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
from app.utils.masking import mask_api_key

logger = get_logger("secret_detector")

# Compiled regular expressions for secret detection
API_KEY_PATTERNS = [
    re.compile(r"\b(?:sk|pk|rk|ak|AKIA)[a-zA-Z0-9_\-]{16,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?"),
]

PASSWORD_PATTERN = re.compile(
    r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{4,})['\"]?"
)

ACCESS_TOKEN_PATTERN = re.compile(
    r"(?i)(?:bearer|token|access_token|auth_token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-.]{10,})['\"]?"
)

PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"
)

DB_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?:mysql|postgres|postgresql|mongodb|redis)://[^\s]{10,}"
)

# Suspicious column header keywords
SUSPICIOUS_COLUMNS = {
    "password": FindingCategory.PASSWORD,
    "passwd": FindingCategory.PASSWORD,
    "pwd": FindingCategory.PASSWORD,
    "secret": FindingCategory.API_KEY,
    "api_key": FindingCategory.API_KEY,
    "apikey": FindingCategory.API_KEY,
    "token": FindingCategory.ACCESS_TOKEN,
    "access_token": FindingCategory.ACCESS_TOKEN,
    "credential": FindingCategory.PASSWORD,
    "private_key": FindingCategory.PRIVATE_KEY,
}


class SecretDetector:
    """Service to scan dataset cells and column schema for exposed secrets and credentials."""

    def detect(self, dataset_id: str, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Scan every cell and column name for exposed secrets.
        All secrets are flagged with CRITICAL severity.
        """
        findings: list[dict[str, Any]] = []
        now = utcnow()

        # 1. Inspect column names for suspicious labels
        for col_name in df.columns:
            col_str = str(col_name).lower().strip()
            for susp_key, category in SUSPICIOUS_COLUMNS.items():
                if susp_key in col_str:
                    findings.append({
                        "finding_id": generate_storage_id(),
                        "category": category.value,
                        "subcategory": "suspicious_column_name",
                        "severity": Severity.CRITICAL.value,
                        "confidence": 0.95,
                        "dataset_id": dataset_id,
                        "record_index": None,
                        "field_name": str(col_name),
                        "location": f"column:{col_name}",
                        "evidence": mask_api_key(str(col_name)),
                        "explanation": f"Column name '{col_name}' suggests it holds sensitive credential data.",
                        "detection_method": DetectionMethod.FIELD_NAME.value,
                        "recommended_action": PolicyDecision.BLOCK.value,
                        "review_status": ReviewStatus.PENDING.value,
                        "remediation_status": RemediationStatus.PENDING.value,
                        "created_at": now.isoformat(),
                    })
                    break

        # 2. Inspect cell contents across all columns
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

                # API Key Patterns
                for pat in API_KEY_PATTERNS:
                    matches = pat.findall(val_str)
                    for match in matches:
                        raw_key = match if isinstance(match, str) else match[0]
                        findings.append({
                            "finding_id": generate_storage_id(),
                            "category": FindingCategory.API_KEY.value,
                            "subcategory": "api_key",
                            "severity": Severity.CRITICAL.value,
                            "confidence": 0.98,
                            "dataset_id": dataset_id,
                            "record_index": idx,
                            "field_name": str(col_name),
                            "location": location,
                            "evidence": mask_api_key(raw_key),
                            "explanation": "Hardcoded API key or access secret detected in data.",
                            "detection_method": DetectionMethod.REGEX.value,
                            "recommended_action": PolicyDecision.BLOCK.value,
                            "review_status": ReviewStatus.PENDING.value,
                            "remediation_status": RemediationStatus.PENDING.value,
                            "created_at": now.isoformat(),
                        })

                # Password Pattern
                pwd_matches = PASSWORD_PATTERN.findall(val_str)
                for match in pwd_matches:
                    raw_pwd = match if isinstance(match, str) else match[0]
                    findings.append({
                        "finding_id": generate_storage_id(),
                        "category": FindingCategory.PASSWORD.value,
                        "subcategory": "plaintext_password",
                        "severity": Severity.CRITICAL.value,
                        "confidence": 0.95,
                        "dataset_id": dataset_id,
                        "record_index": idx,
                        "field_name": str(col_name),
                        "location": location,
                        "evidence": mask_api_key(raw_pwd),
                        "explanation": "Plaintext password pattern detected in record.",
                        "detection_method": DetectionMethod.REGEX.value,
                        "recommended_action": PolicyDecision.BLOCK.value,
                        "review_status": ReviewStatus.PENDING.value,
                        "remediation_status": RemediationStatus.PENDING.value,
                        "created_at": now.isoformat(),
                    })

                # Access Token Pattern
                token_matches = ACCESS_TOKEN_PATTERN.findall(val_str)
                for match in token_matches:
                    raw_token = match if isinstance(match, str) else match[0]
                    findings.append({
                        "finding_id": generate_storage_id(),
                        "category": FindingCategory.ACCESS_TOKEN.value,
                        "subcategory": "bearer_token",
                        "severity": Severity.CRITICAL.value,
                        "confidence": 0.95,
                        "dataset_id": dataset_id,
                        "record_index": idx,
                        "field_name": str(col_name),
                        "location": location,
                        "evidence": mask_api_key(raw_token),
                        "explanation": "Authentication access token or Bearer authorization detected.",
                        "detection_method": DetectionMethod.REGEX.value,
                        "recommended_action": PolicyDecision.BLOCK.value,
                        "review_status": ReviewStatus.PENDING.value,
                        "remediation_status": RemediationStatus.PENDING.value,
                        "created_at": now.isoformat(),
                    })

                # Private Key Pattern
                if PRIVATE_KEY_PATTERN.search(val_str):
                    findings.append({
                        "finding_id": generate_storage_id(),
                        "category": FindingCategory.PRIVATE_KEY.value,
                        "subcategory": "rsa_private_key",
                        "severity": Severity.CRITICAL.value,
                        "confidence": 1.0,
                        "dataset_id": dataset_id,
                        "record_index": idx,
                        "field_name": str(col_name),
                        "location": location,
                        "evidence": "-----BEGIN PRIVATE KEY----- ... [TRUNCATED]",
                        "explanation": "Unencrypted cryptographic private key header discovered.",
                        "detection_method": DetectionMethod.REGEX.value,
                        "recommended_action": PolicyDecision.BLOCK.value,
                        "review_status": ReviewStatus.PENDING.value,
                        "remediation_status": RemediationStatus.PENDING.value,
                        "created_at": now.isoformat(),
                    })

                # Database Credential Pattern
                db_matches = DB_CREDENTIAL_PATTERN.findall(val_str)
                for match in db_matches:
                    findings.append({
                        "finding_id": generate_storage_id(),
                        "category": FindingCategory.DB_CREDENTIAL.value,
                        "subcategory": "database_connection_uri",
                        "severity": Severity.CRITICAL.value,
                        "confidence": 0.98,
                        "dataset_id": dataset_id,
                        "record_index": idx,
                        "field_name": str(col_name),
                        "location": location,
                        "evidence": mask_api_key(match),
                        "explanation": "Database connection URI with credentials discovered.",
                        "detection_method": DetectionMethod.REGEX.value,
                        "recommended_action": PolicyDecision.BLOCK.value,
                        "review_status": ReviewStatus.PENDING.value,
                        "remediation_status": RemediationStatus.PENDING.value,
                        "created_at": now.isoformat(),
                    })

        logger.info("Secret scan for %s completed with %d findings", dataset_id, len(findings))
        return findings
