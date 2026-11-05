"""
Security analyzer service.

Analyzes dataset records for security threats:
- Prompt injection & jailbreak triggers (HIGH)
- Suspicious exploit payloads: SQL Injection, XSS, Command Injection (HIGH / MEDIUM)
- Malformed records: Extreme length, non-printable binary content, encoding anomalies (MEDIUM)
"""

from __future__ import annotations

import re
import string
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
from app.utils.masking import truncate_evidence

logger = get_logger("security_analyzer")

# Prompt Injection Patterns
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"(?i)\bignore\s+(?:all\s+)?previous\s+instructions\b"),
    re.compile(r"(?i)\bdisregard\s+(?:all\s+)?(?:the\s+)?above\b"),
    re.compile(r"(?i)\byou\s+are\s+now\b"),
    re.compile(r"(?i)\bsystem\s*prompt\s*:"),
    re.compile(r"(?i)\bjailbreak\b"),
    re.compile(r"(?i)\bDAN\s+mode\b"),
    re.compile(r"\n\nHuman:"),
    re.compile(r"<\|im_start\|>system"),
    re.compile(r"```\s*(?:system|assistant|instruction|prompt)", re.IGNORECASE),
]

# Suspicious Payloads (SQL Injection, XSS, Command Injection)
SQL_INJECTION_PATTERNS = [
    re.compile(r"(?i)(?:;\s*DROP\s+TABLE\b|UNION\s+SELECT\b|\bOR\s+1\s*=\s*1\b|\b1\s*=\s*1\s*--|\bEXEC\s*\(\s*@)"),
]

XSS_PATTERNS = [
    re.compile(r"(?i)(?:<script\b[^>]*>|javascript:\s*|\bonerror\s*=\s*|\bonload\s*=\s*)"),
]

COMMAND_INJECTION_PATTERNS = [
    re.compile(r"(?:\|\s*rm\s+-rf\b|;\s*cat\s+/etc/passwd\b|\$\([^)]+\)|\`[^`]+\`)"),
]

PRINTABLE_SET = set(string.printable)


class SecurityAnalyzer:
    """Service for detecting prompt injections, exploit payloads, and malformed records."""

    def analyze(self, dataset_id: str, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Analyze all cells in DataFrame for prompt injection, suspicious exploit payloads,
        and malformed records.
        """
        findings: list[dict[str, Any]] = []
        now = utcnow()

        for col_name in df.columns:
            series = df[col_name]
            for row_idx, val in series.items():
                if val is None or pd.isna(val):
                    continue

                val_str = str(val)
                if not val_str.strip():
                    continue

                idx = int(row_idx) if isinstance(row_idx, (int, float)) else 0
                location = f"row:{idx},col:{col_name}"

                # 1. Prompt Injection (Severity: HIGH)
                for pat in PROMPT_INJECTION_PATTERNS:
                    match = pat.search(val_str)
                    if match:
                        findings.append({
                            "finding_id": generate_storage_id(),
                            "category": FindingCategory.PROMPT_INJECTION.value,
                            "subcategory": "instruction_override",
                            "severity": Severity.HIGH.value,
                            "confidence": 0.92,
                            "dataset_id": dataset_id,
                            "record_index": idx,
                            "field_name": str(col_name),
                            "location": location,
                            "evidence": truncate_evidence(match.group(0)),
                            "explanation": "Potential LLM prompt injection or system prompt override pattern detected.",
                            "detection_method": DetectionMethod.REGEX.value,
                            "recommended_action": PolicyDecision.BLOCK.value,
                            "review_status": ReviewStatus.PENDING.value,
                            "remediation_status": RemediationStatus.PENDING.value,
                            "created_at": now.isoformat(),
                        })
                        break

                # 2. Suspicious Content: SQL Injection (Severity: HIGH)
                sql_found = False
                for pat in SQL_INJECTION_PATTERNS:
                    match = pat.search(val_str)
                    if match:
                        findings.append({
                            "finding_id": generate_storage_id(),
                            "category": FindingCategory.SUSPICIOUS_CONTENT.value,
                            "subcategory": "sql_injection",
                            "severity": Severity.HIGH.value,
                            "confidence": 0.95,
                            "dataset_id": dataset_id,
                            "record_index": idx,
                            "field_name": str(col_name),
                            "location": location,
                            "evidence": truncate_evidence(match.group(0)),
                            "explanation": "SQL injection exploit pattern detected.",
                            "detection_method": DetectionMethod.REGEX.value,
                            "recommended_action": PolicyDecision.REVIEW.value,
                            "review_status": ReviewStatus.PENDING.value,
                            "remediation_status": RemediationStatus.PENDING.value,
                            "created_at": now.isoformat(),
                        })
                        sql_found = True
                        break

                # 3. Suspicious Content: XSS (Severity: HIGH)
                if not sql_found:
                    xss_found = False
                    for pat in XSS_PATTERNS:
                        match = pat.search(val_str)
                        if match:
                            findings.append({
                                "finding_id": generate_storage_id(),
                                "category": FindingCategory.SUSPICIOUS_CONTENT.value,
                                "subcategory": "cross_site_scripting",
                                "severity": Severity.HIGH.value,
                                "confidence": 0.95,
                                "dataset_id": dataset_id,
                                "record_index": idx,
                                "field_name": str(col_name),
                                "location": location,
                                "evidence": truncate_evidence(match.group(0)),
                                "explanation": "Cross-site scripting (XSS) payload detected.",
                                "detection_method": DetectionMethod.REGEX.value,
                                "recommended_action": PolicyDecision.REVIEW.value,
                                "review_status": ReviewStatus.PENDING.value,
                                "remediation_status": RemediationStatus.PENDING.value,
                                "created_at": now.isoformat(),
                            })
                            xss_found = True
                            break

                    # 4. Suspicious Content: Command Injection (Severity: MEDIUM)
                    if not xss_found:
                        for pat in COMMAND_INJECTION_PATTERNS:
                            match = pat.search(val_str)
                            if match:
                                findings.append({
                                    "finding_id": generate_storage_id(),
                                    "category": FindingCategory.SUSPICIOUS_CONTENT.value,
                                    "subcategory": "command_injection",
                                    "severity": Severity.MEDIUM.value,
                                    "confidence": 0.85,
                                    "dataset_id": dataset_id,
                                    "record_index": idx,
                                    "field_name": str(col_name),
                                    "location": location,
                                    "evidence": truncate_evidence(match.group(0)),
                                    "explanation": "Operating system shell command injection pattern detected.",
                                    "detection_method": DetectionMethod.REGEX.value,
                                    "recommended_action": PolicyDecision.REVIEW.value,
                                    "review_status": ReviewStatus.PENDING.value,
                                    "remediation_status": RemediationStatus.PENDING.value,
                                    "created_at": now.isoformat(),
                                })
                                break

                # 5. Malformed Record (Severity: MEDIUM)
                # Check for extreme length
                if len(val_str) > 10000:
                    findings.append({
                        "finding_id": generate_storage_id(),
                        "category": FindingCategory.MALFORMED_RECORD.value,
                        "subcategory": "oversized_content",
                        "severity": Severity.MEDIUM.value,
                        "confidence": 1.0,
                        "dataset_id": dataset_id,
                        "record_index": idx,
                        "field_name": str(col_name),
                        "location": location,
                        "evidence": f"String length: {len(val_str)} chars",
                        "explanation": f"Record value exceeds 10,000 characters ({len(val_str)} chars), which may cause token overflow or memory exhaustion.",
                        "detection_method": DetectionMethod.STRUCTURAL.value,
                        "recommended_action": PolicyDecision.REVIEW.value,
                        "review_status": ReviewStatus.PENDING.value,
                        "remediation_status": RemediationStatus.PENDING.value,
                        "created_at": now.isoformat(),
                    })
                elif len(val_str) > 50:
                    # Check ratio of non-printable characters
                    non_printable_count = sum(1 for c in val_str if c not in PRINTABLE_SET)
                    ratio = non_printable_count / len(val_str)
                    if ratio > 0.30:
                        findings.append({
                            "finding_id": generate_storage_id(),
                            "category": FindingCategory.MALFORMED_RECORD.value,
                            "subcategory": "binary_content",
                            "severity": Severity.MEDIUM.value,
                            "confidence": 0.90,
                            "dataset_id": dataset_id,
                            "record_index": idx,
                            "field_name": str(col_name),
                            "location": location,
                            "evidence": f"Non-printable ratio: {ratio:.1%}",
                            "explanation": "High ratio of non-printable/binary characters detected, indicating corrupted or binary payload.",
                            "detection_method": DetectionMethod.STRUCTURAL.value,
                            "recommended_action": PolicyDecision.REVIEW.value,
                            "review_status": ReviewStatus.PENDING.value,
                            "remediation_status": RemediationStatus.PENDING.value,
                            "created_at": now.isoformat(),
                        })

        logger.info("Security analysis for %s completed with %d findings", dataset_id, len(findings))
        return findings
