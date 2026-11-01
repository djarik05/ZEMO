import re
import math
from typing import Any

from core.database.models.dataset import PrivacyRuleType

from detectors.base import BaseDetector, DetectionContext, Finding
from core.database.models.dataset import (
    FindingCategory,
    FindingSeverity,
    FindingConfidence,
    SecretsRuleType,
)


class SecretsDetector(BaseDetector):
    name = "secrets"
    version = "1.0.0"
    category = FindingCategory.SECRETS

    API_KEY_PATTERNS = {
        "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "aws_secret_key": re.compile(r"\b[a-zA-Z0-9/+=]{40}\b"),
        "github_token": re.compile(r"\bgh[ps]_[a-zA-Z0-9]{36}\b"),
        "gitlab_token": re.compile(r"\bglpat-[a-zA-Z0-9\-_]{20}\b"),
        "slack_token": re.compile(r"\bxox[baprs]-[a-zA-Z0-9-]{10,}\b"),
        "stripe_key": re.compile(r"\bsk_live_[a-zA-Z0-9]{24}\b"),
        "stripe_publishable": re.compile(r"\bpk_live_[a-zA-Z0-9]{24}\b"),
        "openai_key": re.compile(r"\bsk-[a-zA-Z0-9]{48}\b"),
        "anthropic_key": re.compile(r"\bsk-ant-[a-zA-Z0-9\-_]{95}\b"),
        "google_api_key": re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
        "azure_key": re.compile(r"\b[a-f0-9]{32}\b"),
        "generic_api_key": re.compile(r"\b[a-zA-Z0-9_\-]{32,}\b"),
    }

    PASSWORD_PATTERNS = [
        re.compile(r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?"),
        re.compile(r"(?i)(?:secret|token)\s*[:=]\s*['\"]?([^\s'\"]{16,})['\"]?"),
    ]

    PRIVATE_KEY_PATTERNS = [
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"-----BEGIN (?:PGP )?PRIVATE KEY BLOCK-----"),
    ]

    CLOUD_CREDENTIAL_PATTERNS = {
        "aws_session_token": re.compile(r"\b[A-Za-z0-9/+=]{100,}\b"),
        "gcp_service_account": re.compile(r'"type":\s*"service_account"'),
        "azure_client_secret": re.compile(r"(?i)client[_-]?secret\s*[:=]\s*['\"]?([^\s'\"]{20,})['\"]?"),
    }

    DATABASE_CONNECTION_PATTERNS = [
        re.compile(r"(?i)(?:postgres(?:ql)?|mysql|mssql|oracle|mongodb|redis)://[^:\s]+:[^@\s]+@[^/\s]+"),
        re.compile(r"(?i)(?:server|host|database|uid|pwd|password)\s*=\s*[^;]+"),
        re.compile(r"(?i)jdbc:(?:postgresql|mysql|sqlserver|oracle):[^;]+"),
    ]

    HIGH_ENTROPY_THRESHOLD = 4.5
    MIN_ENTROPY_LENGTH = 20

    def detect(self, context: DetectionContext) -> list[Finding]:
        findings = []

        findings.extend(self._detect_api_keys(context))
        findings.extend(self._detect_passwords(context))
        findings.extend(self._detect_private_keys(context))
        findings.extend(self._detect_cloud_credentials(context))
        findings.extend(self._detect_database_connections(context))
        findings.extend(self._detect_high_entropy_strings(context))

        return findings

    def _detect_api_keys(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records:
            return findings

        for row_idx, record in enumerate(context.records):
            for col_name, value in record.items():
                if not isinstance(value, str):
                    continue
                for key_type, pattern in self.API_KEY_PATTERNS.items():
                    matches = pattern.findall(value)
                    for match in matches:
                        severity = FindingSeverity.CRITICAL
                        confidence = FindingConfidence.HIGH
                        if key_type == "generic_api_key":
                            severity = FindingSeverity.HIGH
                            confidence = FindingConfidence.MEDIUM

                        finding = self._create_finding(
                            rule_id=SecretsRuleType.API_KEY.value,
                            severity=severity,
                            confidence=confidence,
                            title=f"API key detected ({key_type})",
                            description=f"Possible {key_type} found in column '{col_name}'",
                            row_number=row_idx + 1,
                            column_name=col_name,
                            evidence=match,
                            masked_evidence=self._mask_sensitive_value(match),
                            remediation_guidance="API keys are highly sensitive credentials. Rotate the key immediately and remove from dataset.",
                            metadata={"key_type": key_type},
                        )
                        findings.append(finding)
        return findings

    def _detect_passwords(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records:
            return findings

        for row_idx, record in enumerate(context.records):
            for col_name, value in record.items():
                if not isinstance(value, str):
                    continue
                for pattern in self.PASSWORD_PATTERNS:
                    matches = pattern.findall(value)
                    for match in matches:
                        finding = self._create_finding(
                            rule_id=SecretsRuleType.PASSWORD.value,
                            severity=FindingSeverity.CRITICAL,
                            confidence=FindingConfidence.HIGH,
                            title="Password or secret detected",
                            description=f"Possible password/secret assignment in column '{col_name}'",
                            row_number=row_idx + 1,
                            column_name=col_name,
                            evidence=match,
                            masked_evidence=self._mask_sensitive_value(match),
                            remediation_guidance="Passwords and secrets should never be stored in datasets. Remove and rotate credentials.",
                        )
                        findings.append(finding)
        return findings

    def _detect_private_keys(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records:
            return findings

        for row_idx, record in enumerate(context.records):
            for col_name, value in record.items():
                if not isinstance(value, str):
                    continue
                for pattern in self.PRIVATE_KEY_PATTERNS:
                    if pattern.search(value):
                        finding = self._create_finding(
                            rule_id=SecretsRuleType.PRIVATE_KEY.value,
                            severity=FindingSeverity.CRITICAL,
                            confidence=FindingConfidence.HIGH,
                            title="Private key detected",
                            description=f"Private key material found in column '{col_name}'",
                            row_number=row_idx + 1,
                            column_name=col_name,
                            evidence=value[:200] + "..." if len(value) > 200 else value,
                            masked_evidence="[PRIVATE KEY REDACTED]",
                            remediation_guidance="Private keys are extremely sensitive. Revoke and regenerate immediately. Never store in datasets.",
                        )
                        findings.append(finding)
        return findings

    def _detect_cloud_credentials(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records:
            return findings

        for row_idx, record in enumerate(context.records):
            for col_name, value in record.items():
                if not isinstance(value, str):
                    continue
                for cred_type, pattern in self.CLOUD_CREDENTIAL_PATTERNS.items():
                    matches = pattern.findall(value)
                    for match in matches:
                        finding = self._create_finding(
                            rule_id=SecretsRuleType.CLOUD_CREDENTIAL.value,
                            severity=FindingSeverity.CRITICAL,
                            confidence=FindingConfidence.HIGH,
                            title=f"Cloud credential detected ({cred_type})",
                            description=f"Possible {cred_type} found in column '{col_name}'",
                            row_number=row_idx + 1,
                            column_name=col_name,
                            evidence=match if isinstance(match, str) else str(match),
                            masked_evidence=self._mask_sensitive_value(str(match)),
                            remediation_guidance="Cloud credentials provide access to infrastructure. Revoke and rotate immediately.",
                            metadata={"credential_type": cred_type},
                        )
                        findings.append(finding)
        return findings

    def _detect_database_connections(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records:
            return findings

        for row_idx, record in enumerate(context.records):
            for col_name, value in record.items():
                if not isinstance(value, str):
                    continue
                for pattern in self.DATABASE_CONNECTION_PATTERNS:
                    matches = pattern.findall(value)
                    for match in matches:
                        finding = self._create_finding(
                            rule_id=SecretsRuleType.DATABASE_CONNECTION_STRING.value,
                            severity=FindingSeverity.CRITICAL,
                            confidence=FindingConfidence.HIGH,
                            title="Database connection string with credentials",
                            description=f"Database connection string with embedded credentials in column '{col_name}'",
                            row_number=row_idx + 1,
                            column_name=col_name,
                            evidence=match,
                            masked_evidence=self._mask_sensitive_value(match),
                            remediation_guidance="Database connection strings with credentials are highly sensitive. Use secure credential management instead.",
                        )
                        findings.append(finding)
        return findings

    def _detect_high_entropy_strings(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records:
            return findings

        for row_idx, record in enumerate(context.records):
            for col_name, value in record.items():
                if not isinstance(value, str) or len(value) < self.MIN_ENTROPY_LENGTH:
                    continue
                entropy = self._calculate_entropy(value)
                if entropy >= self.HIGH_ENTROPY_THRESHOLD:
                    finding = self._create_finding(
                        rule_id=SecretsRuleType.HIGH_ENTROPY_STRING.value,
                        severity=FindingSeverity.MEDIUM,
                        confidence=FindingConfidence.LOW,
                        title="High entropy string detected",
                        description=f"String with high entropy ({entropy:.2f}) in column '{col_name}' may be a secret",
                        row_number=row_idx + 1,
                        column_name=col_name,
                        evidence=value,
                        masked_evidence=self._mask_sensitive_value(value),
                        remediation_guidance="High entropy strings may be encoded secrets. Review manually to confirm.",
                        metadata={"entropy": entropy},
                    )
                    findings.append(finding)
        return findings

    def _calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        entropy = 0.0
        length = len(text)
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        return entropy