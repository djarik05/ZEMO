import re
from typing import Any

from detectors.base import BaseDetector, DetectionContext, Finding
from core.database.models.dataset import (
    FindingCategory,
    FindingSeverity,
    FindingConfidence,
    PrivacyRuleType,
)


class PrivacyDetector(BaseDetector):
    name = "privacy"
    version = "1.0.0"
    category = FindingCategory.PRIVACY

    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )

    PHONE_PATTERNS = [
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"),
        re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        re.compile(r"\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"),
    ]

    GOVERNMENT_ID_PATTERNS = {
        "us_ssn": re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
        "us_itin": re.compile(r"\b9\d{2}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
        "us_passport": re.compile(r"\b[A-Z]{1,2}\d{6,8}\b"),
        "uk_nino": re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b"),
        "eu_vat": re.compile(r"\b[A-Z]{2}\d{2,12}\b"),
        "ca_sin": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3}\b"),
    }

    FINANCIAL_PATTERNS = {
        "credit_card": re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"),
        "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b"),
        "swift_bic": re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b"),
        "bank_account": re.compile(r"\b\d{8,17}\b"),
    }

    SENSITIVE_FIELD_KEYWORDS = {
        "email": ["email", "e-mail", "mail", "email_address", "emailaddress"],
        "phone": ["phone", "telephone", "mobile", "cell", "fax", "contact_number"],
        "name": ["first_name", "last_name", "full_name", "maiden_name", "given_name", "surname"],
        "address": ["address", "street", "city", "state", "zip", "postal", "zipcode", "postcode"],
        "dob": ["dob", "date_of_birth", "birth_date", "birthdate", "age"],
        "id": ["ssn", "social_security", "national_id", "tax_id", "passport", "driver_license", "license_number"],
        "financial": ["credit_card", "cc_number", "card_number", "account_number", "routing_number", "iban", "swift", "bic"],
        "medical": ["diagnosis", "condition", "medication", "prescription", "patient_id", "mrn"],
        "credentials": ["password", "secret", "token", "api_key", "api_secret", "private_key", "certificate"],
    }

    def detect(self, context: DetectionContext) -> list:
        findings = []

        findings.extend(self._detect_emails(context))
        findings.extend(self._detect_phones(context))
        findings.extend(self._detect_government_ids(context))
        findings.extend(self._detect_financial(context))
        findings.extend(self._detect_sensitive_field_names(context))
        findings.extend(self._detect_personal_info_combos(context))

        return findings

    def _detect_emails(self, context: DetectionContext) -> list:
        findings = []
        if not context.records:
            return findings

        for row_idx, record in enumerate(context.records):
            for col_name, value in record.items():
                if not isinstance(value, str):
                    continue
                matches = self.EMAIL_PATTERN.findall(value)
                for email in matches:
                    finding = self._create_finding(
                        rule_id=PrivacyRuleType.EMAIL.value,
                        severity=FindingSeverity.HIGH,
                        confidence=FindingConfidence.HIGH,
                        title="Email address detected",
                        description=f"Email address found in column '{col_name}': {self._mask_email(email)}",
                        row_number=row_idx + 1,
                        column_name=col_name,
                        evidence=email,
                        masked_evidence=self._mask_email(email),
                        remediation_guidance="Consider removing or pseudonymizing email addresses before using this dataset for training.",
                    )
                    findings.append(finding)
        return findings

    def _detect_phones(self, context: DetectionContext) -> list:
        findings = []
        if not context.records:
            return findings

        for row_idx, record in enumerate(context.records):
            for col_name, value in record.items():
                if not isinstance(value, str):
                    continue
                for pattern in self.PHONE_PATTERNS:
                    matches = pattern.findall(value)
                    for match in matches:
                        if isinstance(match, tuple):
                            phone = "".join(match)
                        else:
                            phone = match
                        finding = self._create_finding(
                            rule_id=PrivacyRuleType.PHONE.value,
                            severity=FindingSeverity.HIGH,
                            confidence=FindingConfidence.MEDIUM,
                            title="Phone number detected",
                            description=f"Phone number found in column '{col_name}': {self._mask_phone(phone)}",
                            row_number=row_idx + 1,
                            column_name=col_name,
                            evidence=phone,
                            masked_evidence=self._mask_phone(phone),
                            remediation_guidance="Consider removing or pseudonymizing phone numbers before using this dataset for training.",
                        )
                        findings.append(finding)
        return findings

    def _detect_government_ids(self, context: DetectionContext) -> list:
        findings = []
        if not context.records:
            return findings

        for row_idx, record in enumerate(context.records):
            for col_name, value in record.items():
                if not isinstance(value, str):
                    continue
                for id_type, pattern in self.GOVERNMENT_ID_PATTERNS.items():
                    matches = pattern.findall(value)
                    for match in matches:
                        finding = self._create_finding(
                            rule_id=PrivacyRuleType.GOVERNMENT_ID.value,
                            severity=FindingSeverity.CRITICAL,
                            confidence=FindingConfidence.HIGH,
                            title=f"Government ID detected ({id_type})",
                            description=f"Possible {id_type} found in column '{col_name}': {self._mask_sensitive_value(match)}",
                            row_number=row_idx + 1,
                            column_name=col_name,
                            evidence=match,
                            masked_evidence=self._mask_sensitive_value(match),
                            remediation_guidance="Government identifiers are highly sensitive. Remove or strongly pseudonymize before use.",
                            metadata={"id_type": id_type},
                        )
                        findings.append(finding)
        return findings

    def _detect_financial(self, context: DetectionContext) -> list:
        findings = []
        if not context.records:
            return findings

        for row_idx, record in enumerate(context.records):
            for col_name, value in record.items():
                if not isinstance(value, str):
                    continue
                for fin_type, pattern in self.FINANCIAL_PATTERNS.items():
                    matches = pattern.findall(value)
                    for match in matches:
                        severity = FindingSeverity.CRITICAL if fin_type in ("credit_card", "iban") else FindingSeverity.HIGH
                        finding = self._create_finding(
                            rule_id=PrivacyRuleType.FINANCIAL.value,
                            severity=severity,
                            confidence=FindingConfidence.HIGH,
                            title=f"Financial data detected ({fin_type})",
                            description=f"Possible {fin_type} found in column '{col_name}': {self._mask_sensitive_value(match)}",
                            row_number=row_idx + 1,
                            column_name=col_name,
                            evidence=match,
                            masked_evidence=self._mask_sensitive_value(match),
                            remediation_guidance="Financial data is highly sensitive. Remove or strongly pseudonymize before use.",
                            metadata={"financial_type": fin_type},
                        )
                        findings.append(finding)
        return findings

    def _detect_sensitive_field_names(self, context: DetectionContext) -> list:
        findings = []
        if not context.columns:
            return findings

        for col_name in context.columns:
            col_lower = col_name.lower().replace(" ", "_").replace("-", "_")
            for category, keywords in self.SENSITIVE_FIELD_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in col_lower:
                        severity_map = {
                            "credentials": FindingSeverity.CRITICAL,
                            "medical": FindingSeverity.CRITICAL,
                            "financial": FindingSeverity.HIGH,
                            "id": FindingSeverity.HIGH,
                            "email": FindingSeverity.HIGH,
                            "phone": FindingSeverity.HIGH,
                            "name": FindingSeverity.MEDIUM,
                            "address": FindingSeverity.MEDIUM,
                            "dob": FindingSeverity.MEDIUM,
                        }
                        finding = self._create_finding(
                            rule_id=PrivacyRuleType.SENSITIVE_FIELD_NAME.value,
                            severity=severity_map.get(category, FindingSeverity.MEDIUM),
                            confidence=FindingConfidence.MEDIUM,
                            title=f"Sensitive field name detected ({category})",
                            description=f"Column name '{col_name}' suggests {category} data",
                            column_name=col_name,
                            evidence=col_name,
                            masked_evidence=col_name,
                            remediation_guidance=f"Review column '{col_name}' for {category} data. Apply appropriate pseudonymization or removal.",
                            metadata={"sensitive_category": category},
                        )
                        findings.append(finding)
                        break
        return findings

    def _detect_personal_info_combos(self, context: DetectionContext) -> list:
        findings = []
        if not context.records:
            return findings

        sensitive_columns = set()
        for col_name in context.columns:
            col_lower = col_name.lower().replace(" ", "_").replace("-", "_")
            for category, keywords in self.SENSITIVE_FIELD_KEYWORDS.items():
                if any(keyword in col_lower for keyword in keywords):
                    sensitive_columns.add(col_name)
                    break

        if len(sensitive_columns) >= 2:
            for row_idx, record in enumerate(context.records):
                combo_fields = [col for col in sensitive_columns if col in record and record[col]]
                if len(combo_fields) >= 2:
                    finding = self._create_finding(
                        rule_id=PrivacyRuleType.PERSONAL_INFO_COMBO.value,
                        severity=FindingSeverity.HIGH,
                        confidence=FindingConfidence.MEDIUM,
                        title="Multiple personal information fields in same record",
                        description=f"Record contains multiple sensitive fields: {', '.join(combo_fields)}",
                        row_number=row_idx + 1,
                        column_name=", ".join(combo_fields),
                        evidence=str({k: record[k] for k in combo_fields}),
                        masked_evidence="{...}",
                        remediation_guidance="Multiple personal identifiers in a single record increase re-identification risk. Consider removing or pseudonymizing.",
                        metadata={"combo_fields": combo_fields},
                    )
                    findings.append(finding)
        return findings