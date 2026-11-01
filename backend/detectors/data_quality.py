import re
from collections import Counter
from typing import Any

from detectors.base import BaseDetector, DetectionContext, Finding
from core.database.models.dataset import (
    FindingCategory,
    FindingSeverity,
    FindingConfidence,
    DataQualityRuleType,
)


class DataQualityDetector(BaseDetector):
    name = "data_quality"
    version = "1.0.0"
    category = FindingCategory.DATA_QUALITY

    DEFAULT_THRESHOLDS = {
        "missing_value_pct": 50.0,
        "duplicate_pct": 10.0,
        "repetitive_value_pct": 90.0,
        "max_null_per_column": 0.8,
        "min_unique_ratio": 0.01,
    }

    def __init__(self):
        super().__init__()
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()

    def detect(self, context: DetectionContext) -> list[Finding]:
        findings = []

        if not context.records:
            return findings

        findings.extend(self._detect_missing_values(context))
        findings.extend(self._detect_duplicates(context))
        findings.extend(self._detect_unexpected_data_types(context))
        findings.extend(self._detect_structural_inconsistencies(context))
        findings.extend(self._detect_malformed_records(context))
        findings.extend(self._detect_repetitive_values(context))
        findings.extend(self._detect_schema_inconsistencies(context))

        return findings

    def _detect_missing_values(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records or not context.columns:
            return findings

        total_records = len(context.records)

        for col_name in context.columns:
            null_count = 0
            empty_count = 0

            for record in context.records:
                value = record.get(col_name)
                if value is None:
                    null_count += 1
                elif isinstance(value, str) and value.strip() == "":
                    empty_count += 1

            total_missing = null_count + empty_count
            missing_pct = (total_missing / total_records * 100) if total_records > 0 else 0

            if missing_pct >= self.thresholds["missing_value_pct"]:
                severity = FindingSeverity.HIGH if missing_pct >= 80 else FindingSeverity.MEDIUM
                finding = self._create_finding(
                    rule_id=DataQualityRuleType.MISSING_VALUES.value,
                    severity=severity,
                    confidence=FindingConfidence.HIGH,
                    title=f"High missing value rate in column '{col_name}'",
                    description=f"Column '{col_name}' has {missing_pct:.1f}% missing/empty values ({total_missing}/{total_records})",
                    column_name=col_name,
                    evidence=f"null_count={null_count}, empty_count={empty_count}, total={total_records}",
                    masked_evidence=f"missing_pct={missing_pct:.1f}%",
                    remediation_guidance=f"Column '{col_name}' has excessive missing values. Consider imputation, removal, or investigation of data collection issues.",
                    metadata={
                        "null_count": null_count,
                        "empty_count": empty_count,
                        "total_records": total_records,
                        "missing_percentage": missing_pct,
                    },
                )
                findings.append(finding)

            elif missing_pct > 0:
                finding = self._create_finding(
                    rule_id=DataQualityRuleType.MISSING_VALUES.value,
                    severity=FindingSeverity.INFORMATIONAL,
                    confidence=FindingConfidence.MEDIUM,
                    title=f"Missing values detected in column '{col_name}'",
                    description=f"Column '{col_name}' has {missing_pct:.1f}% missing/empty values",
                    column_name=col_name,
                    evidence=f"null_count={null_count}, empty_count={empty_count}",
                    masked_evidence=f"missing_pct={missing_pct:.1f}%",
                    remediation_guidance="Monitor missing value rate. Consider imputation strategies if rate increases.",
                    metadata={
                        "null_count": null_count,
                        "empty_count": empty_count,
                        "total_records": total_records,
                        "missing_percentage": missing_pct,
                    },
                )
                findings.append(finding)

        return findings

    def _detect_duplicates(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records:
            return findings

        record_tuples = []
        for record in context.records:
            if context.columns:
                record_tuple = tuple(record.get(col) for col in context.columns)
            else:
                record_tuple = tuple(sorted(record.items()))
            record_tuples.append(record_tuple)

        duplicate_counts = Counter(record_tuples)
        duplicate_records = sum(count - 1 for count in duplicate_counts.values() if count > 1)
        duplicate_pct = (duplicate_records / len(context.records) * 100) if context.records else 0

        if duplicate_records > 0:
            severity = FindingSeverity.HIGH if duplicate_pct >= self.thresholds["duplicate_pct"] else FindingSeverity.MEDIUM
            finding = self._create_finding(
                rule_id=DataQualityRuleType.DUPLICATE_RECORDS.value,
                severity=severity,
                confidence=FindingConfidence.HIGH,
                title="Duplicate records detected",
                description=f"Found {duplicate_records} duplicate records ({duplicate_pct:.1f}% of dataset)",
                evidence=f"duplicate_records={duplicate_records}, total={len(context.records)}",
                masked_evidence=f"duplicate_pct={duplicate_pct:.1f}%",
                remediation_guidance="Remove or deduplicate records. Investigate data collection process for duplicate sources.",
                metadata={
                    "duplicate_records": duplicate_records,
                    "total_records": len(context.records),
                    "duplicate_percentage": duplicate_pct,
                },
            )
            findings.append(finding)

        return findings

    def _detect_unexpected_data_types(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records or not context.columns:
            return findings

        for col_name in context.columns:
            type_counts = Counter()
            for record in context.records:
                value = record.get(col_name)
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    type_counts["null/empty"] += 1
                else:
                    type_counts[type(value).__name__] += 1

            if len(type_counts) > 1:
                finding = self._create_finding(
                    rule_id=DataQualityRuleType.UNEXPECTED_DATA_TYPE.value,
                    severity=FindingSeverity.MEDIUM,
                    confidence=FindingConfidence.MEDIUM,
                    title=f"Mixed data types in column '{col_name}'",
                    description=f"Column '{col_name}' contains mixed types: {dict(type_counts)}",
                    column_name=col_name,
                    evidence=str(dict(type_counts)),
                    masked_evidence=f"{len(type_counts)} types detected",
                    remediation_guidance=f"Column '{col_name}' has mixed data types. Standardize to a single type for consistent processing.",
                    metadata={"type_counts": dict(type_counts)},
                )
                findings.append(finding)

        return findings

    def _detect_structural_inconsistencies(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records:
            return findings

        expected_keys = set(context.columns) if context.columns else set()
        inconsistent_count = 0

        for row_idx, record in enumerate(context.records):
            record_keys = set(record.keys())
            missing_keys = expected_keys - record_keys
            extra_keys = record_keys - expected_keys

            if missing_keys or extra_keys:
                inconsistent_count += 1
                if inconsistent_count <= 10:
                    finding = self._create_finding(
                        rule_id=DataQualityRuleType.STRUCTURAL_INCONSISTENCY.value,
                        severity=FindingSeverity.MEDIUM,
                        confidence=FindingConfidence.MEDIUM,
                        title=f"Structural inconsistency in row {row_idx + 1}",
                        description=f"Row {row_idx + 1} has missing keys: {missing_keys}, extra keys: {extra_keys}",
                        row_number=row_idx + 1,
                        column_name=", ".join(missing_keys | extra_keys) if missing_keys | extra_keys else None,
                        evidence=f"missing={missing_keys}, extra={extra_keys}",
                        masked_evidence="structural mismatch",
                        remediation_guidance="Ensure all records have consistent structure. Missing keys may indicate data extraction issues.",
                        metadata={
                            "missing_keys": list(missing_keys),
                            "extra_keys": list(extra_keys),
                        },
                    )
                    findings.append(finding)

        if inconsistent_count > 10:
            finding = self._create_finding(
                rule_id=DataQualityRuleType.STRUCTURAL_INCONSISTENCY.value,
                severity=FindingSeverity.HIGH,
                confidence=FindingConfidence.HIGH,
                title="High rate of structural inconsistencies",
                description=f"{inconsistent_count} out of {len(context.records)} rows have structural issues",
                evidence=f"inconsistent_rows={inconsistent_count}, total={len(context.records)}",
                masked_evidence=f"inconsistency_rate={inconsistent_count/len(context.records)*100:.1f}%",
                remediation_guidance="High rate of structural inconsistencies suggests systematic data extraction issues. Review data source.",
                metadata={
                    "inconsistent_count": inconsistent_count,
                    "total_records": len(context.records),
                },
            )
            findings.append(finding)

        return findings

    def _detect_malformed_records(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records:
            return findings

        malformed_count = 0
        for row_idx, record in enumerate(context.records):
            issues = []
            for col_name, value in record.items():
                if isinstance(value, str):
                    if self._looks_like_truncated_json(value):
                        issues.append("possible_truncated_json")
                    if self._looks_like_unescaped_quotes(value):
                        issues.append("unescaped_quotes")
                    if self._has_control_characters(value):
                        issues.append("control_characters")

            if issues:
                malformed_count += 1
                if malformed_count <= 10:
                    finding = self._create_finding(
                        rule_id=DataQualityRuleType.MALFORMED_RECORDS.value,
                        severity=FindingSeverity.MEDIUM,
                        confidence=FindingConfidence.MEDIUM,
                        title=f"Malformed data in row {row_idx + 1}",
                        description=f"Row {row_idx + 1} has issues: {', '.join(issues)}",
                        row_number=row_idx + 1,
                        evidence=f"issues={issues}",
                        masked_evidence="; ".join(issues),
                        remediation_guidance="Review data parsing/extraction process. Malformed data may indicate encoding or parsing issues.",
                        metadata={"issues": issues},
                    )
                    findings.append(finding)

        if malformed_count > 10:
            finding = self._create_finding(
                rule_id=DataQualityRuleType.MALFORMED_RECORDS.value,
                severity=FindingSeverity.HIGH,
                confidence=FindingConfidence.HIGH,
                title="High rate of malformed records",
                description=f"{malformed_count} out of {len(context.records)} rows have data quality issues",
                evidence=f"malformed={malformed_count}, total={len(context.records)}",
                masked_evidence=f"malformed_rate={malformed_count/len(context.records)*100:.1f}%",
                remediation_guidance="High rate of malformed records suggests systematic parsing/extraction issues. Review data pipeline.",
                metadata={
                    "malformed_count": malformed_count,
                    "total_records": len(context.records),
                },
            )
            findings.append(finding)

        return findings

    def _looks_like_truncated_json(self, value: str) -> bool:
        if not value or len(value) < 10:
            return False
        stripped = value.strip()
        return (stripped.startswith("{") and not stripped.endswith("}")) or \
               (stripped.startswith("[") and not stripped.endswith("]"))

    def _looks_like_unescaped_quotes(self, value: str) -> bool:
        quote_count = value.count('"')
        return quote_count % 2 != 0 and quote_count > 2

    def _has_control_characters(self, value: str) -> bool:
        return any(ord(c) < 32 and c not in '\t\n\r' for c in value)

    def _detect_repetitive_values(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.records or not context.columns:
            return findings

        for col_name in context.columns:
            values = [record.get(col_name) for record in context.records if record.get(col_name) is not None]
            if not values:
                continue

            value_counts = Counter(str(v) for v in values)
            total = len(values)
            most_common = value_counts.most_common(1)[0]
            top_value, top_count = most_common
            top_pct = (top_count / total * 100) if total > 0 else 0

            if top_pct >= self.thresholds["repetitive_value_pct"]:
                finding = self._create_finding(
                    rule_id=DataQualityRuleType.REPETITIVE_VALUES.value,
                    severity=FindingSeverity.MEDIUM,
                    confidence=FindingConfidence.MEDIUM,
                    title=f"Highly repetitive values in column '{col_name}'",
                    description=f"Value '{top_value}' appears in {top_pct:.1f}% of records ({top_count}/{total})",
                    column_name=col_name,
                    evidence=f"top_value={top_value}, count={top_count}, total={total}",
                    masked_evidence=f"repetitive_pct={top_pct:.1f}%",
                    remediation_guidance=f"Column '{col_name}' is highly repetitive. May indicate default values, missing data, or categorical imbalance.",
                    metadata={
                        "top_value": top_value,
                        "top_count": top_count,
                        "total_count": total,
                        "repetitive_percentage": top_pct,
                    },
                )
                findings.append(finding)

        return findings

    def _detect_schema_inconsistencies(self, context: DetectionContext) -> list[Finding]:
        findings = []
        if not context.columns or not context.schema_metadata:
            return findings

        schema_columns = set(context.schema_metadata.get("schema", {}).keys())
        actual_columns = set(context.columns)

        missing_in_schema = actual_columns - schema_columns
        extra_in_schema = schema_columns - actual_columns

        if missing_in_schema:
            finding = self._create_finding(
                rule_id=DataQualityRuleType.SCHEMA_INCONSISTENCY.value,
                severity=FindingSeverity.MEDIUM,
                confidence=FindingConfidence.HIGH,
                title="Columns present in data but missing from schema",
                description=f"Columns in data but not in schema: {', '.join(sorted(missing_in_schema))}",
                column_name=", ".join(sorted(missing_in_schema)),
                evidence=f"missing_from_schema={missing_in_schema}",
                masked_evidence=f"{len(missing_in_schema)} columns missing",
                remediation_guidance="Update schema metadata to include all actual columns, or remove unexpected columns from data.",
                metadata={"missing_columns": list(missing_in_schema)},
            )
            findings.append(finding)

        if extra_in_schema:
            finding = self._create_finding(
                rule_id=DataQualityRuleType.SCHEMA_INCONSISTENCY.value,
                severity=FindingSeverity.LOW,
                confidence=FindingConfidence.MEDIUM,
                title="Columns in schema but missing from data",
                description=f"Columns defined in schema but not in data: {', '.join(sorted(extra_in_schema))}",
                column_name=", ".join(sorted(extra_in_schema)),
                evidence=f"extra_in_schema={extra_in_schema}",
                masked_evidence=f"{len(extra_in_schema)} columns extra",
                remediation_guidance="Schema may be outdated. Remove unused columns from schema or investigate missing data.",
                metadata={"extra_columns": list(extra_in_schema)},
            )
            findings.append(finding)

        return findings