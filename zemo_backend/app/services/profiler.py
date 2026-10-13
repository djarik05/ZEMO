"""
Dataset profiler service.

Computes column-level and dataset-level statistics:
- Data types, null counts, null percentages, uniqueness, sample values
- Duplicate records, empty records, missing value totals
- Quality findings (MISSING_VALUE, DUPLICATE_RECORD, EMPTY_RECORD, SCHEMA_INCONSISTENCY)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import DatasetNotFoundError
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
from app.repositories.dataset_repo import DatasetRepository
from app.schemas.dataset import ColumnProfile, DatasetProfile
from app.utils.masking import truncate_evidence

logger = get_logger("profiler")


class ProfilerService:
    """Service to profile dataset quality, compute distribution statistics, and surface data quality issues."""

    def __init__(self, repo: DatasetRepository | None = None) -> None:
        self.repo = repo or DatasetRepository()

    def profile_dataset(self, dataset_id: str) -> DatasetProfile:
        """
        Load DataFrame from repo, compute per-column stats, detect duplicates, empty records,
        and schema inconsistencies. Generate quality findings and persist profile into metadata.
        """
        meta = self.repo.get_metadata(dataset_id)
        if meta is None:
            raise DatasetNotFoundError(
                f"Dataset with ID '{dataset_id}' not found.",
                detail={"dataset_id": dataset_id},
            )

        df = self.repo.get_dataframe(dataset_id)
        if df is None:
            raise DatasetNotFoundError(
                f"DataFrame for dataset '{dataset_id}' not found in storage.",
                detail={"dataset_id": dataset_id},
            )

        record_count = len(df)
        column_count = len(df.columns)
        now = utcnow()

        # 1. Per-column statistics
        column_profiles: list[ColumnProfile] = []
        total_missing_values = 0
        findings: list[dict[str, Any]] = self.repo.get_findings(dataset_id) or []

        for col_name in df.columns:
            col_series = df[col_name]
            null_count = int(col_series.isna().sum())
            total_missing_values += null_count
            non_null_count = record_count - null_count
            null_pct = float(null_count / record_count * 100.0) if record_count > 0 else 0.0

            unique_count = int(col_series.nunique(dropna=True))

            # Non-null sample values safely converted to string
            non_null_samples = (
                col_series.dropna()
                .astype(str)
                .head(5)
                .tolist()
            )
            # Truncate each sample for clean profiling display
            samples = [truncate_evidence(str(s), max_len=40) for s in non_null_samples]

            col_profile = ColumnProfile(
                name=str(col_name),
                dtype=str(col_series.dtype),
                non_null_count=non_null_count,
                null_count=null_count,
                null_percentage=round(null_pct, 2),
                unique_count=unique_count,
                sample_values=samples,
            )
            column_profiles.append(col_profile)

            # High missing value finding (> 20% null)
            if null_pct > 20.0 and record_count > 0:
                severity = Severity.HIGH if null_pct > 50.0 else Severity.MEDIUM
                findings.append({
                    "finding_id": generate_storage_id(),
                    "category": FindingCategory.MISSING_VALUE.value,
                    "subcategory": "high_null_percentage",
                    "severity": severity.value,
                    "confidence": 1.0,
                    "dataset_id": dataset_id,
                    "record_index": None,
                    "field_name": str(col_name),
                    "location": f"column:{col_name}",
                    "evidence": f"Null rate: {null_pct:.1f}% ({null_count}/{record_count} records)",
                    "explanation": f"Column '{col_name}' has {null_pct:.1f}% missing values, which can degrade model quality.",
                    "detection_method": DetectionMethod.STATISTICAL.value,
                    "recommended_action": PolicyDecision.REVIEW.value,
                    "review_status": ReviewStatus.PENDING.value,
                    "remediation_status": RemediationStatus.PENDING.value,
                    "created_at": now.isoformat(),
                })

        # 2. Count duplicate records
        duplicate_count = 0
        if record_count > 1:
            duplicate_mask = df.duplicated()
            duplicate_count = int(duplicate_mask.sum())
            if duplicate_count > 0:
                dup_indices = df[duplicate_mask].index.tolist()[:10]
                findings.append({
                    "finding_id": generate_storage_id(),
                    "category": FindingCategory.DUPLICATE_RECORD.value,
                    "subcategory": "duplicate_rows",
                    "severity": Severity.MEDIUM.value if duplicate_count < 100 else Severity.HIGH.value,
                    "confidence": 1.0,
                    "dataset_id": dataset_id,
                    "record_index": dup_indices[0] if dup_indices else None,
                    "field_name": None,
                    "location": f"rows:{dup_indices}",
                    "evidence": f"{duplicate_count} duplicate rows found out of {record_count} total",
                    "explanation": f"Dataset contains {duplicate_count} exact duplicate rows, which may introduce training bias or leakage.",
                    "detection_method": DetectionMethod.STATISTICAL.value,
                    "recommended_action": PolicyDecision.REVIEW.value,
                    "review_status": ReviewStatus.PENDING.value,
                    "remediation_status": RemediationStatus.PENDING.value,
                    "created_at": now.isoformat(),
                })

        # 3. Empty records (entire row is null or empty strings)
        empty_record_count = 0
        if record_count > 0 and column_count > 0:
            is_empty_cell = df.isna() | (df.astype(str).apply(lambda col: col.str.strip().eq("")))
            empty_rows = is_empty_cell.all(axis=1)
            empty_record_count = int(empty_rows.sum())
            if empty_record_count > 0:
                empty_indices = df[empty_rows].index.tolist()[:10]
                findings.append({
                    "finding_id": generate_storage_id(),
                    "category": FindingCategory.EMPTY_RECORD.value,
                    "subcategory": "blank_rows",
                    "severity": Severity.LOW.value,
                    "confidence": 1.0,
                    "dataset_id": dataset_id,
                    "record_index": empty_indices[0] if empty_indices else None,
                    "field_name": None,
                    "location": f"rows:{empty_indices}",
                    "evidence": f"{empty_record_count} completely blank records found",
                    "explanation": f"Dataset contains {empty_record_count} empty records with no usable information.",
                    "detection_method": DetectionMethod.VALIDATION.value,
                    "recommended_action": PolicyDecision.REVIEW.value,
                    "review_status": ReviewStatus.PENDING.value,
                    "remediation_status": RemediationStatus.PENDING.value,
                    "created_at": now.isoformat(),
                })

        # 4. Schema consistency check: mixed data types in object columns
        schema_consistent = True
        for col_name in df.columns:
            if df[col_name].dtype == object:
                types = {type(val).__name__ for val in df[col_name].dropna()}
                if len(types) > 1:
                    schema_consistent = False
                    findings.append({
                        "finding_id": generate_storage_id(),
                        "category": FindingCategory.SCHEMA_INCONSISTENCY.value,
                        "subcategory": "mixed_types",
                        "severity": Severity.LOW.value,
                        "confidence": 0.95,
                        "dataset_id": dataset_id,
                        "record_index": None,
                        "field_name": str(col_name),
                        "location": f"column:{col_name}",
                        "evidence": f"Mixed types in column: {', '.join(sorted(types))}",
                        "explanation": f"Column '{col_name}' contains multiple incompatible Python types: {types}",
                        "detection_method": DetectionMethod.STRUCTURAL.value,
                        "recommended_action": PolicyDecision.REVIEW.value,
                        "review_status": ReviewStatus.PENDING.value,
                        "remediation_status": RemediationStatus.PENDING.value,
                        "created_at": now.isoformat(),
                    })

        # Save findings back to repo
        self.repo.save_findings(dataset_id, findings)

        # Build DatasetProfile schema object
        profile = DatasetProfile(
            dataset_id=dataset_id,
            record_count=record_count,
            column_count=column_count,
            columns=column_profiles,
            duplicate_count=duplicate_count,
            empty_record_count=empty_record_count,
            malformed_record_count=0,
            missing_value_total=total_missing_values,
            schema_consistent=schema_consistent,
            profiled_at=now,
        )

        # Save profile inside metadata
        meta["profile"] = profile.model_dump(mode="json")
        self.repo.save_metadata(meta)

        logger.info(
            "Profiled dataset %s: %d records, %d cols, %d missing values, %d duplicates",
            dataset_id,
            record_count,
            column_count,
            total_missing_values,
            duplicate_count,
        )

        return profile
