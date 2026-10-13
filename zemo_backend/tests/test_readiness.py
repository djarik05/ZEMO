"""
Unit and integration tests for ReadinessService.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.models.enums import FindingCategory, ReadinessStatus, Severity
from app.repositories.dataset_repo import DatasetRepository
from app.schemas.scan import ScanRequest
from app.services.ingestion import IngestionService
from app.services.readiness import ReadinessService
from app.services.scan_service import ScanService


def test_ready_clean_dataset(sample_clean_csv_bytes: bytes, dataset_repo: DatasetRepository) -> None:
    """Clean dataset after scan should assess as READY with 100.0 score."""
    ingestion = IngestionService(repo=dataset_repo)
    meta = ingestion.ingest(sample_clean_csv_bytes, "clean_products.csv")

    scanner = ScanService(repo=dataset_repo)
    scanner.run_scan(ScanRequest(dataset_id=meta.dataset_id))

    readiness = ReadinessService(repo=dataset_repo)
    report = readiness.assess(meta.dataset_id)

    assert report.overall_status == ReadinessStatus.READY
    assert report.overall_score == 100.0
    assert len(report.blocking_issues) == 0


def test_not_ready_with_findings(dataset_repo: DatasetRepository) -> None:
    """Dataset with unresolved non-critical findings should assess as NOT_READY."""
    df = pd.DataFrame({
        "username": ["john_doe", "jane_doe"],
        "email": ["john@example.com", "jane@example.com"],
    })
    dataset_id = "ds-not-ready"
    dataset_repo.save_dataframe(dataset_id, df)
    dataset_repo.save_metadata({
        "dataset_id": dataset_id,
        "original_filename": "users.csv",
        "sanitized_filename": "users.csv",
        "format": "csv",
        "record_count": 2,
        "column_count": 2,
        "columns": ["username", "email"],
        "file_size_bytes": 100,
        "status": "scanned",
        "version": 1,
    })

    # Save findings with only general PII (Email: HIGH severity)
    dataset_repo.save_findings(dataset_id, [
        {
            "finding_id": "f1",
            "category": FindingCategory.EMAIL.value,
            "severity": Severity.HIGH.value,
            "confidence": 0.95,
            "dataset_id": dataset_id,
            "field_name": "email",
        }
    ])

    readiness = ReadinessService(repo=dataset_repo)
    report = readiness.assess(dataset_id)

    assert report.overall_status == ReadinessStatus.NOT_READY
    assert report.overall_score < 100.0
    assert len(report.warnings) > 0


def test_blocked_with_critical(dataset_repo: DatasetRepository) -> None:
    """Dataset with critical secrets should evaluate as BLOCKED."""
    df = pd.DataFrame({
        "key": ["sk-proj-1234567890abcdef1234"]
    })
    dataset_id = "ds-blocked"
    dataset_repo.save_dataframe(dataset_id, df)
    dataset_repo.save_metadata({
        "dataset_id": dataset_id,
        "original_filename": "secrets.csv",
        "sanitized_filename": "secrets.csv",
        "format": "csv",
        "record_count": 1,
        "column_count": 1,
        "columns": ["key"],
        "file_size_bytes": 50,
        "status": "scanned",
        "version": 1,
    })

    dataset_repo.save_findings(dataset_id, [
        {
            "finding_id": "f-crit",
            "category": FindingCategory.API_KEY.value,
            "severity": Severity.CRITICAL.value,
            "confidence": 0.99,
            "dataset_id": dataset_id,
            "field_name": "key",
        }
    ])

    readiness = ReadinessService(repo=dataset_repo)
    report = readiness.assess(dataset_id)

    assert report.overall_status == ReadinessStatus.BLOCKED
    assert len(report.blocking_issues) >= 1


def test_score_calculation(dataset_repo: DatasetRepository) -> None:
    """Verify weighted readiness score is computed correctly based on check weights."""
    df = pd.DataFrame({"col": ["val"]})
    dataset_id = "ds-score-calc"
    dataset_repo.save_dataframe(dataset_id, df)
    dataset_repo.save_metadata({
        "dataset_id": dataset_id,
        "original_filename": "test.csv",
        "sanitized_filename": "test.csv",
        "format": "csv",
        "record_count": 1,
        "column_count": 1,
        "columns": ["col"],
        "file_size_bytes": 10,
        "status": "scanned",
        "version": 1,
    })

    # No findings -> all checks pass -> 100.0
    readiness = ReadinessService(repo=dataset_repo)
    report = readiness.assess(dataset_id)
    assert report.overall_score == 100.0

    total_weight = sum(c.weight for c in report.checks)
    passed_weight = sum(c.weight for c in report.checks if c.passed)
    expected_score = round((passed_weight / total_weight) * 100.0, 2)
    assert report.overall_score == expected_score
