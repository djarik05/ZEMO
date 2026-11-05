"""
Unit and integration tests for SanitizationService.
"""

from __future__ import annotations

import pytest

from app.repositories.dataset_repo import DatasetRepository
from app.schemas.scan import ScanRequest
from app.services.ingestion import IngestionService
from app.services.sanitization import SanitizationService
from app.services.scan_service import ScanService


@pytest.fixture
def scanned_dataset(sample_csv_bytes: bytes, dataset_repo: DatasetRepository) -> str:
    """Helper fixture that ingests and scans the sample CSV with PII and prompt injection."""
    ingestion = IngestionService(repo=dataset_repo)
    meta = ingestion.ingest(sample_csv_bytes, "users_to_sanitize.csv")
    scanner = ScanService(repo=dataset_repo)
    scanner.run_scan(ScanRequest(dataset_id=meta.dataset_id))
    return meta.dataset_id


def test_sanitize_creates_new_version(scanned_dataset: str, dataset_repo: DatasetRepository) -> None:
    """Verify sanitized_dataset_id != original dataset_id and is persisted."""
    service = SanitizationService(repo=dataset_repo)
    result = service.sanitize(scanned_dataset)

    assert result.sanitized_dataset_id is not None
    assert result.sanitized_dataset_id != scanned_dataset
    assert dataset_repo.exists(result.sanitized_dataset_id)

    new_meta = dataset_repo.get_metadata(result.sanitized_dataset_id)
    assert new_meta is not None
    assert new_meta["parent_dataset_id"] == scanned_dataset
    assert new_meta["version"] == 2


def test_original_unchanged(scanned_dataset: str, dataset_repo: DatasetRepository) -> None:
    """Original dataset data must not be modified after sanitization."""
    orig_df_before = dataset_repo.get_dataframe(scanned_dataset)
    assert orig_df_before is not None
    assert len(orig_df_before) == 20

    service = SanitizationService(repo=dataset_repo)
    service.sanitize(scanned_dataset)

    orig_df_after = dataset_repo.get_dataframe(scanned_dataset)
    assert orig_df_after is not None
    assert len(orig_df_after) == 20

    # Ensure unmasked email is still present in the original dataset
    assert "alice.smith@example.com" in orig_df_after["email"].values


def test_sanitize_redacts_email(scanned_dataset: str, dataset_repo: DatasetRepository) -> None:
    """Verify email values are masked in sanitized output."""
    service = SanitizationService(repo=dataset_repo)
    result = service.sanitize(scanned_dataset)

    clean_df = dataset_repo.get_dataframe(result.sanitized_dataset_id)
    assert clean_df is not None

    for email_val in clean_df["email"]:
        assert "alice.smith@example.com" not in str(email_val)
        assert "***" in str(email_val)


def test_sanitize_removes_prompt_injection_records(scanned_dataset: str, dataset_repo: DatasetRepository) -> None:
    """Records containing prompt injection triggers must be removed."""
    service = SanitizationService(repo=dataset_repo)
    result = service.sanitize(scanned_dataset)

    clean_df = dataset_repo.get_dataframe(result.sanitized_dataset_id)
    assert clean_df is not None

    for note in clean_df["notes"]:
        assert "ignore previous instructions" not in str(note).lower()

    assert result.records_removed >= 1


def test_sanitize_counts(scanned_dataset: str, dataset_repo: DatasetRepository) -> None:
    """Verify records_removed, fields_redacted, fields_masked counts."""
    service = SanitizationService(repo=dataset_repo)
    result = service.sanitize(scanned_dataset)

    assert result.original_record_count == 20
    assert result.records_removed >= 1
    assert result.sanitized_record_count == 20 - result.records_removed
    assert result.fields_masked > 0
    assert len(result.actions_applied) > 0
