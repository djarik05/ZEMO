"""
Unit and integration tests for IngestionService.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    DatasetParsingError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from app.models.enums import DatasetFormat, DatasetStatus
from app.repositories.dataset_repo import DatasetRepository
from app.services.ingestion import IngestionService


def test_ingest_csv_success(sample_csv_bytes: bytes, dataset_repo: DatasetRepository) -> None:
    """Upload CSV, verify dataset_id, record_count, column list, status=parsed."""
    service = IngestionService(repo=dataset_repo)
    meta = service.ingest(sample_csv_bytes, "customers.csv")

    assert meta.dataset_id is not None
    assert len(meta.dataset_id) > 0
    assert meta.record_count == 20
    assert meta.column_count == 5
    assert set(meta.columns) == {"name", "email", "phone", "address", "notes"}
    assert meta.status == DatasetStatus.PARSED
    assert meta.format == DatasetFormat.CSV

    # Verify repo persistence
    saved_meta = dataset_repo.get_metadata(meta.dataset_id)
    assert saved_meta is not None
    assert saved_meta["record_count"] == 20

    df = dataset_repo.get_dataframe(meta.dataset_id)
    assert df is not None
    assert len(df) == 20


def test_ingest_json_success(sample_json_bytes: bytes, dataset_repo: DatasetRepository) -> None:
    """Upload JSON, verify dataset_id, record_count, column list, status=parsed."""
    service = IngestionService(repo=dataset_repo)
    meta = service.ingest(sample_json_bytes, "data.json")

    assert meta.dataset_id is not None
    assert meta.record_count == 20
    assert meta.column_count == 5
    assert set(meta.columns) == {"name", "email", "phone", "address", "notes"}
    assert meta.status == DatasetStatus.PARSED
    assert meta.format == DatasetFormat.JSON

    df = dataset_repo.get_dataframe(meta.dataset_id)
    assert df is not None
    assert len(df) == 20


def test_ingest_jsonl_success(sample_jsonl_bytes: bytes, dataset_repo: DatasetRepository) -> None:
    """Upload JSONL, verify dataset_id, record_count, column list, status=parsed."""
    service = IngestionService(repo=dataset_repo)
    meta = service.ingest(sample_jsonl_bytes, "stream.jsonl")

    assert meta.dataset_id is not None
    assert meta.record_count == 20
    assert meta.column_count == 5
    assert set(meta.columns) == {"name", "email", "phone", "address", "notes"}
    assert meta.status == DatasetStatus.PARSED
    assert meta.format == DatasetFormat.JSONL

    df = dataset_repo.get_dataframe(meta.dataset_id)
    assert df is not None
    assert len(df) == 20


def test_ingest_unsupported_format(dataset_repo: DatasetRepository) -> None:
    """Uploading .xlsx should raise UnsupportedFormatError."""
    service = IngestionService(repo=dataset_repo)
    dummy_xlsx = b"PK\x03\x04fake content"

    with pytest.raises(UnsupportedFormatError) as exc_info:
        service.ingest(dummy_xlsx, "spreadsheet.xlsx")

    assert "unsupported" in exc_info.value.message.lower() or "xlsx" in str(exc_info.value.detail)


def test_ingest_file_too_large(dataset_repo: DatasetRepository, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify FileTooLargeError for oversized content."""
    service = IngestionService(repo=dataset_repo)
    # Simulate a small limit of 50 bytes via settings property
    monkeypatch.setattr("app.core.config.Settings.max_upload_bytes", property(lambda self: 50))

    oversized_data = b"name,email\n" + b"Alice,alice@example.com\n" * 5

    with pytest.raises(FileTooLargeError):
        service.ingest(oversized_data, "large.csv")


def test_ingest_empty_csv(dataset_repo: DatasetRepository) -> None:
    """Verify appropriate error for empty CSV file."""
    service = IngestionService(repo=dataset_repo)
    empty_content = b"   \n\n  "

    with pytest.raises(DatasetParsingError) as exc_info:
        service.ingest(empty_content, "empty.csv")

    assert "empty" in exc_info.value.message.lower()


def test_ingest_malformed_json(dataset_repo: DatasetRepository) -> None:
    """Verify DatasetParsingError for invalid JSON syntax."""
    service = IngestionService(repo=dataset_repo)
    malformed_json = b"{'invalid': json syntax, missing quotes}"

    with pytest.raises(DatasetParsingError):
        service.ingest(malformed_json, "bad.json")
