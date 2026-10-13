"""
Dataset ingestion service.

Handles parsing of CSV, JSON, and JSONL formats with encoding validation,
delimiter detection, record limits, security checks, and persistent storage
via DatasetRepository.
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import (
    DatasetParsingError,
    FileTooLargeError,
    RecordLimitExceededError,
    UnsupportedFormatError,
)
from app.core.logging import get_logger
from app.core.security import generate_storage_id, sanitize_filename, validate_extension
from app.models.enums import DatasetFormat, DatasetStatus, utcnow
from app.repositories.dataset_repo import DatasetRepository
from app.schemas.dataset import DatasetMetadata

logger = get_logger("ingestion")


class IngestionService:
    """Service responsible for uploading, validating, parsing, and storing datasets."""

    def __init__(self, repo: DatasetRepository | None = None) -> None:
        self.repo = repo or DatasetRepository()
        self.settings = get_settings()

    def parse_csv(self, content: bytes, filename: str) -> pd.DataFrame:
        """
        Parse CSV bytes, validate encoding, detect delimiter, and enforce record limit.
        """
        # Try UTF-8 first, fallback to latin-1 / cp1252
        text: str
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                text = content.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise DatasetParsingError(
                f"Failed to decode CSV '{filename}' with supported encodings (UTF-8, Latin-1, CP1252).",
                detail={"filename": filename},
            )

        if not text.strip():
            raise DatasetParsingError(
                f"CSV file '{filename}' is empty.",
                detail={"filename": filename},
            )

        # Detect delimiter using csv.Sniffer
        sample_chars = text[:10000]
        delimiter = ","
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_chars, delimiters=[",", ";", "\t", "|"])
            delimiter = dialect.delimiter
        except Exception:
            # Fallback heuristic
            for cand in [",", "\t", ";", "|"]:
                if cand in sample_chars:
                    delimiter = cand
                    break

        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=delimiter,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as exc:
            logger.error("Failed to parse CSV %s: %s", filename, exc)
            raise DatasetParsingError(
                f"Failed to parse CSV content for '{filename}': {exc}",
                detail={"filename": filename, "error": str(exc)},
            ) from exc

        if len(df) > self.settings.max_record_count:
            raise RecordLimitExceededError(
                f"Dataset contains {len(df)} records, which exceeds the limit of {self.settings.max_record_count}.",
                detail={"record_count": len(df), "max_records": self.settings.max_record_count},
            )

        return df

    def parse_json(self, content: bytes, filename: str) -> pd.DataFrame:
        """
        Parse JSON array or object, normalizing nested structures to a flat DataFrame.
        """
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception as exc:
                raise DatasetParsingError(
                    f"Failed to decode JSON '{filename}': {exc}",
                    detail={"filename": filename},
                ) from exc

        if not text.strip():
            raise DatasetParsingError(f"JSON file '{filename}' is empty.", detail={"filename": filename})

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetParsingError(
                f"Malformed JSON in '{filename}': {exc}",
                detail={"filename": filename, "error": str(exc)},
            ) from exc

        if isinstance(data, list):
            if not data:
                return pd.DataFrame()
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            # Check if any top-level key contains a list of records
            found_records = False
            for key, val in data.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    df = pd.json_normalize(val)
                    found_records = True
                    break
            if not found_records:
                # Single record object or dictionary of simple key-values
                df = pd.json_normalize([data])
        else:
            raise DatasetParsingError(
                f"JSON in '{filename}' must be an array of objects or an object.",
                detail={"filename": filename},
            )

        if len(df) > self.settings.max_record_count:
            raise RecordLimitExceededError(
                f"Dataset contains {len(df)} records, exceeding limit of {self.settings.max_record_count}.",
                detail={"record_count": len(df), "max_records": self.settings.max_record_count},
            )

        return df

    def parse_jsonl(self, content: bytes, filename: str) -> pd.DataFrame:
        """
        Parse line-by-line JSON (JSON Lines), gracefully handling malformed lines.
        """
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception as exc:
                raise DatasetParsingError(
                    f"Failed to decode JSONL '{filename}': {exc}",
                    detail={"filename": filename},
                ) from exc

        records: list[dict[str, Any]] = []
        malformed_count = 0

        for line_num, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
                else:
                    records.append({"value": item})
            except json.JSONDecodeError:
                malformed_count += 1
                logger.warning("Skipping malformed line %d in %s", line_num + 1, filename)

        if not records and malformed_count > 0:
            raise DatasetParsingError(
                f"All lines in JSONL '{filename}' were malformed.",
                detail={"filename": filename, "malformed_lines": malformed_count},
            )

        df = pd.json_normalize(records) if records else pd.DataFrame()

        if len(df) > self.settings.max_record_count:
            raise RecordLimitExceededError(
                f"Dataset contains {len(df)} records, exceeding limit of {self.settings.max_record_count}.",
                detail={"record_count": len(df), "max_records": self.settings.max_record_count},
            )

        return df

    def ingest(self, file_content: bytes, filename: str) -> DatasetMetadata:
        """
        Full ingestion pipeline:
        1. Validate file size and extension.
        2. Parse file into pandas DataFrame.
        3. Save raw file, parquet representation, and metadata.
        4. Return DatasetMetadata.
        """
        if len(file_content) > self.settings.max_upload_bytes:
            raise FileTooLargeError(
                f"File size {len(file_content)} bytes exceeds maximum {self.settings.max_upload_bytes} bytes.",
                detail={"size": len(file_content), "max_size": self.settings.max_upload_bytes},
            )

        ext = validate_extension(filename)
        safe_name = sanitize_filename(filename)
        dataset_id = generate_storage_id()
        now = utcnow()

        fmt: DatasetFormat
        if ext == ".csv":
            fmt = DatasetFormat.CSV
            df = self.parse_csv(file_content, safe_name)
        elif ext == ".json":
            fmt = DatasetFormat.JSON
            df = self.parse_json(file_content, safe_name)
        elif ext == ".jsonl":
            fmt = DatasetFormat.JSONL
            df = self.parse_jsonl(file_content, safe_name)
        else:
            raise UnsupportedFormatError(f"Unsupported format: {ext}", detail={"extension": ext})

        # Convert all column names to string
        df.columns = [str(c) for c in df.columns]

        # Persist raw file and parquet
        self.repo.save_raw_file(dataset_id, safe_name, file_content)
        self.repo.save_dataframe(dataset_id, df)

        metadata_dict = {
            "dataset_id": dataset_id,
            "original_filename": filename,
            "sanitized_filename": safe_name,
            "format": fmt.value,
            "file_size_bytes": len(file_content),
            "record_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "status": DatasetStatus.PARSED.value,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "version": 1,
            "parent_dataset_id": None,
            "storage_key": f"datasets/{dataset_id}/data.parquet",
        }

        self.repo.save_metadata(metadata_dict)

        logger.info(
            "Ingested dataset %s: %s (%d records, %d cols)",
            dataset_id,
            safe_name,
            len(df),
            len(df.columns),
        )

        return DatasetMetadata(
            dataset_id=dataset_id,
            original_filename=filename,
            sanitized_filename=safe_name,
            format=fmt,
            file_size_bytes=len(file_content),
            record_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            status=DatasetStatus.PARSED,
            created_at=now,
            updated_at=now,
            version=1,
            parent_dataset_id=None,
            storage_key=metadata_dict["storage_key"],
        )
