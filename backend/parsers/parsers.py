import abc
from dataclasses import dataclass
from typing import Any, BinaryIO, Generator

import pandas as pd

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParseResult:
    records: list[dict[str, Any]]
    columns: list[str]
    record_count: int
    column_count: int
    errors: list[dict[str, Any]]
    warnings: list[str]


class ParseError(Exception):
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class BaseParser(abc.ABC):
    def __init__(self):
        self.settings = get_settings()

    @abc.abstractmethod
    def parse(self, file: BinaryIO) -> ParseResult:
        pass

    def _validate_limits(self, record_count: int, column_count: int) -> list[str]:
        warnings = []
        if record_count > self.settings.max_records:
            warnings.append(f"Record count {record_count} exceeds maximum {self.settings.max_records}. Processing truncated.")
        if column_count > self.settings.max_columns:
            warnings.append(f"Column count {column_count} exceeds maximum {self.settings.max_columns}.")
        return warnings

    def _truncate_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(records) > self.settings.max_records:
            return records[: self.settings.max_records]
        return records

    def _truncate_columns(self, records: list[dict[str, Any]], columns: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
        if len(columns) > self.settings.max_columns:
            truncated_columns = columns[: self.settings.max_columns]
            truncated_records = [{k: v for k, v in r.items() if k in truncated_columns} for r in records]
            return truncated_records, truncated_columns
        return records, columns


class CSVParser(BaseParser):
    def parse(self, file: BinaryIO) -> ParseResult:
        errors = []
        warnings = []
        records = []
        columns = []

        try:
            file.seek(0)
            df = pd.read_csv(
                file,
                encoding="utf-8",
                on_bad_lines="warn",
                engine="python",
                dtype=str,
                na_filter=False,
            )
        except UnicodeDecodeError:
            try:
                file.seek(0)
                df = pd.read_csv(
                    file,
                    encoding="latin-1",
                    on_bad_lines="warn",
                    engine="python",
                    dtype=str,
                    na_filter=False,
                )
                warnings.append("File encoding detected as latin-1 (not UTF-8)")
            except Exception as e:
                raise ParseError(f"Failed to parse CSV: {e}") from e
        except pd.errors.EmptyDataError:
            return ParseResult(
                records=[],
                columns=[],
                record_count=0,
                column_count=0,
                errors=[{"type": "empty_file", "message": "CSV file is empty"}],
                warnings=[],
            )
        except Exception as e:
            raise ParseError(f"Failed to parse CSV: {e}") from e

        if df.empty:
            return ParseResult(
                records=[],
                columns=list(df.columns),
                record_count=0,
                column_count=len(df.columns),
                errors=[],
                warnings=["CSV file has no data rows"],
            )

        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            errors.append({
                "type": "duplicate_columns",
                "message": f"Duplicate column names found: {duplicate_columns}",
                "columns": duplicate_columns,
            })

        columns = list(df.columns)
        records = df.to_dict(orient="records")

        record_count = len(records)
        column_count = len(columns)

        warnings.extend(self._validate_limits(record_count, column_count))
        records = self._truncate_records(records)
        records, columns = self._truncate_columns(records, columns)

        return ParseResult(
            records=records,
            columns=columns,
            record_count=record_count,
            column_count=column_count,
            errors=errors,
            warnings=warnings,
        )


class JSONParser(BaseParser):
    def parse(self, file: BinaryIO) -> ParseResult:
        errors = []
        warnings = []
        records = []
        columns = []

        try:
            file.seek(0)
            import json
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e}") from e
        except Exception as e:
            raise ParseError(f"Failed to parse JSON: {e}") from e

        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            if not data:
                return ParseResult(
                    records=[],
                    columns=[],
                    record_count=0,
                    column_count=0,
                    errors=[{"type": "empty_array", "message": "JSON array is empty"}],
                    warnings=[],
                )
            records = data
        else:
            raise ParseError("JSON must be an object or array of objects")

        if not all(isinstance(r, dict) for r in records):
            raise ParseError("All JSON records must be objects")

        all_keys = set()
        for r in records:
            all_keys.update(r.keys())
        columns = sorted(list(all_keys))

        for i, r in enumerate(records):
            for col in columns:
                if col not in r:
                    r[col] = None

        record_count = len(records)
        column_count = len(columns)

        warnings.extend(self._validate_limits(record_count, column_count))
        records = self._truncate_records(records)
        records, columns = self._truncate_columns(records, columns)

        return ParseResult(
            records=records,
            columns=columns,
            record_count=record_count,
            column_count=column_count,
            errors=errors,
            warnings=warnings,
        )


class JSONLParser(BaseParser):
    def parse(self, file: BinaryIO) -> ParseResult:
        errors = []
        warnings = []
        records = []
        columns = set()

        try:
            file.seek(0)
            import json
            line_number = 0
            for line in file:
                line_number += 1
                line = line.strip()
                if not line:
                    warnings.append(f"Empty line at line {line_number}, skipping")
                    continue

                if len(line) > self.settings.max_field_size_bytes:
                    errors.append({
                        "type": "line_too_long",
                        "message": f"Line {line_number} exceeds maximum line size",
                        "line": line_number,
                    })
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append({
                        "type": "json_decode_error",
                        "message": f"Invalid JSON at line {line_number}: {e}",
                        "line": line_number,
                    })
                    continue

                if not isinstance(record, dict):
                    errors.append({
                        "type": "invalid_record",
                        "message": f"Line {line_number} is not a JSON object",
                        "line": line_number,
                    })
                    continue

                records.append(record)
                columns.update(record.keys())

                if len(records) >= self.settings.max_records:
                    warnings.append(f"Reached maximum record limit of {self.settings.max_records}, stopping")
                    break

        except Exception as e:
            raise ParseError(f"Failed to parse JSONL: {e}") from e

        if not records:
            return ParseResult(
                records=[],
                columns=[],
                record_count=0,
                column_count=0,
                errors=[{"type": "no_valid_records", "message": "No valid records found in JSONL file"}],
                warnings=warnings,
            )

        columns_list = sorted(list(columns))

        for r in records:
            for col in columns_list:
                if col not in r:
                    r[col] = None

        record_count = len(records)
        column_count = len(columns_list)

        warnings.extend(self._validate_limits(record_count, column_count))
        records = self._truncate_records(records)
        records, columns_list = self._truncate_columns(records, columns_list)

        return ParseResult(
            records=records,
            columns=columns_list,
            record_count=record_count,
            column_count=column_count,
            errors=errors,
            warnings=warnings,
        )


def get_parser(format: str) -> BaseParser:
    format = format.lower()
    if format == "csv":
        return CSVParser()
    elif format == "json":
        return JSONParser()
    elif format == "jsonl":
        return JSONLParser()
    else:
        raise ParseError(f"Unsupported format: {format}")