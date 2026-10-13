"""Pydantic schemas for dataset ingestion, metadata, and profiling."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import DatasetFormat, DatasetStatus


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: list[str] = Field(default_factory=list, max_length=5)


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    dataset_id: str
    original_filename: str
    sanitized_filename: str
    format: DatasetFormat
    file_size_bytes: int
    record_count: int
    column_count: int
    columns: list[str] = Field(default_factory=list)
    status: DatasetStatus
    created_at: datetime
    updated_at: datetime
    version: int = 1
    parent_dataset_id: str | None = None
    storage_key: str | None = None


class DatasetProfile(BaseModel):
    dataset_id: str
    record_count: int
    column_count: int
    columns: list[ColumnProfile] = Field(default_factory=list)
    duplicate_count: int = 0
    empty_record_count: int = 0
    malformed_record_count: int = 0
    missing_value_total: int = 0
    schema_consistent: bool = True
    profiled_at: datetime | None = None


class DatasetListItem(BaseModel):
    dataset_id: str
    original_filename: str
    format: DatasetFormat
    record_count: int
    status: DatasetStatus
    created_at: datetime


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    format: DatasetFormat
    record_count: int
    column_count: int
    file_size_bytes: int
    status: DatasetStatus
    message: str
