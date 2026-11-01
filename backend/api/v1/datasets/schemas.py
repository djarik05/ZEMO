from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, ConfigDict, Field


class DatasetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    original_filename: str
    format: str
    file_size: int
    status: str
    validation_status: str
    profiling_status: str
    record_count: Optional[int] = None
    column_count: Optional[int] = None
    current_version: int = 1


class DatasetCreateResponse(DatasetBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    content_hash: str
    created_at: datetime
    updated_at: datetime


class DatasetListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: List[DatasetCreateResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DatasetVersionBase(BaseModel):
    version_number: int
    original_filename: str
    file_size: int
    format: str
    status: str
    validation_status: str
    profiling_status: str
    record_count: Optional[int] = None
    column_count: Optional[int] = None


class DatasetVersionResponse(DatasetVersionBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    content_hash: str
    storage_path: str
    created_at: datetime


class PreviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    records: List[dict]
    columns: List[str]
    total_rows: int
    total_columns: int
    preview_rows: int
    preview_columns: int
    truncated: bool


class ProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    record_count: int
    column_count: int
    columns: List[str]
    schema: dict
    statistics: dict
    duplicate_records: int
    empty_columns: List[str]


class ValidationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: str
    errors: List[dict] = []
    warnings: List[str] = []
    valid_records: int = 0
    invalid_records: int = 0
    total_records: int = 0
    partial_ingestion: bool = False
    error_summary: Optional[str] = None


class ProcessDatasetRequest(BaseModel):
    pass


class ValidateDatasetRequest(BaseModel):
    pass


class ProfileDatasetRequest(BaseModel):
    pass


class AddVersionRequest(BaseModel):
    format: str = Field(..., pattern="^(csv|json|jsonl)$")


class DeleteDatasetResponse(BaseModel):
    success: bool
    message: str