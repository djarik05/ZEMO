from datetime import datetime
from typing import Optional, List, Any

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
    format: str = Field(..., pattern="^(csv|json|jsonl|xlsx|parquet)$")


class DeleteDatasetResponse(BaseModel):
    success: bool
    message: str


class ColumnProfileResponse(BaseModel):
    name: str
    inferred_type: str
    dtype: str
    is_nullable: bool
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any]
    stats: dict = Field(default_factory=dict)


class DatasetAnalysisResponse(BaseModel):
    dataset_id: str
    version_id: str
    total_rows: int
    total_columns: int
    missing_cells_total: int
    missing_cells_percentage: float
    duplicate_rows: int
    duplicate_rows_percentage: float
    memory_usage_bytes: int
    memory_usage_formatted: str
    column_types_summary: dict
    columns: List[ColumnProfileResponse]
    target_candidates: List[str]


class VisualizationResponse(BaseModel):
    dataset_id: str
    version_id: str
    primary_numeric: Optional[str] = None
    primary_categorical: Optional[str] = None
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    histograms: dict = Field(default_factory=dict)
    box_plots: dict = Field(default_factory=dict)
    categories: dict = Field(default_factory=dict)
    missing_overview: List[dict] = Field(default_factory=list)
    correlation_matrix: dict = Field(default_factory=dict)
    scatter_plot: dict = Field(default_factory=dict)


class ColumnOverrideRequest(BaseModel):
    columns: dict[str, str]