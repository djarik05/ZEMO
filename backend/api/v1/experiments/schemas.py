from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class TrainExperimentRequest(BaseModel):
    dataset_id: str
    name: Optional[str] = None
    task: Literal["classification", "regression", "clustering"]
    target_column: Optional[str] = None
    selected_features: Optional[List[str]] = None
    excluded_columns: List[str] = Field(default_factory=list)
    type_overrides: Dict[str, str] = Field(default_factory=dict)

    # Preprocessing
    missing_numeric_strategy: Literal["mean", "median", "constant", "drop"] = "mean"
    missing_numeric_fill_value: Optional[float] = 0.0
    missing_categorical_strategy: Literal["most_frequent", "constant", "drop"] = "most_frequent"
    missing_categorical_fill_value: str = "missing"
    encoding_strategy: Literal["auto", "onehot", "ordinal", "none"] = "auto"
    scaling_strategy: Literal["standard", "minmax", "robust", "none"] = "standard"

    # Outliers
    outlier_method: Literal["none", "iqr_clip", "zscore_clip"] = "none"
    outlier_factor: float = 1.5
    remove_constant_columns: bool = True

    # Estimator
    algorithm_id: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

    # Train / test split & Cross-validation
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = 42
    cv_folds: Optional[int] = Field(default=None, ge=2, le=10)
    stratify: bool = True


class ExperimentResponse(BaseModel):
    id: str
    name: str
    dataset_id: str
    dataset_version_id: Optional[str] = None
    task: str
    algorithm_id: str
    status: str
    config: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifact_path: Optional[str] = None
    model_size_bytes: Optional[int] = None
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class ExperimentListResponse(BaseModel):
    total: int
    experiments: List[ExperimentResponse]


class CancelExperimentResponse(BaseModel):
    experiment_id: str
    status: str
    message: str


class CompareExperimentsRequest(BaseModel):
    experiment_ids: List[str] = Field(..., min_length=2, description="List of at least two experiment IDs to compare.")


class CompareExperimentsResponse(BaseModel):
    dataset_id: str
    task: str
    compatible: bool
    primary_metric: str
    experiments_count: int
    experiments: List[Dict[str, Any]]
    best_model: Optional[Dict[str, Any]] = None
    metrics_comparison: Dict[str, Dict[str, Any]]
    recommendation: str


class PredictionRequest(BaseModel):
    data: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(..., description="Single row object or list of row objects containing feature values.")
    include_probabilities: bool = Field(default=True, description="Whether to include probability distribution for classification.")


class PredictionResponse(BaseModel):
    experiment_id: str
    task: str
    algorithm_id: str
    predictions: List[Any]
    probabilities: Optional[List[Dict[str, float]]] = None
    sample_count: int
    features_used: List[str]
    latency_ms: float
    timestamp: str


class ModelMetadataResponse(BaseModel):
    model_id: str
    name: str
    task: str
    algorithm_id: str
    dataset_id: str
    dataset_version_id: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    feature_schema: Dict[str, Any]
    preprocessing_strategy: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    framework: str = "Scikit-learn"
    serialization_format: str = "Joblib"
    sample_prediction_code: str


