"""
Dynamic ML Pipeline Builder for Zemo.

Constructs reusable Scikit-learn Pipeline objects combining:
- Feature extraction & column management
- Missing-value imputation
- Categorical encoding
- Numeric scaling and transformations
- Estimator instantiation from AlgorithmRegistry
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

try:
    from core.exceptions import AppValidationError as ValidationError
    from services.ml.analyzer import DatasetAnalyzer
    from services.ml.preprocessor import OutlierClipper, SafeLogTransformer
    from services.ml.registry import get_algorithm_registry
except ImportError:
    from backend.core.exceptions import AppValidationError as ValidationError
    from backend.services.ml.analyzer import DatasetAnalyzer
    from backend.services.ml.preprocessor import OutlierClipper, SafeLogTransformer
    from backend.services.ml.registry import get_algorithm_registry

logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):
    """User-provided training configuration specification."""
    dataset_id: str
    dataset_version_id: Optional[str] = None
    experiment_name: Optional[str] = None
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

    # Outliers & Feature Engineering
    outlier_method: Literal["none", "iqr_clip", "zscore_clip"] = "none"
    outlier_factor: float = 1.5
    log_transform_columns: List[str] = Field(default_factory=list)
    remove_constant_columns: bool = True

    # Estimator & Hyperparameters
    algorithm_id: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

    # Splitting & Cross Validation
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = 42
    cv_folds: Optional[int] = Field(default=None, ge=2, le=10)
    stratify: bool = True


class PipelineBuildResult:
    def __init__(
        self,
        pipeline: Pipeline,
        feature_columns: List[str],
        numeric_columns: List[str],
        categorical_columns: List[str],
        target_column: Optional[str],
        algorithm_info: Any,
        sanitized_hyperparameters: Dict[str, Any],
    ):
        self.pipeline = pipeline
        self.feature_columns = feature_columns
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.algorithm_info = algorithm_info
        self.sanitized_hyperparameters = sanitized_hyperparameters


class DynamicPipelineBuilder:
    """
    Programmatically constructs an end-to-end Scikit-learn Pipeline
    from data metadata and user configuration.
    """

    def __init__(self, analyzer: Optional[DatasetAnalyzer] = None):
        self.analyzer = analyzer or DatasetAnalyzer()
        self.registry = get_algorithm_registry()

    def build_pipeline(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
    ) -> PipelineBuildResult:
        """
        Validates configuration and creates an unfitted Pipeline ready for .fit().
        """
        # 1. Validate Algorithm and Task
        algo_info = self.registry.get(config.algorithm_id, task=config.task)
        if algo_info.task != config.task:
            raise ValidationError(
                f"Algorithm '{algo_info.display_name}' ({algo_info.task}) is not compatible with requested task '{config.task}'."
            )

        # 2. Validate and sanitize hyperparameters
        is_valid, sanitized_params, errors = self.registry.validate_hyperparameters(
            config.algorithm_id, config.hyperparameters, task=config.task
        )
        if not is_valid:
            raise ValidationError(
                f"Invalid hyperparameters for {algo_info.display_name}: {'; '.join(errors)}"
            )

        # 3. Determine feature columns
        all_cols = list(df.columns)
        target_col = config.target_column if config.task != "clustering" else None

        if config.task != "clustering":
            if not target_col:
                raise ValidationError("A target column is required for supervised learning tasks.")
            if target_col not in all_cols:
                raise ValidationError(f"Target column '{target_col}' not found in dataset columns: {all_cols}")

        excluded = set(config.excluded_columns)
        if target_col:
            excluded.add(target_col)

        if config.selected_features:
            feature_cols = [c for c in config.selected_features if c in all_cols and c not in excluded]
        else:
            feature_cols = [c for c in all_cols if c not in excluded]

        if not feature_cols:
            raise ValidationError(
                "No valid feature columns remaining after applying selections and exclusions."
            )

        # 4. Partition features into numeric and categorical
        analysis = self.analyzer.analyze_dataset(df[feature_cols], type_overrides=config.type_overrides)
        col_type_map = {c["name"]: c["inferred_type"] for c in analysis["columns"]}

        numeric_cols = [c for c in feature_cols if col_type_map.get(c) == "numeric"]
        categorical_cols = [c for c in feature_cols if col_type_map.get(c) in ("categorical", "boolean", "text")]

        # 5. Build ColumnTransformer
        transformers: List[Tuple[str, Any, List[str]]] = []

        # Numeric sub-pipeline
        if numeric_cols:
            num_steps: List[Tuple[str, BaseEstimator]] = []

            # Missing value imputation
            num_impute_strategy = config.missing_numeric_strategy
            if num_impute_strategy == "drop":
                # Fallback to median in pipeline transform to prevent NaN crashes
                num_impute_strategy = "median"
            
            if num_impute_strategy == "constant":
                num_steps.append((
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value=config.missing_numeric_fill_value or 0.0),
                ))
            else:
                num_steps.append((
                    "imputer",
                    SimpleImputer(strategy=num_impute_strategy),
                ))

            # Outlier clipping
            if config.outlier_method in ("iqr_clip", "zscore_clip"):
                outlier_m = "zscore" if config.outlier_method == "zscore_clip" else "iqr"
                num_steps.append(("outliers", OutlierClipper(method=outlier_m, factor=config.outlier_factor)))

            # Feature scaling
            if config.scaling_strategy == "standard":
                num_steps.append(("scaler", StandardScaler()))
            elif config.scaling_strategy == "minmax":
                num_steps.append(("scaler", MinMaxScaler()))
            elif config.scaling_strategy == "robust":
                num_steps.append(("scaler", RobustScaler()))

            transformers.append(("numeric", Pipeline(num_steps), numeric_cols))

        # Categorical sub-pipeline
        if categorical_cols:
            cat_steps: List[Tuple[str, BaseEstimator]] = []

            # Missing categorical imputation
            cat_impute_strategy = config.missing_categorical_strategy
            if cat_impute_strategy == "drop":
                cat_impute_strategy = "most_frequent"

            if cat_impute_strategy == "constant":
                cat_steps.append((
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value=config.missing_categorical_fill_value or "missing"),
                ))
            else:
                cat_steps.append((
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ))

            # Encoding
            enc_strategy = config.encoding_strategy
            if enc_strategy == "auto":
                # Check cardinalities
                high_card = any(df[c].nunique() > 20 for c in categorical_cols if c in df.columns)
                enc_strategy = "ordinal" if high_card else "onehot"

            if enc_strategy == "onehot":
                cat_steps.append((
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ))
            elif enc_strategy == "ordinal":
                cat_steps.append((
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ))

            transformers.append(("categorical", Pipeline(cat_steps), categorical_cols))

        if not transformers:
            raise ValidationError("Could not construct any feature transformers from the provided columns.")

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        # 6. Optional constant-column removal
        pipe_steps: List[Tuple[str, Any]] = [("preprocessor", preprocessor)]
        if config.remove_constant_columns:
            pipe_steps.append(("variance_thresh", VarianceThreshold(threshold=0.0)))

        # 7. Estimator instantiation
        estimator = self.registry.create_estimator(
            config.algorithm_id,
            sanitized_params,
            task=config.task,
        )
        pipe_steps.append(("model", estimator))

        full_pipeline = Pipeline(pipe_steps)

        active_feature_cols = [c for c in feature_cols if c in (set(numeric_cols) | set(categorical_cols))]

        return PipelineBuildResult(
            pipeline=full_pipeline,
            feature_columns=active_feature_cols,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            target_column=target_col,
            algorithm_info=algo_info,
            sanitized_hyperparameters=sanitized_params,
        )
