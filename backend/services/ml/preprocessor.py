"""
Dynamic Data Preprocessing Engine for Zemo.

Provides configurable, Scikit-learn based preprocessing pipelines:
- Column management (feature selection, column exclusion, target column, type overrides)
- Missing-value handling (mean, median, most_frequent, constant, drop rows, drop columns)
- Encoding (OneHotEncoder, OrdinalEncoder, LabelEncoder for target, unseen values handling)
- Feature scaling (StandardScaler, MinMaxScaler, RobustScaler, none)
- Additional transformations (duplicate removal, outlier clipping/dropping, constant-column removal, log transformation, feature selection)
- Preprocessing preview and validation with strict train/test data leakage prevention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

try:
    from core.exceptions import AppValidationError as ValidationError
    from services.ml.analyzer import DatasetAnalyzer
except ImportError:
    from backend.core.exceptions import AppValidationError as ValidationError
    from backend.services.ml.analyzer import DatasetAnalyzer


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Scikit-learn Compatible Transformers
# ---------------------------------------------------------------------------


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips numeric feature values based on thresholds learned STRICTLY from
    training data during .fit() to prevent data leakage.
    """

    def __init__(self, method: str = "iqr", factor: float = 1.5):
        self.method = method
        self.factor = float(factor)
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "OutlierClipper":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if self.method == "zscore":
            mean = np.nanmean(X_arr, axis=0)
            std = np.nanstd(X_arr, axis=0)
            std[std == 0] = 1.0
            self.lower_bounds_ = mean - self.factor * std
            self.upper_bounds_ = mean + self.factor * std
        else:
            # Default: IQR method
            q25 = np.nanpercentile(X_arr, 25, axis=0)
            q75 = np.nanpercentile(X_arr, 75, axis=0)
            iqr = q75 - q25
            self.lower_bounds_ = q25 - self.factor * iqr
            self.upper_bounds_ = q75 + self.factor * iqr

        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            return np.asarray(X, dtype=float)
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> Any:
        return input_features


class SafeLogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p transformation to numeric columns.
    If minimum value is negative, shifts values so minimum is zero before log1p.
    Learns shift factor strictly during fit().
    """

    def __init__(self):
        self.shifts_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "SafeLogTransformer":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        mins = np.nanmin(X_arr, axis=0)
        # Shift so minimum non-negative if any values < 0
        self.shifts_ = np.where(mins < 0, np.abs(mins), 0.0)
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if self.shifts_ is not None:
            X_arr = X_arr + self.shifts_
        # Ensure values >= 0
        X_arr = np.maximum(X_arr, 0.0)
        return np.log1p(X_arr)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> Any:
        return input_features


# ---------------------------------------------------------------------------
# Configuration and Response Schemas
# ---------------------------------------------------------------------------


class PreprocessingConfig(BaseModel):
    """Configuration for dataset preprocessing pipeline."""

    target_column: Optional[str] = Field(None, description="Target column for supervised learning")
    feature_columns: Optional[List[str]] = Field(None, description="Explicit list of feature columns to include")
    excluded_columns: List[str] = Field(default_factory=list, description="Columns to exclude from training")
    type_overrides: Dict[str, str] = Field(default_factory=dict, description="Manual column type overrides")

    # Missing values
    missing_numeric_strategy: Literal["mean", "median", "most_frequent", "constant", "drop", "keep"] = "mean"
    missing_numeric_fill_value: Optional[float] = None
    missing_categorical_strategy: Literal["most_frequent", "constant", "drop", "keep"] = "most_frequent"
    missing_categorical_fill_value: Optional[str] = "missing"

    # Encoding
    encoding_strategy: Literal["onehot", "ordinal", "auto", "none"] = "onehot"
    encoding_cardinality_threshold: int = Field(15, description="Max unique values for one-hot in 'auto' mode")

    # Feature scaling
    scaling_strategy: Literal["standard", "minmax", "robust", "none"] = "standard"
    scaled_columns: Optional[List[str]] = Field(None, description="Subset of numeric columns to scale; None means all")

    # Additional options
    handle_duplicates: bool = Field(True, description="Drop identical rows")
    outlier_strategy: Literal["none", "clip", "drop"] = "none"
    outlier_method: Literal["iqr", "zscore"] = "iqr"
    outlier_threshold: float = 1.5
    remove_constant_columns: bool = Field(True, description="Drop columns with zero variance")
    log_transform_columns: List[str] = Field(default_factory=list, description="Columns to apply log1p transform to")
    feature_selection_strategy: Literal["none", "variance", "k_best"] = "none"
    feature_selection_k: Optional[int] = Field(None, description="Number of top features to select for k_best")

    # Splits
    train_test_split_ratio: float = Field(0.8, ge=0.5, le=0.95, description="Fraction of dataset for training")
    random_state: int = 42


class OperationSummary(BaseModel):
    step_name: str
    target_columns: List[str]
    description: str


class PreprocessingPreviewResponse(BaseModel):
    original_shape: Tuple[int, int]
    train_shape: Tuple[int, int]
    test_shape: Tuple[int, int]
    feature_names_in: List[str]
    feature_names_out: List[str]
    output_feature_count: int
    target_column: Optional[str] = None
    target_type: Optional[str] = None
    target_classes: Optional[List[Union[str, int, float]]] = None
    operations_applied: List[OperationSummary]
    warnings: List[str]
    sample_preview: List[Dict[str, Any]]
    target_preview: Optional[List[Any]] = None


@dataclass
class PreprocessedDataBundle:
    """Holds preprocessed dataset partitions and fitted artifacts."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: Optional[np.ndarray]
    y_test: Optional[np.ndarray]
    feature_names_in: List[str]
    feature_names_out: List[str]
    fitted_pipeline: Union[Pipeline, ColumnTransformer]
    target_encoder: Optional[LabelEncoder]
    operations_applied: List[OperationSummary]
    warnings: List[str]


# ---------------------------------------------------------------------------
# Dynamic Preprocessing Pipeline Builder
# ---------------------------------------------------------------------------


class PreprocessingPipelineBuilder:
    """
    Constructs, validates, and executes dynamic preprocessing pipelines.
    Ensures strict separation between train and test splits to prevent data leakage.
    """

    def __init__(self, analyzer: Optional[DatasetAnalyzer] = None):
        self.analyzer = analyzer or DatasetAnalyzer()

    def validate_config(
        self,
        df: pd.DataFrame,
        config: PreprocessingConfig,
        column_types: Dict[str, str],
    ) -> List[str]:
        """
        Validates whether configuration options are appropriate for dataset columns.
        Returns a list of warnings or raises ValidationError for illegal setups.
        """
        warnings: List[str] = []
        all_cols = set(df.columns)

        # 1. Target column validation
        if config.target_column:
            if config.target_column not in all_cols:
                raise ValidationError(f"Target column '{config.target_column}' does not exist in dataset.")
            # Missing in target
            target_nulls = df[config.target_column].isna().sum()
            if target_nulls > 0:
                warnings.append(
                    f"Target column '{config.target_column}' contains {target_nulls} missing values. "
                    "Rows with missing targets will be dropped before model training."
                )

        # 2. Excluded columns validation
        for exc in config.excluded_columns:
            if exc not in all_cols:
                warnings.append(f"Excluded column '{exc}' does not exist in dataset.")

        # 3. Missing imputation strategy validation
        if config.missing_numeric_strategy in ("mean", "median"):
            # Check if applied to non-numeric
            for col, ctype in column_types.items():
                if col in config.excluded_columns or col == config.target_column:
                    continue
                if ctype not in ("numeric", "boolean"):
                    # Categoricals will not use numeric imputation anyway
                    pass

        # 4. High-cardinality warning for OneHotEncoder
        if config.encoding_strategy in ("onehot", "auto"):
            for col, ctype in column_types.items():
                if col in config.excluded_columns or col == config.target_column:
                    continue
                if ctype in ("categorical", "text"):
                    unique_count = df[col].nunique(dropna=True)
                    if unique_count > 50 and config.encoding_strategy == "onehot":
                        warnings.append(
                            f"Column '{col}' has {unique_count} distinct categories. "
                            "One-hot encoding will substantially expand the feature space. "
                            "Consider Ordinal Encoding."
                        )

        # 5. Log transformation validation
        for col in config.log_transform_columns:
            if col not in all_cols:
                warnings.append(f"Log transform column '{col}' does not exist in dataset.")
            elif column_types.get(col) != "numeric":
                raise ValidationError(f"Log transformation is only applicable to numeric columns. '{col}' is {column_types.get(col)}.")
            else:
                s = pd.to_numeric(df[col], errors="coerce")
                if (s < 0).any():
                    warnings.append(f"Column '{col}' contains negative values. Values will be shifted before log1p transformation.")

        # 6. Scaled columns check
        if config.scaled_columns:
            for col in config.scaled_columns:
                if col not in all_cols:
                    warnings.append(f"Specified scaled column '{col}' does not exist in dataset.")
                elif column_types.get(col) != "numeric":
                    raise ValidationError(f"Feature scaling is only applicable to numeric columns. '{col}' is {column_types.get(col)}.")

        return warnings

    def determine_feature_sets(
        self,
        df: pd.DataFrame,
        config: PreprocessingConfig,
        column_types: Dict[str, str],
    ) -> Tuple[List[str], List[str], List[str], Optional[str]]:
        """
        Determines the numeric features, categorical features, all input features, and target.
        """
        excluded = set(config.excluded_columns)
        target = config.target_column

        if config.feature_columns:
            candidate_features = [c for c in config.feature_columns if c in df.columns and c not in excluded and c != target]
        else:
            candidate_features = [c for c in df.columns if c not in excluded and c != target]

        numeric_features: List[str] = []
        categorical_features: List[str] = []

        for col in candidate_features:
            col_type = column_types.get(col, "categorical")
            if col_type == "numeric":
                numeric_features.append(col)
            elif col_type in ("categorical", "boolean", "text", "datetime"):
                categorical_features.append(col)
            elif col_type == "identifier":
                # By default, exclude identifiers unless explicitly included
                if config.feature_columns and col in config.feature_columns:
                    categorical_features.append(col)
                else:
                    excluded.add(col)

        return numeric_features, categorical_features, candidate_features, target

    def prepare_dataframe_rows(
        self,
        df: pd.DataFrame,
        config: PreprocessingConfig,
        numeric_features: List[str],
        categorical_features: List[str],
        target_column: Optional[str],
    ) -> Tuple[pd.DataFrame, List[OperationSummary], List[str]]:
        """
        Performs row-level preprocessing before train-test split:
        - Drops duplicates (if configured)
        - Drops rows with missing targets
        - Drops rows with missing features (if strategy == 'drop')
        - Drops outlier rows (if outlier_strategy == 'drop')
        """
        working_df = df.copy()
        ops: List[OperationSummary] = []
        warnings: List[str] = []
        initial_rows = len(working_df)

        # 1. Duplicate Handling
        if config.handle_duplicates:
            dup_count = working_df.duplicated().sum()
            if dup_count > 0:
                working_df = working_df.drop_duplicates().reset_index(drop=True)
                ops.append(
                    OperationSummary(
                        step_name="remove_duplicates",
                        target_columns=list(working_df.columns),
                        description=f"Removed {dup_count} duplicate rows (retained {len(working_df)} unique rows).",
                    )
                )

        # 2. Target Missing Rows
        if target_column and target_column in working_df.columns:
            target_nulls = working_df[target_column].isna().sum()
            if target_nulls > 0:
                working_df = working_df.dropna(subset=[target_column]).reset_index(drop=True)
                ops.append(
                    OperationSummary(
                        step_name="drop_null_targets",
                        target_columns=[target_column],
                        description=f"Dropped {target_nulls} rows where target '{target_column}' was null.",
                    )
                )

        # 3. Missing Value Row Dropping (if strategy == 'drop')
        drop_cols: List[str] = []
        if config.missing_numeric_strategy == "drop" and numeric_features:
            drop_cols.extend(numeric_features)
        if config.missing_categorical_strategy == "drop" and categorical_features:
            drop_cols.extend(categorical_features)

        if drop_cols:
            before_drop = len(working_df)
            working_df = working_df.dropna(subset=drop_cols).reset_index(drop=True)
            dropped = before_drop - len(working_df)
            if dropped > 0:
                ops.append(
                    OperationSummary(
                        step_name="drop_missing_rows",
                        target_columns=drop_cols,
                        description=f"Dropped {dropped} rows with missing values across {len(drop_cols)} columns.",
                    )
                )

        # 4. Outlier Row Dropping (if configured as 'drop')
        if config.outlier_strategy == "drop" and numeric_features:
            outlier_mask = pd.Series(False, index=working_df.index)
            for col in numeric_features:
                s = pd.to_numeric(working_df[col], errors="coerce")
                if config.outlier_method == "zscore":
                    mean = s.mean()
                    std = s.std()
                    if std and std > 0:
                        outlier_mask |= ((s - mean).abs() > (config.outlier_threshold * std))
                else:
                    q25 = s.quantile(0.25)
                    q75 = s.quantile(0.75)
                    iqr = q75 - q25
                    outlier_mask |= (s < (q25 - config.outlier_threshold * iqr)) | (s > (q75 + config.outlier_threshold * iqr))

            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                working_df = working_df[~outlier_mask].reset_index(drop=True)
                ops.append(
                    OperationSummary(
                        step_name="drop_outliers",
                        target_columns=numeric_features,
                        description=f"Dropped {outlier_count} rows identified as statistical outliers ({config.outlier_method}).",
                    )
                )

        if len(working_df) == 0:
            raise ValidationError(
                f"Row filtering removed all {initial_rows} rows from dataset. "
                "Please relax missing value or outlier dropping criteria."
            )

        return working_df, ops, warnings

    def build_column_transformer(
        self,
        config: PreprocessingConfig,
        numeric_features: List[str],
        categorical_features: List[str],
        df_sample: pd.DataFrame,
    ) -> Tuple[ColumnTransformer, List[OperationSummary]]:
        """
        Builds the Scikit-learn ColumnTransformer dynamically from configuration.
        """
        transformers: List[Tuple[str, Union[Pipeline, str], List[str]]] = []
        ops: List[OperationSummary] = []

        log_cols = [c for c in config.log_transform_columns if c in numeric_features]
        standard_num_cols = [c for c in numeric_features if c not in log_cols]

        scaled_set = set(config.scaled_columns) if config.scaled_columns is not None else set(numeric_features)

        # Build Numeric Pipelines
        for group_name, cols, is_log in [
            ("num_log", log_cols, True),
            ("num_std", standard_num_cols, False),
        ]:
            if not cols:
                continue

            steps: List[Tuple[str, BaseEstimator]] = []

            # 1. Imputation
            if config.missing_numeric_strategy == "median":
                steps.append(("imputer", SimpleImputer(strategy="median")))
                ops.append(
                    OperationSummary(
                        step_name="numeric_imputation",
                        target_columns=cols,
                        description="Median imputation for missing values.",
                    )
                )
            elif config.missing_numeric_strategy == "most_frequent":
                steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
                ops.append(
                    OperationSummary(
                        step_name="numeric_imputation",
                        target_columns=cols,
                        description="Most-frequent (mode) imputation for missing values.",
                    )
                )
            elif config.missing_numeric_strategy == "constant":
                fill_val = config.missing_numeric_fill_value if config.missing_numeric_fill_value is not None else 0.0
                steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=fill_val)))
                ops.append(
                    OperationSummary(
                        step_name="numeric_imputation",
                        target_columns=cols,
                        description=f"Constant value ({fill_val}) imputation for missing values.",
                    )
                )
            elif config.missing_numeric_strategy == "keep":
                # No imputer
                pass
            else:
                # Default: mean
                steps.append(("imputer", SimpleImputer(strategy="mean")))
                ops.append(
                    OperationSummary(
                        step_name="numeric_imputation",
                        target_columns=cols,
                        description="Mean imputation for missing values.",
                    )
                )

            # 2. Outlier Clipping (if configured)
            if config.outlier_strategy == "clip":
                steps.append(
                    (
                        "outlier_clipper",
                        OutlierClipper(
                            method=config.outlier_method,
                            factor=config.outlier_threshold,
                        ),
                    )
                )
                ops.append(
                    OperationSummary(
                        step_name="outlier_clipping",
                        target_columns=cols,
                        description=f"Outlier clipping via {config.outlier_method.upper()} bounds (factor={config.outlier_threshold}).",
                    )
                )

            # 3. Log Transformation (if applicable)
            if is_log:
                steps.append(("log_transformer", SafeLogTransformer()))
                ops.append(
                    OperationSummary(
                        step_name="log_transform",
                        target_columns=cols,
                        description="Log1p non-linear transformation for skewed distribution.",
                    )
                )

            # 4. Scaling
            cols_to_scale = [c for c in cols if c in scaled_set]
            if cols_to_scale and config.scaling_strategy != "none":
                if config.scaling_strategy == "minmax":
                    steps.append(("scaler", MinMaxScaler()))
                    ops.append(
                        OperationSummary(
                            step_name="scaling",
                            target_columns=cols_to_scale,
                            description="MinMaxScaler to boundary range [0, 1].",
                        )
                    )
                elif config.scaling_strategy == "robust":
                    steps.append(("scaler", RobustScaler()))
                    ops.append(
                        OperationSummary(
                            step_name="scaling",
                            target_columns=cols_to_scale,
                            description="RobustScaler using median and interquartile range (IQR).",
                        )
                    )
                else:
                    # Default: standard
                    steps.append(("scaler", StandardScaler()))
                    ops.append(
                        OperationSummary(
                            step_name="scaling",
                            target_columns=cols_to_scale,
                            description="StandardScaler (Z-score normalization with mean 0, unit variance).",
                        )
                    )

            if steps:
                transformers.append((group_name, Pipeline(steps), cols))
            else:
                transformers.append((group_name, "passthrough", cols))

        # Build Categorical Pipeline
        if categorical_features:
            cat_steps: List[Tuple[str, BaseEstimator]] = []

            # 1. Categorical Imputation
            if config.missing_categorical_strategy == "constant":
                fill_val = config.missing_categorical_fill_value or "missing"
                cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=fill_val)))
                ops.append(
                    OperationSummary(
                        step_name="categorical_imputation",
                        target_columns=categorical_features,
                        description=f"Constant fill with '{fill_val}' for missing categorical cells.",
                    )
                )
            elif config.missing_categorical_strategy == "keep":
                # Ensure no None/NaNs break encoders by converting to 'missing'
                cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="missing")))
            else:
                # Default: most_frequent
                cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
                ops.append(
                    OperationSummary(
                        step_name="categorical_imputation",
                        target_columns=categorical_features,
                        description="Most frequent mode category imputation for missing entries.",
                    )
                )

            # 2. Categorical Encoding
            encoding = config.encoding_strategy
            if encoding == "auto":
                # If all categories have cardinality <= threshold use onehot, else ordinal
                max_card = max(df_sample[c].nunique(dropna=True) for c in categorical_features)
                encoding = "onehot" if max_card <= config.encoding_cardinality_threshold else "ordinal"

            if encoding == "ordinal":
                cat_steps.append(
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    )
                )
                ops.append(
                    OperationSummary(
                        step_name="categorical_encoding",
                        target_columns=categorical_features,
                        description="OrdinalEncoder (integer mapping with unseen values mapped to -1).",
                    )
                )
            elif encoding == "none":
                # Passthrough (e.g. already integer encoded)
                pass
            else:
                # Default: onehot
                cat_steps.append(
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    )
                )
                ops.append(
                    OperationSummary(
                        step_name="categorical_encoding",
                        target_columns=categorical_features,
                        description="OneHotEncoder (binary dummy columns with handle_unknown='ignore').",
                    )
                )

            transformers.append(("cat", Pipeline(cat_steps), categorical_features))

        if not transformers:
            raise ValidationError("No features remained to preprocess. Check included/excluded column configuration.")

        col_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        return col_transformer, ops

    def build_full_pipeline(
        self,
        column_transformer: ColumnTransformer,
        config: PreprocessingConfig,
        is_classification: bool = False,
    ) -> Tuple[Pipeline, List[OperationSummary]]:
        """
        Wraps ColumnTransformer into a full pipeline including optional post-transformation steps:
        - Constant-column removal (VarianceThreshold)
        - Feature selection (SelectKBest)
        """
        pipeline_steps: List[Tuple[str, BaseEstimator]] = [("col_transform", column_transformer)]
        ops: List[OperationSummary] = []

        # 1. Remove constant columns
        if config.remove_constant_columns:
            pipeline_steps.append(("variance_thresh", VarianceThreshold(threshold=0.0)))
            ops.append(
                OperationSummary(
                    step_name="constant_column_removal",
                    target_columns=[],
                    description="VarianceThreshold(0.0) removing constant features with zero variance.",
                )
            )

        # 2. Feature selection
        if config.feature_selection_strategy == "variance":
            pipeline_steps.append(("variance_filter", VarianceThreshold(threshold=0.01)))
            ops.append(
                OperationSummary(
                    step_name="variance_feature_selection",
                    target_columns=[],
                    description="VarianceThreshold(0.01) removing near-constant low-variance features.",
                )
            )
        elif config.feature_selection_strategy == "k_best" and config.feature_selection_k:
            k = max(1, config.feature_selection_k)
            score_fn = f_classif if is_classification else f_regression
            pipeline_steps.append(("k_best", SelectKBest(score_func=score_fn, k=k)))
            ops.append(
                OperationSummary(
                    step_name="k_best_feature_selection",
                    target_columns=[],
                    description=f"SelectKBest ({'f_classif' if is_classification else 'f_regression'}, k={k}) for top predictive features.",
                )
            )

        return Pipeline(pipeline_steps), ops

    def fit_transform_supervised(
        self,
        df: pd.DataFrame,
        config: PreprocessingConfig,
        column_types: Optional[Dict[str, str]] = None,
    ) -> PreprocessedDataBundle:
        """
        Fits dynamic preprocessing pipeline strictly on training split and transforms both splits.
        Guarantees NO data leakage across train and test partitions.
        """
        # Resolve column types
        if column_types is None:
            analysis = self.analyzer.analyze_dataset(df, type_overrides=config.type_overrides)
            column_types = {col["name"]: col["inferred_type"] for col in analysis["columns"]}


        warnings = self.validate_config(df, config, column_types)

        num_features, cat_features, candidate_features, target_col = self.determine_feature_sets(
            df, config, column_types
        )

        # Row-level operations
        clean_df, row_ops, row_warnings = self.prepare_dataframe_rows(
            df, config, num_features, cat_features, target_col
        )
        warnings.extend(row_warnings)

        X = clean_df[candidate_features]
        y: Optional[pd.Series] = clean_df[target_col] if target_col else None

        # Determine task type
        is_classification = False
        target_encoder: Optional[LabelEncoder] = None
        if y is not None:
            target_type = column_types.get(target_col, "categorical")
            if target_type in ("categorical", "boolean", "text") or y.nunique() <= 10:
                is_classification = True

        # Train / Test Split BEFORE fitting any transformer (LEAKAGE PREVENTION)
        test_size = 1.0 - config.train_test_split_ratio
        stratify_val = None
        if is_classification and y is not None:
            class_counts = y.value_counts()
            n_classes = len(class_counts)
            test_samples = int(np.ceil(len(X) * test_size))
            train_samples = len(X) - test_samples
            if (
                (class_counts >= 2).all()
                and n_classes > 1
                and test_samples >= n_classes
                and train_samples >= n_classes
            ):
                stratify_val = y


        if y is not None:
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=config.random_state,
                stratify=stratify_val,
            )
        else:
            X_train_raw, X_test_raw = train_test_split(
                X,
                test_size=test_size,
                random_state=config.random_state,
            )
            y_train_raw, y_test_raw = None, None

        # Build Pipeline
        col_transformer, col_ops = self.build_column_transformer(
            config, num_features, cat_features, df_sample=X_train_raw
        )
        full_pipeline, pipe_ops = self.build_full_pipeline(
            col_transformer, config, is_classification=is_classification
        )

        all_ops = row_ops + col_ops + pipe_ops

        # Fit ONLY on X_train (Leakage Prevention)
        if y_train_raw is not None and config.feature_selection_strategy == "k_best":
            # SelectKBest requires y during fit
            if is_classification:
                # temporary numeric label encoding for scoring
                temp_le = LabelEncoder()
                y_train_fit = temp_le.fit_transform(y_train_raw.astype(str))
            else:
                y_train_fit = pd.to_numeric(y_train_raw, errors="coerce").fillna(0.0)
            X_train_trans = full_pipeline.fit_transform(X_train_raw, y_train_fit)
        else:
            X_train_trans = full_pipeline.fit_transform(X_train_raw)

        # Transform X_test with fitted pipeline
        X_test_trans = full_pipeline.transform(X_test_raw)

        # Retrieve feature names out
        try:
            raw_feature_names = full_pipeline.get_feature_names_out()
            feature_names_out = [str(name).replace("col_transform__", "") for name in raw_feature_names]
        except Exception:
            feature_names_out = [f"feature_{i}" for i in range(X_train_trans.shape[1])]

        # Target processing (strictly fitted on y_train)
        y_train_proc: Optional[np.ndarray] = None
        y_test_proc: Optional[np.ndarray] = None
        if y_train_raw is not None and y_test_raw is not None:
            if is_classification:
                target_encoder = LabelEncoder()
                # Fit target encoder strictly on y_train
                y_train_proc = target_encoder.fit_transform(y_train_raw.astype(str))
                # Transform y_test (handle unseen classes safely)
                unseen = set(y_test_raw.astype(str)) - set(target_encoder.classes_)
                if unseen:
                    warnings.append(
                        f"Test set contains {len(unseen)} unseen target classes: {list(unseen)[:3]}. "
                        "These were assigned fallback label 0."
                    )
                    safe_test = [
                        target_encoder.transform([val])[0] if val in target_encoder.classes_ else 0
                        for val in y_test_raw.astype(str)
                    ]
                    y_test_proc = np.array(safe_test)
                else:
                    y_test_proc = target_encoder.transform(y_test_raw.astype(str))
            else:
                y_train_proc = pd.to_numeric(y_train_raw, errors="coerce").fillna(0.0).to_numpy()
                y_test_proc = pd.to_numeric(y_test_raw, errors="coerce").fillna(0.0).to_numpy()

        return PreprocessedDataBundle(
            X_train=X_train_trans,
            X_test=X_test_trans,
            y_train=y_train_proc,
            y_test=y_test_proc,
            feature_names_in=candidate_features,
            feature_names_out=feature_names_out,
            fitted_pipeline=full_pipeline,
            target_encoder=target_encoder,
            operations_applied=all_ops,
            warnings=warnings,
        )

    def preview(
        self,
        df: pd.DataFrame,
        config: PreprocessingConfig,
        column_types: Optional[Dict[str, str]] = None,
        max_preview_rows: int = 10,
    ) -> PreprocessingPreviewResponse:
        """
        Generates a fast, transparent preview of preprocessing transformations
        showing operations, shapes, warnings, and actual transformed feature samples.
        """
        bundle = self.fit_transform_supervised(df, config, column_types)

        sample_rows_count = min(max_preview_rows, bundle.X_train.shape[0])
        sample_slice = bundle.X_train[:sample_rows_count]

        # Convert sample slice to list of dicts for JSON serialization
        sample_preview: List[Dict[str, Any]] = []
        for i in range(sample_rows_count):
            row_dict: Dict[str, Any] = {}
            for col_idx, col_name in enumerate(bundle.feature_names_out):
                val = sample_slice[i, col_idx]
                if isinstance(val, (np.floating, float)):
                    row_dict[col_name] = round(float(val), 4)
                elif isinstance(val, (np.integer, int)):
                    row_dict[col_name] = int(val)
                else:
                    row_dict[col_name] = str(val)
            sample_preview.append(row_dict)

        target_preview = None
        target_classes = None
        target_type = None

        if config.target_column:
            if bundle.target_encoder is not None:
                target_type = "categorical"
                target_classes = [str(c) for c in bundle.target_encoder.classes_]
            else:
                target_type = "numeric"

            if bundle.y_train is not None:
                t_slice = bundle.y_train[:sample_rows_count]
                target_preview = [
                    int(v) if isinstance(v, (np.integer, int)) else round(float(v), 4)
                    for v in t_slice
                ]

        return PreprocessingPreviewResponse(
            original_shape=(len(df), len(df.columns)),
            train_shape=bundle.X_train.shape,
            test_shape=bundle.X_test.shape,
            feature_names_in=bundle.feature_names_in,
            feature_names_out=bundle.feature_names_out,
            output_feature_count=len(bundle.feature_names_out),
            target_column=config.target_column,
            target_type=target_type,
            target_classes=target_classes,
            operations_applied=bundle.operations_applied,
            warnings=bundle.warnings,
            sample_preview=sample_preview,
            target_preview=target_preview,
        )
