import pytest
import numpy as np
import pandas as pd
from httpx import AsyncClient, ASGITransport
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from main import app
from services.ml.preprocessor import (
    PreprocessingPipelineBuilder,
    PreprocessingConfig,
    OutlierClipper,
    SafeLogTransformer,
)
from core.database.config import get_database





@pytest.fixture
def sample_dataset():
    """Returns a rich dataset with mixed types, missing values, duplicates, outliers, and constants."""
    return pd.DataFrame({
        "id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 110],  # row 10 and 11 are duplicates
        "age": [25.0, 32.0, np.nan, 47.0, 52.0, 29.0, 150.0, 36.0, 41.0, 28.0, 28.0], # 150 is outlier, nan is missing
        "income": [50000, 62000, 48000, 85000, np.nan, 53000, 120000, 71000, 68000, 54000, 54000],
        "city": ["NY", "SF", "LA", "NY", "Chicago", "SF", "NY", "LA", "Chicago", "NY", "NY"],
        "tier": ["Gold", "Silver", "Bronze", "Silver", "Gold", "Bronze", "Gold", "Silver", "Bronze", "Gold", "Gold"],
        "constant_col": [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999], # constant column
        "purchased": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0] # target
    })


def test_column_management_and_exclusion(sample_dataset):
    builder = PreprocessingPipelineBuilder()
    config = PreprocessingConfig(
        target_column="purchased",
        excluded_columns=["id", "constant_col"],
        scaling_strategy="standard",
        encoding_strategy="onehot",
        train_test_split_ratio=0.8,
        random_state=42
    )

    bundle = builder.fit_transform_supervised(sample_dataset, config)
    assert "id" not in bundle.feature_names_in
    assert "constant_col" not in bundle.feature_names_in
    assert "purchased" not in bundle.feature_names_in
    assert "age" in bundle.feature_names_in
    assert "income" in bundle.feature_names_in
    assert "city" in bundle.feature_names_in


def test_missing_value_imputation_strategies(sample_dataset):
    builder = PreprocessingPipelineBuilder()

    # 1. Mean imputation
    config_mean = PreprocessingConfig(
        target_column="purchased",
        excluded_columns=["id", "constant_col"],
        missing_numeric_strategy="mean",
        scaling_strategy="none",
        encoding_strategy="ordinal",
    )
    bundle_mean = builder.fit_transform_supervised(sample_dataset, config_mean)
    assert not np.isnan(bundle_mean.X_train).any()
    assert not np.isnan(bundle_mean.X_test).any()

    # 2. Median imputation
    config_median = PreprocessingConfig(
        target_column="purchased",
        excluded_columns=["id", "constant_col"],
        missing_numeric_strategy="median",
        scaling_strategy="none",
        encoding_strategy="ordinal",
    )
    bundle_median = builder.fit_transform_supervised(sample_dataset, config_median)
    assert not np.isnan(bundle_median.X_train).any()

    # 3. Constant imputation
    config_constant = PreprocessingConfig(
        target_column="purchased",
        excluded_columns=["id", "constant_col"],
        missing_numeric_strategy="constant",
        missing_numeric_fill_value=-999.0,
        scaling_strategy="none",
        encoding_strategy="ordinal",
    )
    bundle_const = builder.fit_transform_supervised(sample_dataset, config_constant)
    assert not np.isnan(bundle_const.X_train).any()

    # 4. Drop rows with missing values
    config_drop = PreprocessingConfig(
        target_column="purchased",
        excluded_columns=["id", "constant_col"],
        missing_numeric_strategy="drop",
        scaling_strategy="none",
        encoding_strategy="ordinal",
    )
    bundle_drop = builder.fit_transform_supervised(sample_dataset, config_drop)
    # The duplicate is dropped, plus rows with NaN age and NaN income are dropped
    assert bundle_drop.X_train.shape[0] + bundle_drop.X_test.shape[0] < len(sample_dataset)


def test_categorical_encoding_onehot_and_ordinal(sample_dataset):
    builder = PreprocessingPipelineBuilder()

    # OneHot
    config_onehot = PreprocessingConfig(
        target_column="purchased",
        excluded_columns=["id", "constant_col"],
        encoding_strategy="onehot",
        scaling_strategy="none",
    )
    bundle_onehot = builder.fit_transform_supervised(sample_dataset, config_onehot)
    # Binary expansion should increase number of feature columns
    assert bundle_onehot.X_train.shape[1] > 4
    assert any("city" in name for name in bundle_onehot.feature_names_out)
    assert any("tier" in name for name in bundle_onehot.feature_names_out)

    # Ordinal
    config_ordinal = PreprocessingConfig(
        target_column="purchased",
        excluded_columns=["id", "constant_col"],
        encoding_strategy="ordinal",
        scaling_strategy="none",
    )
    bundle_ordinal = builder.fit_transform_supervised(sample_dataset, config_ordinal)
    # Ordinal keeps same number of columns as input features
    assert bundle_ordinal.X_train.shape[1] == 4


def test_feature_scaling_options(sample_dataset):
    builder = PreprocessingPipelineBuilder()

    for scaler in ["standard", "minmax", "robust", "none"]:
        config = PreprocessingConfig(
            target_column="purchased",
            excluded_columns=["id", "constant_col"],
            scaling_strategy=scaler,
            encoding_strategy="ordinal",
        )
        bundle = builder.fit_transform_supervised(sample_dataset, config)
        assert bundle.X_train.shape[0] > 0
        assert bundle.X_test.shape[0] > 0


def test_strict_data_leakage_prevention():
    """
    Verify that scalers and imputers are strictly fitted on X_train.
    If X_test has extreme outliers or different distributions, they must not alter X_train statistics.
    """
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 1000.0],
        "cat": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    builder = PreprocessingPipelineBuilder()
    config = PreprocessingConfig(
        target_column="target",
        scaling_strategy="standard",
        encoding_strategy="ordinal",
        train_test_split_ratio=0.8,
        random_state=42
    )

    bundle = builder.fit_transform_supervised(df, config)

    # The pipeline is fitted on X_train
    # Find the scaler step in the fitted pipeline
    col_trans = bundle.fitted_pipeline.named_steps["col_transform"]
    num_pipe = col_trans.named_transformers_["num_std"]
    scaler: StandardScaler = num_pipe.named_steps["scaler"]

    # Verify scaler mean_ was calculated ONLY from training indices
    assert scaler.n_samples_seen_ == 8
    assert len(bundle.y_train) == 8
    assert len(bundle.y_test) == 2


def test_outlier_clipping_and_log_transform():
    df = pd.DataFrame({
        "val": [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 100.0], # 100 is an extreme outlier
        "skewed": [10.0, 50.0, 200.0, 800.0, 3000.0, 12000.0, 50000.0, 200000.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1]
    })

    builder = PreprocessingPipelineBuilder()
    config = PreprocessingConfig(
        target_column="target",
        outlier_strategy="clip",
        outlier_method="iqr",
        outlier_threshold=1.5,
        log_transform_columns=["skewed"],
        scaling_strategy="minmax",
        train_test_split_ratio=0.75,
        random_state=42
    )

    bundle = builder.fit_transform_supervised(df, config)
    assert bundle.X_train.shape[1] == 2
    # Verify no infinite or NaN values after log and minmax
    assert np.isfinite(bundle.X_train).all()
    assert np.isfinite(bundle.X_test).all()


def test_preview_generation_response(sample_dataset):
    builder = PreprocessingPipelineBuilder()
    config = PreprocessingConfig(
        target_column="purchased",
        excluded_columns=["id"],
        scaling_strategy="standard",
        encoding_strategy="onehot",
        train_test_split_ratio=0.8,
        random_state=42
    )

    preview = builder.preview(sample_dataset, config, max_preview_rows=5)

    assert preview.original_shape == (11, 7)
    assert preview.train_shape[0] == 8
    assert preview.test_shape[0] == 2
    assert preview.output_feature_count > 0
    assert len(preview.sample_preview) == 5
    assert len(preview.operations_applied) > 0
    assert preview.target_column == "purchased"
    assert preview.target_type == "categorical"


@pytest.mark.asyncio
async def test_preprocessing_api_endpoints(async_client: AsyncClient):
    # 1. Upload a CSV dataset
    csv_data = b"age,income,tier,target\n25,50000,Gold,1\n35,70000,Silver,0\n45,90000,Bronze,1\n55,120000,Gold,0\n65,150000,Silver,1\n"
    files = {"file": ("test_preprocess.csv", csv_data, "text/csv")}
    data = {"name": "Test Preprocess Dataset"}

    create_res = await async_client.post("/api/v1/datasets", files=files, data=data)
    assert create_res.status_code == 201
    dataset_id = create_res.json()["id"]

    # 2. Trigger processing
    proc_res = await async_client.post(f"/api/v1/datasets/{dataset_id}/process")
    assert proc_res.status_code == 200

    # 3. Test Preview Endpoint
    preview_req = {
        "target_column": "target",
        "scaling_strategy": "standard",
        "encoding_strategy": "onehot",
        "train_test_split_ratio": 0.8
    }
    prev_res = await async_client.post(f"/api/v1/datasets/{dataset_id}/preprocess/preview", json=preview_req)
    assert prev_res.status_code == 200
    prev_json = prev_res.json()
    assert prev_json["output_feature_count"] > 0
    assert len(prev_json["sample_preview"]) > 0
    assert prev_json["target_column"] == "target"

    # 4. Test Validation Endpoint
    val_req = {
        "target_column": "target",
        "log_transform_columns": ["age"],
        "scaling_strategy": "robust"
    }
    val_res = await async_client.post(f"/api/v1/datasets/{dataset_id}/preprocess/validate", json=val_req)
    assert val_res.status_code == 200
    val_json = val_res.json()
    assert val_json["valid"] is True
    assert "detected_types" in val_json

    # 5. Test Config Save and Retrieve
    cfg_save_res = await async_client.post(f"/api/v1/datasets/{dataset_id}/preprocess/config", json=preview_req)
    assert cfg_save_res.status_code == 200

    cfg_get_res = await async_client.get(f"/api/v1/datasets/{dataset_id}/preprocess/config")
    assert cfg_get_res.status_code == 200
    saved_cfg = cfg_get_res.json()["config"]
    assert saved_cfg["scaling_strategy"] == "standard"
    assert saved_cfg["encoding_strategy"] == "onehot"
