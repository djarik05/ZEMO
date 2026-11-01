"""
Tests for Phase 4: Dynamic ML Pipeline Builder, Training Engine, Background Job Manager, and API.
"""

import asyncio
import io
import json
import os
import tempfile
import time
from typing import AsyncGenerator

import joblib
import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sklearn.pipeline import Pipeline

from core.database.config import Database, get_database
from core.database.models.dataset import Dataset, DatasetFormat, DatasetStatus
from core.database.models.experiment import Experiment, ExperimentStatus
from main import create_app
from services.ml.pipeline_builder import DynamicPipelineBuilder, TrainingConfig
from services.ml.trainer import MLTrainingEngine


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.uniform(20000, 120000, size=n),
        "category": np.random.choice(["retail", "tech", "finance"], size=n),
        "subscribed": np.random.choice(["yes", "no"], size=n, p=[0.4, 0.6]),
    })


@pytest.fixture
def sample_regression_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    sqft = np.random.uniform(500, 3500, size=n)
    bedrooms = np.random.randint(1, 6, size=n)
    neighborhood = np.random.choice(["suburban", "urban", "rural"], size=n)
    price = 50000 + (sqft * 150) + (bedrooms * 20000) + np.random.normal(0, 10000, size=n)
    return pd.DataFrame({
        "sqft": sqft,
        "bedrooms": bedrooms,
        "neighborhood": neighborhood,
        "price": price,
    })


def test_dynamic_pipeline_builder_construction(sample_classification_df):
    """Verify builder produces Scikit-learn Pipeline with ColumnTransformer and estimator."""
    builder = DynamicPipelineBuilder()
    config = TrainingConfig(
        dataset_id="test-ds-1",
        task="classification",
        target_column="subscribed",
        algorithm_id="random-forest",
        scaling_strategy="standard",
        encoding_strategy="onehot",
        hyperparameters={"n_estimators": 50, "max_depth": 5},
    )

    build_res = builder.build_pipeline(sample_classification_df, config)
    assert isinstance(build_res.pipeline, Pipeline)
    assert "preprocessor" in build_res.pipeline.named_steps
    assert "model" in build_res.pipeline.named_steps
    assert "subscribed" not in build_res.feature_columns
    assert "age" in build_res.numeric_columns
    assert "category" in build_res.categorical_columns


def test_different_user_configs_produce_different_pipelines(sample_classification_df):
    """Verify different user preprocessing choices yield distinct fitted pipelines."""
    builder = DynamicPipelineBuilder()

    # Config 1: Standard scaling + One-Hot encoding + Random Forest
    config_1 = TrainingConfig(
        dataset_id="test-ds-1",
        task="classification",
        target_column="subscribed",
        algorithm_id="random-forest",
        scaling_strategy="standard",
        encoding_strategy="onehot",
    )
    res_1 = builder.build_pipeline(sample_classification_df, config_1)

    # Config 2: MinMax scaling + Ordinal encoding + Logistic Regression
    config_2 = TrainingConfig(
        dataset_id="test-ds-1",
        task="classification",
        target_column="subscribed",
        algorithm_id="logistic-regression",
        scaling_strategy="minmax",
        encoding_strategy="ordinal",
    )
    res_2 = builder.build_pipeline(sample_classification_df, config_2)

    # Check estimator types differ
    assert type(res_1.pipeline.named_steps["model"]) != type(res_2.pipeline.named_steps["model"])

    # Check transformers differ
    prep_1 = res_1.pipeline.named_steps["preprocessor"]
    prep_2 = res_2.pipeline.named_steps["preprocessor"]

    trans_map_1 = {name: pipe for name, pipe, _ in prep_1.transformers}
    trans_map_2 = {name: pipe for name, pipe, _ in prep_2.transformers}

    num_pipe_1 = trans_map_1["numeric"]
    num_pipe_2 = trans_map_2["numeric"]

    scaler_1 = dict(num_pipe_1.steps)["scaler"].__class__.__name__
    scaler_2 = dict(num_pipe_2.steps)["scaler"].__class__.__name__

    assert scaler_1 == "StandardScaler"
    assert scaler_2 == "MinMaxScaler"


def test_train_classification_model(sample_classification_df, tmp_path):
    """Verify real training of a classification model produces metrics and a saved artifact."""
    engine = MLTrainingEngine(artifacts_dir=str(tmp_path))
    config = TrainingConfig(
        dataset_id="test-ds-1",
        task="classification",
        target_column="subscribed",
        algorithm_id="random-forest",
        scaling_strategy="standard",
        encoding_strategy="onehot",
        hyperparameters={"n_estimators": 25, "max_depth": 4},
        test_size=0.25,
        random_state=42,
    )

    result = engine.train("exp-test-clf", sample_classification_df, config)

    assert result.task == "classification"
    assert "accuracy" in result.metrics
    assert "f1" in result.metrics
    assert "confusion_matrix" in result.metrics
    assert 0.0 <= result.metrics["accuracy"] <= 1.0

    # Verify model artifact is loadable and can predict
    assert os.path.exists(result.artifact_path)
    loaded_pipe = joblib.load(result.artifact_path)
    assert isinstance(loaded_pipe, Pipeline)

    # Predict on new sample
    new_sample = pd.DataFrame({
        "age": [35],
        "income": [65000.0],
        "category": ["tech"],
    })
    preds = loaded_pipe.predict(new_sample)
    assert len(preds) == 1


def test_train_regression_model(sample_regression_df, tmp_path):
    """Verify real training of a regression model produces RMSE, MAE, R2, and saved artifact."""
    engine = MLTrainingEngine(artifacts_dir=str(tmp_path))
    config = TrainingConfig(
        dataset_id="test-ds-2",
        task="regression",
        target_column="price",
        algorithm_id="linear-regression",
        scaling_strategy="standard",
        encoding_strategy="onehot",
        cv_folds=3,
        test_size=0.2,
        random_state=42,
    )

    result = engine.train("exp-test-reg", sample_regression_df, config)

    assert result.task == "regression"
    assert "rmse" in result.metrics
    assert "r2" in result.metrics
    assert "mae" in result.metrics
    assert "cv_scores" in result.metrics
    assert len(result.metrics["cv_scores"]) == 3
    assert result.metrics["r2"] > 0.5  # Linear relationship in synthetic data

    assert os.path.exists(result.artifact_path)
    loaded_pipe = joblib.load(result.artifact_path)
    assert isinstance(loaded_pipe, Pipeline)

    new_house = pd.DataFrame({
        "sqft": [2000.0],
        "bedrooms": [3],
        "neighborhood": ["urban"],
    })
    pred_price = loaded_pipe.predict(new_house)
    assert pred_price[0] > 100000


def test_train_clustering_unsupervised_workflow(sample_regression_df, tmp_path):
    """Verify unsupervised clustering does not require target or supervised split."""
    engine = MLTrainingEngine(artifacts_dir=str(tmp_path))
    config = TrainingConfig(
        dataset_id="test-ds-3",
        task="clustering",
        target_column=None,  # No target!
        algorithm_id="kmeans",
        scaling_strategy="standard",
        encoding_strategy="onehot",
        hyperparameters={"n_clusters": 3, "random_state": 42},
    )

    result = engine.train("exp-test-cluster", sample_regression_df, config)

    assert result.task == "clustering"
    assert result.metrics["num_clusters"] == 3
    assert "cluster_distribution" in result.metrics
    assert "silhouette" in result.metrics
    assert -1.0 <= result.metrics["silhouette"] <= 1.0

    assert os.path.exists(result.artifact_path)


@pytest.mark.asyncio
async def test_experiments_api_workflow(sample_classification_df, async_client):
    """Full integration test of experiment endpoints from creation through training and download."""
    client = async_client
    # Upload dataset via API
    csv_buffer = io.BytesIO()
    sample_classification_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()

    upload_resp = await client.post(
        "/api/v1/datasets",
        files={"file": ("churn_integration.csv", csv_bytes, "text/csv")},
    )
    assert upload_resp.status_code == 201
    dataset_id = upload_resp.json()["id"]

    # Launch training
    train_payload = {
        "dataset_id": dataset_id,
        "name": "Integration Random Forest Test",
        "task": "classification",
        "target_column": "subscribed",
        "algorithm_id": "random-forest",
        "scaling_strategy": "standard",
        "encoding_strategy": "onehot",
        "hyperparameters": {"n_estimators": 10, "max_depth": 3},
        "test_size": 0.25,
        "random_state": 42,
    }

    resp = await client.post("/api/v1/experiments/train", json=train_payload)
    assert resp.status_code == 202
    data = resp.json()
    exp_id = data["id"]
    assert exp_id is not None
    assert data["status"] in ("pending", "running", "completed")

    # Poll status until completed (with timeout)
    completed = False
    for _ in range(30):
        await asyncio.sleep(0.2)
        check_resp = await client.get(f"/api/v1/experiments/{exp_id}")
        assert check_resp.status_code == 200
        check_data = check_resp.json()
        if check_data["status"] == "completed":
            completed = True
            assert "accuracy" in check_data["metrics"]
            assert len(check_data["logs"]) > 0
            break
        elif check_data["status"] == "failed":
            pytest.fail(f"Experiment failed: {check_data.get('error_message')}")

    assert completed, "Experiment did not complete within expected time."

    # List experiments
    list_resp = await client.get("/api/v1/experiments")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    assert any(e["id"] == exp_id for e in list_data["experiments"])

    # Download trained artifact
    dl_resp = await client.get(f"/api/v1/experiments/{exp_id}/download")
    assert dl_resp.status_code == 200
    assert len(dl_resp.content) > 100

    # Delete experiment
    del_resp = await client.delete(f"/api/v1/experiments/{exp_id}")
    assert del_resp.status_code == 200
