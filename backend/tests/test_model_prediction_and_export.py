"""
Tests for Phase 6: Model Export, Prediction Engine, Security, and Standalone Script Acceptance.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import pytest
from httpx import AsyncClient
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.database.models.dataset import Dataset
from core.database.models.experiment import Experiment, ExperimentStatus
from services.ml.predictor import ModelPredictor, get_model_predictor


@pytest.fixture
def test_predictor(tmp_path):
    return ModelPredictor(base_artifacts_dir=tmp_path)


@pytest.mark.asyncio
async def test_trusted_path_security(test_predictor):
    """Ensure path traversal and invalid experiment identifiers are strictly blocked."""
    # 1. Directory traversal attempt
    with pytest.raises(ValueError, match="Invalid experiment identifier"):
        test_predictor.validate_trusted_path("../evil_path")

    with pytest.raises(ValueError, match="Invalid experiment identifier"):
        test_predictor.validate_trusted_path("exp/../../../etc/passwd")

    with pytest.raises(ValueError, match="Invalid experiment identifier"):
        test_predictor.validate_trusted_path("exp\\..\\evil")

    # 2. Non-existent experiment
    with pytest.raises(FileNotFoundError):
        test_predictor.validate_trusted_path("valid-uuid-but-does-not-exist")


@pytest.mark.asyncio
async def test_classification_prediction_and_probabilities(tmp_path):
    """Test trained classification pipeline inference with probability distribution."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    feature_names = list(X.columns)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    pipe.fit(X, y)

    exp_id = "test-iris-exp-123"
    exp_dir = tmp_path / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_path = exp_dir / "model.joblib"
    joblib.dump(pipe, model_path)

    predictor = ModelPredictor(base_artifacts_dir=tmp_path)

    class MockExp:
        id = exp_id
        task = "classification"
        algorithm_id = "random_forest_classifier"
        config = {"selected_features": feature_names}
        metrics = {"classes": ["setosa", "versicolor", "virginica"]}

    # Single row dict
    sample_row = X.iloc[0].to_dict()
    result = predictor.predict(exp_id, MockExp(), sample_row, include_probabilities=True)

    assert result.sample_count == 1
    assert len(result.predictions) == 1
    assert result.predictions[0] == "setosa"
    assert result.probabilities is not None
    assert len(result.probabilities) == 1
    assert "setosa" in result.probabilities[0]
    assert result.probabilities[0]["setosa"] > 0.8
    assert result.latency_ms > 0

    # Batch list of dicts
    batch_rows = X.iloc[:5].to_dict(orient="records")
    batch_result = predictor.predict(exp_id, MockExp(), batch_rows, include_probabilities=True)
    assert batch_result.sample_count == 5
    assert len(batch_result.predictions) == 5

    # Missing column validation
    with pytest.raises(ValueError, match="missing required feature columns"):
        predictor.predict(exp_id, MockExp(), {"sepal length (cm)": 5.1})


@pytest.mark.asyncio
async def test_regression_prediction(tmp_path):
    """Test trained regression pipeline inference."""
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data.iloc[:, :4]
    y = diabetes.target
    feature_names = list(X.columns)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])
    pipe.fit(X, y)

    exp_id = "test-reg-exp-456"
    exp_dir = tmp_path / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_path = exp_dir / "model.joblib"
    joblib.dump(pipe, model_path)

    predictor = ModelPredictor(base_artifacts_dir=tmp_path)

    class MockExp:
        id = exp_id
        task = "regression"
        algorithm_id = "linear_regression"
        config = {"selected_features": feature_names}
        metrics = {"r2_score": 0.45}

    sample_row = X.iloc[0].to_dict()
    result = predictor.predict(exp_id, MockExp(), sample_row, include_probabilities=False)

    assert result.sample_count == 1
    assert len(result.predictions) == 1
    assert isinstance(result.predictions[0], float)
    assert result.probabilities is None


@pytest.mark.asyncio
async def test_export_bundle_generation(tmp_path):
    """Test generation of the .zip export bundle containing joblib, pkl, metadata, and script."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    pipe = Pipeline([("rf", RandomForestClassifier(n_estimators=10, random_state=42))])
    pipe.fit(X, y)

    exp_id = "test-export-789"
    exp_dir = tmp_path / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_path = exp_dir / "model.joblib"
    joblib.dump(pipe, model_path)

    predictor = ModelPredictor(base_artifacts_dir=tmp_path)

    class MockExp:
        id = exp_id
        name = "Iris Random Forest"
        task = "classification"
        algorithm_id = "random_forest_classifier"
        dataset_id = "dataset-123"
        dataset_version_id = "v-1"
        created_at = "2026-11-01T12:00:00"
        completed_at = "2026-11-01T12:01:00"
        duration_seconds = 60.0
        hyperparameters = {"n_estimators": 10}
        config = {
            "selected_features": list(X.columns),
            "target_column": "target",
        }
        metrics = {"accuracy": 0.98, "classes": ["setosa", "versicolor", "virginica"]}

    zip_buffer = predictor.generate_export_bundle(MockExp())
    assert isinstance(zip_buffer, io.BytesIO)

    # Inspect zip contents
    with zipfile.ZipFile(zip_buffer, "r") as zf:
        file_list = zf.namelist()
        assert "model.joblib" in file_list
        assert "model.pkl" in file_list
        assert "metadata.json" in file_list
        assert "predict_example.py" in file_list

        # Verify metadata
        meta = json.loads(zf.read("metadata.json"))
        assert meta["model_id"] == exp_id
        assert meta["feature_schema"]["features"] == list(X.columns)

        # Verify standalone script content
        script = zf.read("predict_example.py").decode("utf-8")
        assert "pipeline = joblib.load" in script
        assert "pipeline.predict" in script


@pytest.mark.asyncio
async def test_api_predict_and_download_flow(async_client: AsyncClient, tmp_path):
    """End-to-end API test: metadata, predict endpoint, joblib download, pkl download, and bundle export."""
    from core.database.config import get_database

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    features = list(X.columns)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    pipe.fit(X, y)

    # Use backend data experiments directory
    real_exp_dir = Path("data/experiments/test-api-pred-exp")
    real_exp_dir.mkdir(parents=True, exist_ok=True)
    real_model_path = real_exp_dir / "model.joblib"
    joblib.dump(pipe, str(real_model_path))

    from core.database.models.dataset import DatasetFormat, DatasetStatus

    dataset = Dataset(
        id="ds-pred-test",
        name="Iris Dataset",
        original_filename="iris.csv",
        format=DatasetFormat.CSV,
        file_size=1000,
        content_hash="hash",
        status=DatasetStatus.UPLOADED,
    )

    exp = Experiment(
        id="test-api-pred-exp",
        dataset_id=dataset.id,
        name="API Prediction Test Experiment",
        task="classification",
        algorithm_id="random_forest_classifier",
        status=ExperimentStatus.COMPLETED.value,
        artifact_path=str(real_model_path),
        hyperparameters={"n_estimators": 10},
        config={
            "selected_features": features,
            "target_column": "target",
            "missing_numeric_strategy": "mean",
        },
        metrics={"accuracy": 1.0, "classes": ["setosa", "versicolor", "virginica"]},
    )

    db = get_database()
    async with db.session() as session:
        session.add(dataset)
        session.add(exp)


    try:
        # 1. Test Metadata endpoint
        meta_res = await async_client.get(f"/api/v1/experiments/{exp.id}/metadata")
        assert meta_res.status_code == 200
        meta_data = meta_res.json()
        assert meta_data["model_id"] == exp.id
        assert meta_data["feature_schema"]["features"] == features
        assert "sample_prediction_code" in meta_data

        # 2. Test Predict endpoint
        pred_payload = {
            "data": [
                {features[0]: 5.1, features[1]: 3.5, features[2]: 1.4, features[3]: 0.2},
                {features[0]: 6.7, features[1]: 3.0, features[2]: 5.2, features[3]: 2.3},
            ],
            "include_probabilities": True,
        }
        pred_res = await async_client.post(f"/api/v1/experiments/{exp.id}/predict", json=pred_payload)
        assert pred_res.status_code == 200
        pred_data = pred_res.json()
        assert pred_data["sample_count"] == 2
        assert len(pred_data["predictions"]) == 2
        assert pred_data["predictions"][0] == "setosa"
        assert pred_data["probabilities"] is not None
        assert len(pred_data["probabilities"]) == 2

        # 3. Test Download joblib
        dl_res = await async_client.get(f"/api/v1/experiments/{exp.id}/download?format=joblib")
        assert dl_res.status_code == 200
        assert len(dl_res.content) > 0

        # 4. Test Download pkl
        dl_pkl_res = await async_client.get(f"/api/v1/experiments/{exp.id}/download?format=pkl")
        assert dl_pkl_res.status_code == 200
        assert len(dl_pkl_res.content) > 0

        # 5. Test Export ZIP bundle
        bundle_res = await async_client.get(f"/api/v1/experiments/{exp.id}/export-bundle")
        assert bundle_res.status_code == 200
        assert bundle_res.headers["content-type"] == "application/zip"

        with zipfile.ZipFile(io.BytesIO(bundle_res.content)) as zf:
            assert "model.joblib" in zf.namelist()
            assert "predict_example.py" in zf.namelist()

    finally:
        # Cleanup artifact
        if real_model_path.exists():
            real_model_path.unlink()
        if real_exp_dir.exists():
            real_exp_dir.rmdir()


def test_acceptance_criteria_standalone_execution():
    """
    CRITICAL ACCEPTANCE CRITERIA:
    Train a model, download it, load it in a separate Python script and successfully
    make predictions using new data.
    """
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    feature_names = list(X.columns)

    # Build fitted pipeline with preprocessor and estimator
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    pipeline.fit(X, y)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        exp_id = "acceptance-exp-001"
        model_dir = temp_path / exp_id
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib_file = model_dir / "model.joblib"
        joblib.dump(pipeline, joblib_file)

        predictor = ModelPredictor(base_artifacts_dir=temp_path)

        class MockExp:
            id = exp_id
            name = "Acceptance Test Model"
            task = "classification"
            algorithm_id = "random_forest_classifier"
            dataset_id = "ds-acc"
            dataset_version_id = "v-1"
            created_at = "2026-11-01T12:00:00"
            completed_at = "2026-11-01T12:01:00"
            duration_seconds = 10.0
            hyperparameters = {"n_estimators": 10}
            config = {"selected_features": feature_names, "target_column": "target"}
            metrics = {"classes": ["setosa", "versicolor", "virginica"]}

        # 1. Generate export bundle
        zip_buffer = predictor.generate_export_bundle(MockExp())

        # 2. Extract into a completely separate external directory (simulating user download)
        extract_dir = temp_path / "user_extracted_model"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            zf.extractall(extract_dir)

        assert (extract_dir / "model.joblib").exists()
        assert (extract_dir / "predict_example.py").exists()

        # 3. Create a separate Python script that loads model.joblib and tests new unseen data
        test_script = extract_dir / "test_run.py"
        test_script.write_text("""
import joblib
import pandas as pd
import sys

# Load complete Scikit-learn pipeline
pipeline = joblib.load('model.joblib')

# New unseen sample data
new_samples = pd.DataFrame([
    {
        'sepal length (cm)': 5.0,
        'sepal width (cm)': 3.4,
        'petal length (cm)': 1.5,
        'petal width (cm)': 0.2
    },
    {
        'sepal length (cm)': 6.5,
        'sepal width (cm)': 3.0,
        'petal length (cm)': 5.2,
        'petal width (cm)': 2.0
    }
])

preds = pipeline.predict(new_samples)
pred_list = [int(p) for p in preds]
print("PREDICTIONS_SUCCESS:", pred_list)
assert len(pred_list) == 2
assert pred_list == [0, 2]
sys.exit(0)
""")

        # 4. Execute the separate Python script via subprocess
        proc = subprocess.run(
            [sys.executable, str(test_script)],
            cwd=str(extract_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert proc.returncode == 0, f"External script failed with error:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        assert "PREDICTIONS_SUCCESS: [0, 2]" in proc.stdout

