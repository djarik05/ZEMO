"""
Tests for Phase 5 — Model Evaluation and Comparison Engine.
"""

import io
import pytest
import numpy as np
import pandas as pd
from httpx import ASGITransport, AsyncClient
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from main import app
from services.ml.evaluator import ModelEvaluator, ExperimentComparator
from services.ml.pipeline_builder import DynamicPipelineBuilder, TrainingConfig
from services.ml.trainer import MLTrainingEngine
from core.database.models.experiment import Experiment, ExperimentStatus


@pytest.fixture
def sample_classification_data():
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n) * 2.0
    cat1 = np.random.choice(["A", "B", "C"], size=n)
    # Target correlated with x1
    target = (x1 > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat1": cat1, "target": target})
    return df


@pytest.fixture
def sample_regression_data():
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n) * 1.5
    cat1 = np.random.choice(["Low", "Med", "High"], size=n)
    target = 2.5 * x1 - 1.2 * x2 + np.random.randn(n) * 0.1
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat1": cat1, "target": target})
    return df


def test_classification_comprehensive_evaluation(sample_classification_data):
    df = sample_classification_data
    cfg = TrainingConfig(
        dataset_id="test_ds",
        task="classification",
        target_column="target",
        algorithm_id="random_forest",
        hyperparameters={"n_estimators": 10, "max_depth": 3},
        test_size=0.25,
        random_state=42,
    )
    builder = DynamicPipelineBuilder()
    build_res = builder.build_pipeline(df, cfg)

    X_train = df[["x1", "x2", "cat1"]].iloc[:75]
    y_train = df["target"].iloc[:75]
    X_test = df[["x1", "x2", "cat1"]].iloc[75:]
    y_test = df["target"].iloc[75:]

    build_res.pipeline.fit(X_train, y_train)

    metrics = ModelEvaluator.evaluate_classification(
        build_res.pipeline, X_test, y_test, target_classes=["0", "1"]
    )

    # Core metrics
    assert "accuracy" in metrics and 0.0 <= metrics["accuracy"] <= 1.0
    assert "precision_weighted" in metrics and 0.0 <= metrics["precision_weighted"] <= 1.0
    assert "precision_macro" in metrics and 0.0 <= metrics["precision_macro"] <= 1.0
    assert "recall_weighted" in metrics and 0.0 <= metrics["recall_weighted"] <= 1.0
    assert "recall_macro" in metrics and 0.0 <= metrics["recall_macro"] <= 1.0
    assert "f1_weighted" in metrics and 0.0 <= metrics["f1_weighted"] <= 1.0
    assert "f1_macro" in metrics and 0.0 <= metrics["f1_macro"] <= 1.0

    # Confusion matrix and classification report
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 2
    assert "classification_report" in metrics
    assert "0" in metrics["classification_report"]
    assert "1" in metrics["classification_report"]

    # ROC-AUC
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    # Prediction samples
    assert "prediction_samples" in metrics
    assert len(metrics["prediction_samples"]) == 25
    assert "actual" in metrics["prediction_samples"][0]
    assert "predicted" in metrics["prediction_samples"][0]

    # Feature importance
    feat_imp = ModelEvaluator.extract_feature_importance(
        build_res.pipeline, X_train, y_train, build_res.feature_columns
    )
    assert feat_imp["available"] is True
    assert feat_imp["method"] == "tree_feature_importances"
    assert len(feat_imp["importances"]) > 0
    # x1 was used to generate target, so x1 should have positive importance
    feats = [item["feature"] for item in feat_imp["importances"]]
    assert any("x1" in f for f in feats)


def test_regression_comprehensive_evaluation(sample_regression_data):
    df = sample_regression_data
    cfg = TrainingConfig(
        dataset_id="test_ds",
        task="regression",
        target_column="target",
        algorithm_id="random_forest_regressor",
        hyperparameters={"n_estimators": 10, "max_depth": 3},
        test_size=0.25,
        random_state=42,
    )
    builder = DynamicPipelineBuilder()
    build_res = builder.build_pipeline(df, cfg)

    X_train = df[["x1", "x2", "cat1"]].iloc[:75]
    y_train = df["target"].iloc[:75]
    X_test = df[["x1", "x2", "cat1"]].iloc[75:]
    y_test = df["target"].iloc[75:]

    build_res.pipeline.fit(X_train, y_train)

    metrics = ModelEvaluator.evaluate_regression(build_res.pipeline, X_test, y_test)

    # Core regression metrics
    assert "mae" in metrics and metrics["mae"] >= 0.0
    assert "mse" in metrics and metrics["mse"] >= 0.0
    assert "rmse" in metrics and metrics["rmse"] >= 0.0
    assert "r2" in metrics
    assert "explained_variance" in metrics

    # Residuals & Prediction points
    assert "mean_residual" in metrics
    assert "std_residual" in metrics
    assert "prediction_points" in metrics
    assert len(metrics["prediction_points"]) == 25
    assert "actual" in metrics["prediction_points"][0]
    assert "predicted" in metrics["prediction_points"][0]
    assert "residual" in metrics["prediction_points"][0]

    assert "residual_distribution" in metrics
    assert len(metrics["residual_distribution"]["counts"]) == 10

    # Learning curve
    lc = ModelEvaluator.compute_learning_curve(build_res.pipeline, X_train, y_train, "regression")
    assert lc is not None
    assert "curve_points" in lc
    assert len(lc["curve_points"]) > 0


def test_clustering_comprehensive_evaluation():
    np.random.seed(42)
    X = pd.DataFrame({
        "a": np.concatenate([np.random.randn(40), np.random.randn(40) + 5]),
        "b": np.concatenate([np.random.randn(40), np.random.randn(40) - 5]),
    })
    cfg = TrainingConfig(
        dataset_id="test_cluster",
        task="clustering",
        algorithm_id="kmeans",
        hyperparameters={"n_clusters": 2},
    )
    builder = DynamicPipelineBuilder()
    build_res = builder.build_pipeline(X, cfg)
    build_res.pipeline.fit(X)

    labels = build_res.pipeline.named_steps["model"].labels_
    metrics = ModelEvaluator.evaluate_clustering(build_res.pipeline, X, labels)

    assert metrics["num_clusters"] == 2
    assert "cluster_distribution" in metrics
    assert metrics["silhouette_score"] is not None
    assert metrics["davies_bouldin"] is not None
    assert metrics["calinski_harabasz"] is not None
    assert metrics["silhouette_score"] > 0.4  # distinct well-separated clusters


def test_clustering_single_cluster_explanation():
    # When all points get assigned to a single cluster
    pipeline = Pipeline([("scale", StandardScaler())])
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    labels = np.array([0, 0, 0])

    metrics = ModelEvaluator.evaluate_clustering(pipeline, X, labels)
    assert metrics["num_clusters"] == 1
    assert metrics["silhouette_score"] is None
    assert metrics["evaluation_notes"] is not None
    assert "require at least 2 distinct clusters" in metrics["evaluation_notes"]


def test_feature_importance_fallback_for_knn(sample_classification_data):
    df = sample_classification_data
    cfg = TrainingConfig(
        dataset_id="test_ds",
        task="classification",
        target_column="target",
        algorithm_id="knn",
        hyperparameters={"n_neighbors": 3},
    )
    builder = DynamicPipelineBuilder()
    build_res = builder.build_pipeline(df, cfg)

    X_train = df[["x1", "x2", "cat1"]].iloc[:80]
    y_train = df["target"].iloc[:80]
    build_res.pipeline.fit(X_train, y_train)

    feat_imp = ModelEvaluator.extract_feature_importance(
        build_res.pipeline, X_train, y_train, build_res.feature_columns
    )
    assert feat_imp["available"] is True
    assert feat_imp["method"] == "permutation_importance"
    assert "does not produce native split weights" in feat_imp["explanation"]
    assert len(feat_imp["importances"]) > 0


def test_experiment_comparator_logic():
    exp1 = Experiment(
        id="exp-1",
        name="Random Forest Default",
        dataset_id="ds-100",
        task="classification",
        algorithm_id="random_forest",
        status=ExperimentStatus.COMPLETED.value,
        config={"missing_numeric_strategy": "mean", "scaling_strategy": "standard", "test_size": 0.2},
        hyperparameters={"n_estimators": 50},
        metrics={"accuracy": 0.88, "f1_weighted": 0.875, "precision_weighted": 0.89},
        duration_seconds=1.2,
        model_size_bytes=10450,
    )

    exp2 = Experiment(
        id="exp-2",
        name="Logistic Regression",
        dataset_id="ds-100",
        task="classification",
        algorithm_id="logistic_regression",
        status=ExperimentStatus.COMPLETED.value,
        config={"missing_numeric_strategy": "median", "scaling_strategy": "minmax", "test_size": 0.2},
        hyperparameters={"C": 1.0},
        metrics={"accuracy": 0.92, "f1_weighted": 0.918, "precision_weighted": 0.925},
        duration_seconds=0.4,
        model_size_bytes=4200,
    )

    comp = ExperimentComparator.compare_experiments([exp1, exp2])
    assert comp["compatible"] is True
    assert comp["dataset_id"] == "ds-100"
    assert comp["task"] == "classification"
    assert comp["experiments_count"] == 2
    assert comp["best_model"]["id"] == "exp-2"  # Higher f1_weighted
    assert "accuracy" in comp["metrics_comparison"]
    assert comp["metrics_comparison"]["accuracy"]["exp-1"] == 0.88
    assert comp["metrics_comparison"]["accuracy"]["exp-2"] == 0.92


def test_experiment_comparator_incompatible_raises():
    exp1 = Experiment(id="1", name="Exp 1", dataset_id="ds-A", task="classification", status="completed")
    exp2 = Experiment(id="2", name="Exp 2", dataset_id="ds-B", task="classification", status="completed")

    with pytest.raises(ValueError, match="Incompatible datasets"):
        ExperimentComparator.compare_experiments([exp1, exp2])

    exp3 = Experiment(id="3", name="Exp 3", dataset_id="ds-A", task="regression", status="completed")
    with pytest.raises(ValueError, match="Incompatible tasks"):
        ExperimentComparator.compare_experiments([exp1, exp3])


@pytest.mark.asyncio
async def test_api_comparison_and_evaluation_endpoints(sample_classification_data, async_client):
    client = async_client
    # 1. Upload dataset
    csv_buffer = io.BytesIO()
    sample_classification_data.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()
    upload_resp = await client.post(
        "/api/v1/datasets",
        files={"file": ("comp_test.csv", csv_bytes, "text/csv")},
    )
    assert upload_resp.status_code == 201
    dataset_id = upload_resp.json()["id"]

    # 2. Train Model 1 (Random Forest)
    t1_resp = await client.post(
        "/api/v1/experiments/train",
        json={
            "name": "RF Model",
            "dataset_id": dataset_id,
            "task": "classification",
            "target_column": "target",
            "algorithm_id": "random-forest",
            "hyperparameters": {"n_estimators": 10, "max_depth": 2},
            "test_size": 0.25,
            "random_state": 42,
        },
    )
    assert t1_resp.status_code == 202
    exp1_id = t1_resp.json()["id"]

    # 3. Train Model 2 (Logistic Regression)
    t2_resp = await client.post(
        "/api/v1/experiments/train",
        json={
            "name": "LR Model",
            "dataset_id": dataset_id,
            "task": "classification",
            "target_column": "target",
            "algorithm_id": "logistic-regression",
            "hyperparameters": {"C": 0.5},
            "test_size": 0.25,
            "random_state": 42,
        },
    )
    assert t2_resp.status_code == 202
    exp2_id = t2_resp.json()["id"]

    # Wait a moment for background workers to complete
    import asyncio
    for _ in range(30):
        r1 = await client.get(f"/api/v1/experiments/{exp1_id}")
        r2 = await client.get(f"/api/v1/experiments/{exp2_id}")
        if r1.json()["status"] == "completed" and r2.json()["status"] == "completed":
            break
        await asyncio.sleep(0.2)

    # 4. Test detailed evaluation endpoint
    eval_resp = await client.get(f"/api/v1/experiments/{exp1_id}/evaluation")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert eval_data["experiment_id"] == exp1_id
    assert "accuracy" in eval_data["metrics"]
    assert "confusion_matrix" in eval_data["metrics"]
    assert "feature_importance" in eval_data
    assert "prediction_samples" in eval_data

    # 5. Test comparison endpoint
    comp_resp = await client.post(
        "/api/v1/experiments/compare",
        json={"experiment_ids": [exp1_id, exp2_id]},
    )
    assert comp_resp.status_code == 200
    comp_data = comp_resp.json()
    assert comp_data["compatible"] is True
    assert comp_data["experiments_count"] == 2
    assert comp_data["best_model"] is not None
    assert "accuracy" in comp_data["metrics_comparison"]
    assert exp1_id in comp_data["metrics_comparison"]["accuracy"]
    assert exp2_id in comp_data["metrics_comparison"]["accuracy"]
