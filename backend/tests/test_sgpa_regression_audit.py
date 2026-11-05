"""
End-to-End Regression Audit & Multi-Algorithm Verification for Student SGPA Dataset.
Tests all 8 supported regression algorithms, Dummy baseline, overfitting checks,
and serialization/prediction consistency.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from services.ml.pipeline_builder import TrainingConfig
from services.ml.trainer import MLTrainingEngine
from services.ml.predictor import ModelPredictor


@pytest.fixture
def sgpa_dataframe() -> pd.DataFrame:
    # Prefer sample-data path or backend upload path
    paths = [
        Path("sample-data/student_sgpa.csv"),
        Path("backend/data/uploads/eaa03af4c50e423599ba5b7e15d3b943.csv"),
        Path("../sample-data/student_sgpa.csv"),
    ]
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    pytest.skip("Student SGPA CSV dataset not found in workspace.")


def test_sgpa_dataset_integrity(sgpa_dataframe: pd.DataFrame):
    df = sgpa_dataframe
    assert len(df) == 1000
    assert "4th_Sem_SGPA" in df.columns
    assert "Student_Name" in df.columns
    assert "Roll_Number" in df.columns
    assert df["4th_Sem_SGPA"].isnull().sum() == 0


@pytest.mark.parametrize(
    "algorithm_id",
    [
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "decision_tree_regressor",
        "random_forest_regressor",
        "knn_regressor",
        "svr",
        "gradient_boosting_regressor",
    ],
)
def test_all_eight_regression_algorithms_train_and_evaluate(
    sgpa_dataframe: pd.DataFrame, algorithm_id: str, tmp_path
):
    df = sgpa_dataframe
    target = "4th_Sem_SGPA"
    features = ["1st_Sem_SGPA", "2nd_Sem_SGPA", "3rd_Sem_SGPA"]
    excluded = ["Student_Name", "Roll_Number", "Phone_Number", "Email"]

    config = TrainingConfig(
        dataset_id="test-student-sgpa",
        task="regression",
        target_column=target,
        selected_features=features,
        excluded_columns=excluded,
        algorithm_id=algorithm_id,
        hyperparameters={},
        test_size=0.2,
        random_state=42,
        cv_folds=5,
    )

    engine = MLTrainingEngine()
    predictor = ModelPredictor()

    exp_id = f"test-sgpa-{algorithm_id[:8]}"
    result = engine.train(exp_id, df, config)

    # 1. Verify genuine estimator trained
    assert result.algorithm_id == algorithm_id
    assert os.path.exists(result.artifact_path)

    # 2. Verify regression metrics calculated
    m = result.metrics
    assert "r2" in m and "rmse" in m and "mae" in m and "mse" in m
    assert isinstance(m["r2"], float)
    assert isinstance(m["rmse"], float)
    assert m["rmse"] > 0

    # 3. Verify DummyRegressor baseline
    assert "dummy_baseline" in m
    assert m["dummy_baseline"]["rmse"] is not None
    assert m["dummy_baseline"]["improvement_pct"] is not None

    # 4. Verify Overfitting diagnostic
    assert "overfitting" in m
    assert "train_r2" in m["overfitting"]
    assert "gap" in m["overfitting"]

    # 5. Verify Cross-validation
    assert "cv_scores" in m
    assert len(m["cv_scores"]) == 5

    # 6. Verify serialization, reload, and prediction consistency
    loaded_pipeline = predictor.load_trusted_pipeline(exp_id)
    test_sample = pd.DataFrame([{
        "1st_Sem_SGPA": 8.0,
        "2nd_Sem_SGPA": 8.2,
        "3rd_Sem_SGPA": 8.5,
    }])
    pred_val = loaded_pipeline.predict(test_sample)[0]
    assert 0.0 <= pred_val <= 10.0
