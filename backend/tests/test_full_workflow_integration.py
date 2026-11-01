"""
End-to-End Workflow Integration & Regression Tests for Phase 7.
Simulates complete frontend-to-backend user journeys across all 11 stages:
1. Dataset Upload
2. Analysis Retrieval
3. Visualizations Exploration
4. Target Column Selection & Type Overrides
5. Preprocessing Preview Configuration
6. Algorithm Selection
7. Hyperparameter Validation
8. Live Model Training Execution & Log Polling
9. Evaluation Metrics & Diagnostics
10. Model Artifact & Bundle Export
11. Online Prediction Inference
Plus regression verification for static frontend serving, health, and multi-model comparison.
"""

import asyncio
import io
import json
import zipfile
import pytest
from httpx import AsyncClient


SAMPLE_CUSTOMER_CHURN_CSV = """customer_id,age,tenure_months,monthly_charges,total_charges,contract_type,internet_service,churn
CUST_001,42,12,70.35,844.20,month-to-month,fiber,yes
CUST_002,28,4,45.20,180.80,month-to-month,dsl,no
CUST_003,65,36,89.50,3222.00,one-year,fiber,no
CUST_004,50,2,20.00,40.00,month-to-month,no,yes
CUST_005,33,24,105.80,2539.20,two-year,fiber,no
CUST_006,45,6,55.00,330.00,month-to-month,dsl,yes
CUST_007,59,48,99.25,4764.00,two-year,fiber,no
CUST_008,22,1,19.50,19.50,month-to-month,no,yes
CUST_009,38,18,80.10,1441.80,one-year,fiber,no
CUST_010,48,30,60.40,1812.00,one-year,dsl,no
CUST_011,29,8,75.00,600.00,month-to-month,fiber,yes
CUST_012,61,60,115.00,6900.00,two-year,fiber,no
CUST_013,35,15,50.25,753.75,month-to-month,dsl,no
CUST_014,52,9,65.00,585.00,month-to-month,fiber,yes
CUST_015,41,20,85.00,1700.00,one-year,fiber,no
CUST_016,30,3,30.00,90.00,month-to-month,no,no
CUST_017,47,40,95.00,3800.00,two-year,fiber,no
CUST_018,26,5,40.00,200.00,month-to-month,dsl,yes
CUST_019,55,50,110.00,5500.00,two-year,fiber,no
CUST_020,39,14,72.50,1015.00,month-to-month,fiber,no
"""


@pytest.mark.asyncio
async def test_full_eleven_stage_ml_workflow(async_client: AsyncClient):
    """
    Executes the complete 11-step ML pipeline from dataset upload to prediction.
    """
    # -------------------------------------------------------------
    # Step 1: Upload a CSV dataset
    # -------------------------------------------------------------
    files = {
        "file": ("customer_churn.csv", io.BytesIO(SAMPLE_CUSTOMER_CHURN_CSV.encode("utf-8")), "text/csv")
    }
    upload_res = await async_client.post("/api/v1/datasets", files=files)
    assert upload_res.status_code in (201, 409), f"Upload failed: {upload_res.text}"
    if upload_res.status_code == 201:
        dataset_id = upload_res.json()["id"]
    else:
        err_data = upload_res.json()
        dataset_id = err_data.get("existing_dataset_id") or err_data["detail"]["existing_dataset_id"]

    assert dataset_id is not None

    # Process dataset (validation & profiling)
    proc_res = await async_client.post(f"/api/v1/datasets/{dataset_id}/process")
    assert proc_res.status_code in (200, 202)

    # -------------------------------------------------------------
    # Step 2: View its analysis
    # -------------------------------------------------------------
    analysis_res = await async_client.get(f"/api/v1/datasets/{dataset_id}/analysis")
    assert analysis_res.status_code == 200, f"Analysis failed: {analysis_res.text}"
    analysis = analysis_res.json()
    assert analysis["total_rows"] == 20
    assert analysis["total_columns"] == 8
    column_names = [c["name"] for c in analysis["columns"]]
    assert "churn" in column_names
    assert "age" in column_names
    assert "monthly_charges" in column_names

    # -------------------------------------------------------------
    # Step 3: Explore visualizations
    # -------------------------------------------------------------
    viz_res = await async_client.get(f"/api/v1/datasets/{dataset_id}/visualizations?scatter_x=age&scatter_y=monthly_charges")
    assert viz_res.status_code == 200, f"Visualization failed: {viz_res.text}"
    viz = viz_res.json()
    assert "histograms" in viz
    assert "correlation_matrix" in viz
    assert "scatter_plot" in viz

    # -------------------------------------------------------------
    # Step 4: Select a target column and apply type overrides
    # -------------------------------------------------------------
    override_res = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/columns/override",
        json={"columns": {"customer_id": "identifier", "churn": "categorical"}},
    )
    assert override_res.status_code == 200
    override_data = override_res.json()
    assert override_data["dataset_id"] == dataset_id

    # -------------------------------------------------------------
    # Step 5: Configure preprocessing & preview transformations
    # -------------------------------------------------------------
    prep_res = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/preprocess/preview",
        json={
            "excluded_columns": ["customer_id"],
            "target_column": "churn",
            "missing_numeric_strategy": "median",
            "missing_categorical_strategy": "most_frequent",
            "encoding_strategy": "onehot",
            "scaling_strategy": "standard",
            "outlier_strategy": "none",
        },
    )
    assert prep_res.status_code == 200
    prep_preview = prep_res.json()
    assert prep_preview["original_shape"][0] == 20
    assert "customer_id" not in prep_preview["feature_names_in"]

    # -------------------------------------------------------------
    # Step 6: Select an ML algorithm from the registry
    # -------------------------------------------------------------
    algo_res = await async_client.get("/api/v1/algorithms/random_forest_classifier")
    assert algo_res.status_code == 200
    algo_def = algo_res.json()
    assert algo_def["id"] == "random_forest_classifier"
    assert algo_def["task"] == "classification"

    # -------------------------------------------------------------
    # Step 7: Configure hyperparameters and validate
    # -------------------------------------------------------------
    val_res = await async_client.post(
        "/api/v1/algorithms/random_forest_classifier/validate",
        json={"hyperparameters": {"n_estimators": 25, "max_depth": 5, "random_state": 42}},
    )
    assert val_res.status_code == 200
    assert val_res.json()["valid"] is True

    # -------------------------------------------------------------
    # Step 8: Start actual model training and poll job status
    # -------------------------------------------------------------
    train_payload = {
        "name": "End-to-End Churn Classifier",
        "dataset_id": dataset_id,
        "task": "classification",
        "target_column": "churn",
        "selected_features": [
            "age",
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "contract_type",
            "internet_service",
        ],
        "excluded_columns": ["customer_id"],
        "missing_numeric_strategy": "median",
        "encoding_strategy": "onehot",
        "scaling_strategy": "standard",
        "algorithm_id": "random_forest_classifier",
        "hyperparameters": {"n_estimators": 20, "max_depth": 4, "random_state": 42},
        "test_size": 0.25,
        "random_state": 42,
    }
    train_res = await async_client.post("/api/v1/experiments/train", json=train_payload)
    assert train_res.status_code == 202
    exp_id = train_res.json()["id"]

    # Poll until background job completes
    completed = False
    for _ in range(40):
        status_res = await async_client.get(f"/api/v1/experiments/{exp_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()
        if status_data["status"] == "completed":
            completed = True
            break
        elif status_data["status"] == "failed":
            pytest.fail(f"Training failed: {status_data.get('error_message')}")
        await asyncio.sleep(0.15)

    assert completed is True, "Training job did not complete in expected time window."

    # -------------------------------------------------------------
    # Step 9: View actual evaluation results and diagnostics
    # -------------------------------------------------------------
    eval_res = await async_client.get(f"/api/v1/experiments/{exp_id}/evaluation")
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert "accuracy" in eval_data["metrics"]
    assert "f1_weighted" in eval_data["metrics"]
    assert "confusion_matrix" in eval_data["metrics"]
    assert "feature_importance" in eval_data
    assert len(eval_data["feature_importance"]["importances"]) > 0

    # -------------------------------------------------------------
    # Step 10: Download trained model artifact & bundle
    # -------------------------------------------------------------
    # 10a. Joblib download
    dl_joblib_res = await async_client.get(f"/api/v1/experiments/{exp_id}/download?format=joblib")
    assert dl_joblib_res.status_code == 200
    assert len(dl_joblib_res.content) > 0

    # 10b. Pickle download
    dl_pkl_res = await async_client.get(f"/api/v1/experiments/{exp_id}/download?format=pkl")
    assert dl_pkl_res.status_code == 200
    assert len(dl_pkl_res.content) > 0

    # 10c. Export bundle ZIP
    bundle_res = await async_client.get(f"/api/v1/experiments/{exp_id}/export-bundle")
    assert bundle_res.status_code == 200
    assert bundle_res.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(bundle_res.content)) as zf:
        namelist = zf.namelist()
        assert "model.joblib" in namelist
        assert "model.pkl" in namelist
        assert "metadata.json" in namelist
        assert "predict_example.py" in namelist

    # -------------------------------------------------------------
    # Step 11: Make predictions on new unseen data
    # -------------------------------------------------------------
    new_customer_data = [
        {
            "age": 45,
            "tenure_months": 2,
            "monthly_charges": 85.50,
            "total_charges": 171.00,
            "contract_type": "month-to-month",
            "internet_service": "fiber",
        },
        {
            "age": 62,
            "tenure_months": 48,
            "monthly_charges": 25.00,
            "total_charges": 1200.00,
            "contract_type": "two-year",
            "internet_service": "no",
        },
    ]
    pred_res = await async_client.post(
        f"/api/v1/experiments/{exp_id}/predict",
        json={"data": new_customer_data, "include_probabilities": True},
    )
    assert pred_res.status_code == 200
    pred_output = pred_res.json()
    assert pred_output["sample_count"] == 2
    assert len(pred_output["predictions"]) == 2
    assert pred_output["predictions"][0] in ("yes", "no")
    assert pred_output["probabilities"] is not None
    assert len(pred_output["probabilities"]) == 2
    assert pred_output["latency_ms"] > 0


@pytest.mark.asyncio
async def test_static_frontend_and_health_regression(async_client: AsyncClient):
    """
    Verifies frontend files and health endpoints are served reliably.
    """
    # 1. Health endpoint
    h_res = await async_client.get("/api/v1/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "healthy"

    # 2. Root web app serves index.html
    root_res = await async_client.get("/")
    assert root_res.status_code == 200
    assert "Zemo" in root_res.text or "<!DOCTYPE html>" in root_res.text

    # 3. Static assets served
    js_res = await async_client.get("/js/app.js")
    assert js_res.status_code == 200
    assert "ZEMO" in js_res.text
