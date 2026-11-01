"""
FastAPI Router for ML Experiments and Model Training.
"""

from __future__ import annotations

import json
import os
import pickle
import uuid
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.config import get_db_session
from core.database.models.dataset import Dataset
from core.database.models.experiment import Experiment, ExperimentStatus
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from services.ml.job_manager import TrainingJobManager
from services.ml.loader import DatasetLoader, get_dataset_loader
from services.ml.pipeline_builder import TrainingConfig
from services.ml.evaluator import ExperimentComparator
from services.ml.predictor import ModelPredictor, get_model_predictor
from api.v1.experiments.schemas import (
    CancelExperimentResponse,
    CompareExperimentsRequest,
    CompareExperimentsResponse,
    ExperimentListResponse,
    ExperimentResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
    TrainExperimentRequest,
)

router = APIRouter(prefix="", tags=["experiments"])


def _build_experiment_response(exp: Experiment, active_job: Optional[Any] = None) -> ExperimentResponse:
    d = exp.to_dict()
    # Overlay in-memory active job status & logs if available and fresher
    if active_job:
        d["status"] = active_job.status
        if active_job.logs:
            d["logs"] = active_job.logs
        if active_job.started_at:
            d["started_at"] = active_job.started_at.isoformat()
        if active_job.completed_at:
            d["completed_at"] = active_job.completed_at.isoformat()
        if active_job.error_message:
            d["error_message"] = active_job.error_message
        if active_job.result:
            d["metrics"] = active_job.result.metrics
            d["artifact_path"] = active_job.result.artifact_path
            d["model_size_bytes"] = active_job.result.model_size_bytes
            d["duration_seconds"] = active_job.result.duration_seconds
            d["hyperparameters"] = active_job.result.hyperparameters

    return ExperimentResponse(**d)


@router.post(
    "/train",
    response_model=ExperimentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Launch ML Model Training Job",
    description="Initiates asynchronous, non-blocking Scikit-learn model training in a background worker.",
)
async def launch_training(
    payload: TrainExperimentRequest,
    session: AsyncSession = Depends(get_db_session),
    loader: DatasetLoader = Depends(get_dataset_loader),
) -> ExperimentResponse:
    # 1. Fetch and load dataset
    try:
        df, version = await loader.load_by_dataset_id(payload.dataset_id)
    except NotFoundError:
        raise
    except Exception as e:
        raise NotFoundError(f"Could not load dataset with ID '{payload.dataset_id}': {e}")

    if df.empty:
        raise ValidationError("Uploaded dataset is empty.")

    version_id = version.id if version else None

    # 2. Create TrainingConfig
    exp_name = payload.name or f"{payload.task.capitalize()} {payload.algorithm_id.replace('_', ' ').title()}"
    config = TrainingConfig(
        dataset_id=payload.dataset_id,
        dataset_version_id=version_id,
        experiment_name=exp_name,
        task=payload.task,
        target_column=payload.target_column,
        selected_features=payload.selected_features,
        excluded_columns=payload.excluded_columns,
        type_overrides=payload.type_overrides,
        missing_numeric_strategy=payload.missing_numeric_strategy,
        missing_numeric_fill_value=payload.missing_numeric_fill_value,
        missing_categorical_strategy=payload.missing_categorical_strategy,
        missing_categorical_fill_value=payload.missing_categorical_fill_value,
        encoding_strategy=payload.encoding_strategy,
        scaling_strategy=payload.scaling_strategy,
        outlier_method=payload.outlier_method,
        outlier_factor=payload.outlier_factor,
        remove_constant_columns=payload.remove_constant_columns,
        algorithm_id=payload.algorithm_id,
        hyperparameters=payload.hyperparameters,
        test_size=payload.test_size,
        random_state=payload.random_state,
        cv_folds=payload.cv_folds,
        stratify=payload.stratify,
    )

    # 4. Create database record
    experiment_id = str(uuid.uuid4())
    experiment = Experiment(
        id=experiment_id,
        name=exp_name,
        dataset_id=payload.dataset_id,
        dataset_version_id=version_id,
        task=payload.task,
        algorithm_id=payload.algorithm_id,
        status=ExperimentStatus.PENDING.value,
        config=config.model_dump(),
        hyperparameters=payload.hyperparameters,
        metrics={},
        logs=[{
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "message": f"Experiment created and queued for background training. Name: '{exp_name}'.",
        }],
        created_at=datetime.utcnow(),
    )
    session.add(experiment)
    await session.commit()
    await session.refresh(experiment)

    # 5. Launch non-blocking background job
    job_mgr = TrainingJobManager.get_instance()
    active_job = await job_mgr.launch_job(experiment_id, df, config)

    return _build_experiment_response(experiment, active_job)


@router.get(
    "",
    response_model=ExperimentListResponse,
    summary="List Experiments",
)
async def list_experiments(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    task: Optional[str] = Query(None, description="Filter by task"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> ExperimentListResponse:
    stmt = select(Experiment).order_by(desc(Experiment.created_at))

    if dataset_id:
        stmt = stmt.where(Experiment.dataset_id == dataset_id)
    if task:
        stmt = stmt.where(Experiment.task == task)
    if status_filter:
        stmt = stmt.where(Experiment.status == status_filter)

    stmt = stmt.offset(offset).limit(limit)
    res = await session.execute(stmt)
    experiments = list(res.scalars().all())

    job_mgr = TrainingJobManager.get_instance()
    items = []
    for exp in experiments:
        active_job = job_mgr.get_job_state(exp.id)
        items.append(_build_experiment_response(exp, active_job))

    return ExperimentListResponse(total=len(items), experiments=items)


@router.post(
    "/compare",
    response_model=CompareExperimentsResponse,
    summary="Compare Multiple Experiments",
    description="Compares multiple compatible experiments evaluated on equivalent datasets and task settings.",
)
async def compare_experiments(
    payload: CompareExperimentsRequest,
    session: AsyncSession = Depends(get_db_session),
) -> CompareExperimentsResponse:
    if len(payload.experiment_ids) < 2:
        raise ValidationError("At least two experiment IDs are required for comparison.")

    stmt = select(Experiment).where(Experiment.id.in_(payload.experiment_ids))
    res = await session.execute(stmt)
    experiments = list(res.scalars().all())

    found_ids = {e.id for e in experiments}
    missing_ids = set(payload.experiment_ids) - found_ids
    if missing_ids:
        raise NotFoundError(f"Experiments not found: {', '.join(missing_ids)}")

    # Ensure all experiments have finished training
    not_completed = [e.name for e in experiments if e.status != ExperimentStatus.COMPLETED.value]
    if not_completed:
        raise ValidationError(
            f"Cannot compare experiments that have not completed training: {', '.join(not_completed)}"
        )

    try:
        comparison_dict = ExperimentComparator.compare_experiments(experiments)
    except ValueError as e:
        raise ValidationError(str(e))

    return CompareExperimentsResponse(**comparison_dict)


@router.get(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Get Experiment Details & Status",
)
async def get_experiment(
    experiment_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ExperimentResponse:
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    res = await session.execute(stmt)
    experiment = res.scalar_one_or_none()

    if not experiment:
        raise NotFoundError(f"Experiment '{experiment_id}' not found.")

    job_mgr = TrainingJobManager.get_instance()
    active_job = job_mgr.get_job_state(experiment_id)

    return _build_experiment_response(experiment, active_job)


@router.get(
    "/{experiment_id}/evaluation",
    summary="Get Detailed Model Evaluation & Diagnostics",
    description="Returns comprehensive model metrics, feature importance, prediction samples, residuals, and learning curves.",
)
async def get_experiment_evaluation(
    experiment_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    res = await session.execute(stmt)
    experiment = res.scalar_one_or_none()

    if not experiment:
        raise NotFoundError(f"Experiment '{experiment_id}' not found.")

    if experiment.status != ExperimentStatus.COMPLETED.value:
        raise ValidationError(f"Experiment evaluation is only available for completed runs (current status: '{experiment.status}').")

    metrics = experiment.metrics or {}
    return {
        "experiment_id": experiment.id,
        "name": experiment.name,
        "task": experiment.task,
        "algorithm_id": experiment.algorithm_id,
        "duration_seconds": experiment.duration_seconds,
        "model_size_bytes": experiment.model_size_bytes,
        "metrics": metrics,
        "feature_importance": metrics.get("feature_importance", {}),
        "learning_curve": metrics.get("learning_curve"),
        "prediction_points": metrics.get("prediction_points", []),
        "prediction_samples": metrics.get("prediction_samples", []),
        "residual_distribution": metrics.get("residual_distribution"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "classification_report": metrics.get("classification_report"),
        "roc_auc": metrics.get("roc_auc"),
        "roc_auc_note": metrics.get("roc_auc_note"),
        "cluster_distribution": metrics.get("cluster_distribution"),
    }


@router.post(
    "/{experiment_id}/cancel",
    response_model=CancelExperimentResponse,
    summary="Cancel Running Experiment",
)
async def cancel_experiment(
    experiment_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> CancelExperimentResponse:
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    res = await session.execute(stmt)
    experiment = res.scalar_one_or_none()

    if not experiment:
        raise NotFoundError(f"Experiment '{experiment_id}' not found.")

    job_mgr = TrainingJobManager.get_instance()
    cancelled = job_mgr.cancel_job(experiment_id)

    if cancelled:
        experiment.status = ExperimentStatus.CANCELLED.value
        experiment.completed_at = datetime.utcnow()
        await session.commit()
        return CancelExperimentResponse(
            experiment_id=experiment_id,
            status=ExperimentStatus.CANCELLED.value,
            message="Experiment training has been cancelled.",
        )
    else:
        return CancelExperimentResponse(
            experiment_id=experiment_id,
            status=experiment.status,
            message=f"Experiment could not be cancelled because it is in status '{experiment.status}'.",
        )


@router.delete(
    "/{experiment_id}",
    summary="Delete Experiment",
)
async def delete_experiment(
    experiment_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    res = await session.execute(stmt)
    experiment = res.scalar_one_or_none()

    if not experiment:
        raise NotFoundError(f"Experiment '{experiment_id}' not found.")

    # Remove artifact file if exists
    if experiment.artifact_path and os.path.exists(experiment.artifact_path):
        try:
            os.remove(experiment.artifact_path)
            # Remove parent directory if empty
            parent = os.path.dirname(experiment.artifact_path)
            if os.path.exists(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except Exception:
            pass

    await session.delete(experiment)
    await session.commit()

    return {"message": "Experiment deleted successfully", "id": experiment_id}


@router.get(
    "/{experiment_id}/download",
    summary="Download Trained Model Artifact",
    description="Download the fitted Scikit-learn model artifact in .joblib or .pkl format.",
)
async def download_model(
    experiment_id: str,
    format: Literal["joblib", "pkl"] = Query(default="joblib", description="Artifact format: joblib or pkl"),
    session: AsyncSession = Depends(get_db_session),
    predictor: ModelPredictor = Depends(get_model_predictor),
) -> Response:
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    res = await session.execute(stmt)
    experiment = res.scalar_one_or_none()

    if not experiment:
        raise NotFoundError(f"Experiment '{experiment_id}' not found.")

    if not experiment.artifact_path or not os.path.exists(experiment.artifact_path):
        raise NotFoundError("Model artifact file does not exist or training did not complete.")

    try:
        predictor.validate_trusted_path(experiment_id)
    except Exception as e:
        raise NotFoundError(f"Secure artifact validation failed: {e}")

    filename_base = f"zemo_{experiment.task}_{experiment.algorithm_id}_{experiment.id[:8]}"

    if format == "pkl":
        pipeline = predictor.load_trusted_pipeline(experiment_id)
        pkl_bytes = pickle.dumps(pipeline)
        return Response(
            content=pkl_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.pkl"'},
        )

    return FileResponse(
        experiment.artifact_path,
        media_type="application/octet-stream",
        filename=f"{filename_base}.joblib",
    )


@router.get(
    "/{experiment_id}/metadata",
    response_model=ModelMetadataResponse,
    summary="Get Model Metadata & Schema",
    description="Returns model metadata, feature requirements, hyperparameter configuration, evaluation metrics, and sample Python inference code.",
)
async def get_model_metadata(
    experiment_id: str,
    session: AsyncSession = Depends(get_db_session),
    predictor: ModelPredictor = Depends(get_model_predictor),
) -> ModelMetadataResponse:
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    res = await session.execute(stmt)
    experiment = res.scalar_one_or_none()

    if not experiment:
        raise NotFoundError(f"Experiment '{experiment_id}' not found.")

    cfg = experiment.config or {}
    metrics = experiment.metrics or {}
    feature_columns = cfg.get("selected_features", [])

    sample_dict = {col: 1.0 for col in feature_columns}
    sample_code = f'''import joblib
import pandas as pd

# 1. Load pipeline (fitted preprocessor + estimator)
pipeline = joblib.load("model.joblib")

# 2. New data matching required feature schema
input_data = pd.DataFrame([{json.dumps(sample_dict, indent=4)}])

# 3. Predict directly on raw inputs
predictions = pipeline.predict(input_data)
print("Predictions:", predictions)
'''

    return ModelMetadataResponse(
        model_id=experiment.id,
        name=experiment.name,
        task=experiment.task,
        algorithm_id=experiment.algorithm_id,
        dataset_id=experiment.dataset_id,
        dataset_version_id=experiment.dataset_version_id,
        created_at=experiment.created_at.isoformat() if experiment.created_at else None,
        completed_at=experiment.completed_at.isoformat() if experiment.completed_at else None,
        duration_seconds=experiment.duration_seconds,
        feature_schema={
            "features": feature_columns,
            "target_column": cfg.get("target_column"),
            "classes": metrics.get("classes"),
        },
        preprocessing_strategy={
            "missing_numeric": cfg.get("missing_numeric_strategy", "mean"),
            "missing_categorical": cfg.get("missing_categorical_strategy", "most_frequent"),
            "encoding": cfg.get("encoding_strategy", "auto"),
            "scaling": cfg.get("scaling_strategy", "standard"),
        },
        hyperparameters=experiment.hyperparameters or {},
        metrics={k: v for k, v in metrics.items() if not isinstance(v, (dict, list)) or k in ("confusion_matrix", "classes")},
        framework="Scikit-learn",
        serialization_format="Joblib",
        sample_prediction_code=sample_code,
    )


@router.get(
    "/{experiment_id}/export-bundle",
    summary="Download Model Export Bundle (.zip)",
    description="Downloads a self-contained ZIP archive containing model.joblib, model.pkl, metadata.json, and standalone predict_example.py.",
)
async def export_model_bundle(
    experiment_id: str,
    session: AsyncSession = Depends(get_db_session),
    predictor: ModelPredictor = Depends(get_model_predictor),
) -> StreamingResponse:
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    res = await session.execute(stmt)
    experiment = res.scalar_one_or_none()

    if not experiment:
        raise NotFoundError(f"Experiment '{experiment_id}' not found.")

    if not experiment.artifact_path or not os.path.exists(experiment.artifact_path):
        raise NotFoundError("Model artifact file does not exist or training did not complete.")

    try:
        zip_buffer = predictor.generate_export_bundle(experiment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate export bundle: {e}")

    filename = f"zemo_export_{experiment.task}_{experiment.algorithm_id}_{experiment.id[:8]}.zip"
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/{experiment_id}/predict",
    response_model=PredictionResponse,
    summary="Run Online Model Prediction",
    description="Executes online prediction using the fitted trusted pipeline on single or batch raw input records.",
)
async def predict_with_model(
    experiment_id: str,
    payload: PredictionRequest,
    session: AsyncSession = Depends(get_db_session),
    predictor: ModelPredictor = Depends(get_model_predictor),
) -> PredictionResponse:
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    res = await session.execute(stmt)
    experiment = res.scalar_one_or_none()

    if not experiment:
        raise NotFoundError(f"Experiment '{experiment_id}' not found.")

    if not experiment.artifact_path or not os.path.exists(experiment.artifact_path):
        raise NotFoundError("Model artifact file does not exist or training did not complete.")

    try:
        pred_res = predictor.predict(
            experiment_id=experiment_id,
            experiment_record=experiment,
            input_data=payload.data,
            include_probabilities=payload.include_probabilities,
        )
    except (ValueError, TypeError) as e:
        raise ValidationError(str(e))
    except (FileNotFoundError, PermissionError) as e:
        raise NotFoundError(str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        experiment_id=pred_res.experiment_id,
        task=pred_res.task,
        algorithm_id=pred_res.algorithm_id,
        predictions=pred_res.predictions,
        probabilities=pred_res.probabilities,
        sample_count=pred_res.sample_count,
        features_used=pred_res.features_used,
        latency_ms=pred_res.latency_ms,
        timestamp=datetime.utcnow().isoformat(),
    )

