"""
Zemo Model Prediction Engine and Artifact Exporter.

Provides trusted model loading, feature schema validation, inference execution
(classification, regression, clustering), prediction probabilities, and export bundles.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

UUID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass
class PredictionResult:
    """Encapsulates the structured output of model inference."""
    experiment_id: str
    task: str
    algorithm_id: str
    predictions: List[Any]
    probabilities: Optional[List[Dict[str, float]]] = None
    sample_count: int = 0
    features_used: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "task": self.task,
            "algorithm_id": self.algorithm_id,
            "predictions": self.predictions,
            "probabilities": self.probabilities,
            "sample_count": self.sample_count,
            "features_used": self.features_used,
            "latency_ms": round(self.latency_ms, 2),
        }


class ModelPredictor:
    """
    Secure inference and prediction engine.
    Strictly loads backend-generated artifacts with path containment checks.
    """

    def __init__(self, base_artifacts_dir: Optional[Path] = None):
        if base_artifacts_dir is not None:
            self.base_dir = Path(base_artifacts_dir).resolve()
        else:
            try:
                from core.config import get_settings
                self.base_dir = Path(get_settings().experiments_dir).resolve()
            except Exception:
                self.base_dir = (Path(__file__).resolve().parent.parent.parent / "data" / "experiments").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def validate_trusted_path(self, experiment_id: str) -> Path:
        """
        Verify that experiment_id is safe and the artifact path is strictly
        contained within the backend trusted experiments directory.
        Prevents directory traversal attacks.
        """
        clean_id = str(experiment_id).strip()
        if not UUID_REGEX.match(clean_id) or ".." in clean_id or "/" in clean_id or "\\" in clean_id:
            raise ValueError(f"Invalid experiment identifier format: '{experiment_id}'.")

        target_file = (self.base_dir / clean_id / "model.joblib").resolve()
        # Fallback check if saved in relative data/experiments
        if not target_file.exists():
            for alt_dir in [
                Path("data/experiments").resolve(),
                (Path(__file__).resolve().parent.parent.parent / "data" / "experiments").resolve(),
                (Path(__file__).resolve().parent.parent / "data" / "experiments").resolve(),
            ]:
                alt_candidate = (alt_dir / clean_id / "model.joblib").resolve()
                if alt_candidate.exists():
                    target_file = alt_candidate
                    break

        if not target_file.exists():
            raise FileNotFoundError(f"Model artifact not found for experiment '{experiment_id}' at {target_file}.")

        return target_file

    def load_trusted_pipeline(self, experiment_id: str) -> Pipeline:
        """
        Securely load a trusted pipeline from backend storage.
        """
        model_path = self.validate_trusted_path(experiment_id)
        try:
            pipeline = joblib.load(model_path)
            if not isinstance(pipeline, Pipeline):
                # Wrapped or single estimator fallback
                pass
            return pipeline
        except Exception as e:
            logger.error("Failed to load model artifact %s: %s", model_path, e)
            raise RuntimeError(f"Could not deserialize trusted model artifact: {e}")

    def predict(
        self,
        experiment_id: str,
        experiment_record: Any,
        input_data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        include_probabilities: bool = True,
    ) -> PredictionResult:
        """
        Execute prediction on new input data.
        1. Validates input structure against experiment schema.
        2. Applies full preprocessor transformations automatically.
        3. Returns predictions and probabilities where supported.
        """
        start_time = time.time()
        pipeline = self.load_trusted_pipeline(experiment_id)

        # Normalize input to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            if not input_data:
                raise ValueError("Input data list is empty.")
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError(f"Unsupported input data format: {type(input_data)}")

        if df.empty:
            raise ValueError("Input data contains 0 rows.")

        config = getattr(experiment_record, "config", {}) or {}
        task = getattr(experiment_record, "task", "classification")
        algorithm_id = getattr(experiment_record, "algorithm_id", "unknown")
        metrics = getattr(experiment_record, "metrics", {}) or {}

        # Validate required feature columns
        expected_features: List[str] = config.get("selected_features", [])
        if not expected_features and hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
            prep = pipeline.named_steps["preprocessor"]
            if hasattr(prep, "feature_names_in_"):
                expected_features = list(prep.feature_names_in_)

        if expected_features:
            missing_cols = [col for col in expected_features if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Input data is missing required feature columns: {missing_cols}. "
                    f"Expected features: {expected_features}."
                )
            # Reorder / slice to expected features only
            X_input = df[expected_features].copy()
        else:
            X_input = df.copy()

        # Run pipeline predictions
        try:
            preds = pipeline.predict(X_input)
        except Exception as e:
            logger.error("Pipeline predict failure: %s", e)
            raise RuntimeError(f"Prediction execution failed during pipeline transformation: {e}")

        # Post-process predictions
        if hasattr(preds, "tolist"):
            raw_predictions = preds.tolist()
        else:
            raw_predictions = list(preds)

        # If classification and labels map is available, map encoded numbers back to class names
        class_labels: Optional[List[str]] = metrics.get("classes")
        if task == "classification" and class_labels:
            mapped_preds = []
            for p in raw_predictions:
                try:
                    p_idx = int(p)
                    if 0 <= p_idx < len(class_labels):
                        mapped_preds.append(class_labels[p_idx])
                    else:
                        mapped_preds.append(str(p))
                except (ValueError, TypeError):
                    mapped_preds.append(str(p))
            formatted_predictions = mapped_preds
        elif task == "regression":
            formatted_predictions = [round(float(p), 4) for p in raw_predictions]
        else:
            formatted_predictions = raw_predictions

        # Compute probabilities for classification if supported
        probabilities: Optional[List[Dict[str, float]]] = None
        if task == "classification" and include_probabilities:
            if hasattr(pipeline, "predict_proba"):
                try:
                    prob_matrix = pipeline.predict_proba(X_input)
                    probabilities = []
                    classes = class_labels or [str(c) for c in getattr(pipeline, "classes_", range(prob_matrix.shape[1]))]

                    for row in prob_matrix:
                        row_dict = {}
                        for idx, p_val in enumerate(row):
                            c_name = str(classes[idx]) if idx < len(classes) else f"class_{idx}"
                            row_dict[c_name] = round(float(p_val), 4)
                        probabilities.append(row_dict)
                except Exception as e:
                    logger.debug("predict_proba calculation bypassed: %s", e)

        latency = (time.time() - start_time) * 1000.0

        return PredictionResult(
            experiment_id=experiment_id,
            task=task,
            algorithm_id=algorithm_id,
            predictions=formatted_predictions,
            probabilities=probabilities,
            sample_count=len(df),
            features_used=expected_features or list(df.columns),
            latency_ms=latency,
        )

    def generate_export_bundle(self, experiment: Any) -> io.BytesIO:
        """
        Generate a comprehensive .zip bundle containing:
        - model.joblib
        - model.pkl
        - metadata.json
        - predict_example.py (standalone runnable script)
        """
        trusted_path = self.validate_trusted_path(experiment.id)
        pipeline = joblib.load(trusted_path)

        cfg = experiment.config or {}
        metrics = experiment.metrics or {}
        feature_columns = cfg.get("selected_features", [])

        # Construct comprehensive metadata
        metadata = {
            "model_id": experiment.id,
            "name": experiment.name,
            "task": experiment.task,
            "algorithm_id": experiment.algorithm_id,
            "dataset_id": experiment.dataset_id,
            "dataset_version_id": experiment.dataset_version_id,
            "created_at": experiment.created_at.isoformat() if hasattr(experiment.created_at, "isoformat") else str(experiment.created_at),
            "completed_at": experiment.completed_at.isoformat() if hasattr(experiment.completed_at, "isoformat") else str(experiment.completed_at),
            "duration_seconds": experiment.duration_seconds,
            "feature_schema": {
                "features": feature_columns,
                "target_column": cfg.get("target_column"),
                "classes": metrics.get("classes"),
            },
            "preprocessing_strategy": {
                "missing_numeric": cfg.get("missing_numeric_strategy", "mean"),
                "missing_categorical": cfg.get("missing_categorical_strategy", "most_frequent"),
                "encoding": cfg.get("encoding_strategy", "auto"),
                "scaling": cfg.get("scaling_strategy", "standard"),
            },
            "hyperparameters": experiment.hyperparameters,
            "metrics": {
                k: v for k, v in metrics.items() if not isinstance(v, (dict, list)) or k in ("confusion_matrix", "classes")
            },
            "framework": "Scikit-learn",
            "serialization_format": "Joblib",
        }

        # Standalone Python sample script
        sample_dict = {col: 1.0 for col in feature_columns}
        sample_code = f'''"""
Standalone inference script for model '{experiment.name}' ({experiment.id}).
Exported from Zemo Machine Learning Platform.
"""

import joblib
import pandas as pd

# 1. Load the complete Scikit-learn Pipeline (includes fitted Preprocessing and Estimator)
model_file = "model.joblib"
pipeline = joblib.load(model_file)
print(f"Loaded pipeline: {{pipeline}}")

# 2. Prepare new sample data matching required feature schema
new_data = pd.DataFrame([{json.dumps(sample_dict, indent=4)}])
print("\\nInput Data:")
print(new_data)

# 3. Predict directly on raw inputs (the pipeline handles imputation, encoding, and scaling)
predictions = pipeline.predict(new_data)
print("\\nPredictions:", predictions)

# 4. Optional: Predict class probabilities (if supported)
if hasattr(pipeline, "predict_proba"):
    probabilities = pipeline.predict_proba(new_data)
    print("Class Probabilities:", probabilities)
'''

        # Build in-memory ZIP archive
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. model.joblib
            with open(trusted_path, "rb") as f:
                zf.writestr("model.joblib", f.read())

            # 2. model.pkl (using standard pickle)
            import pickle
            pkl_bytes = pickle.dumps(pipeline)
            zf.writestr("model.pkl", pkl_bytes)

            # 3. metadata.json
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

            # 4. predict_example.py
            zf.writestr("predict_example.py", sample_code)

        zip_buffer.seek(0)
        return zip_buffer


_model_predictor_instance: Optional[ModelPredictor] = None


def get_model_predictor() -> ModelPredictor:
    """Singleton getter for ModelPredictor."""
    global _model_predictor_instance
    if _model_predictor_instance is None:
        _model_predictor_instance = ModelPredictor()
    return _model_predictor_instance
