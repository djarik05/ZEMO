"""
Machine Learning Training Engine for Zemo.

Executes end-to-end model training for:
- Supervised Classification (with stratification, cross-validation, and metrics)
- Supervised Regression (with RMSE, MAE, R2, and cross-validation)
- Unsupervised Clustering (with Silhouette score and cluster statistics)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from core.exceptions import AppValidationError as ValidationError
    from services.ml.pipeline_builder import DynamicPipelineBuilder, TrainingConfig
except ImportError:
    from backend.core.exceptions import AppValidationError as ValidationError
    from backend.services.ml.pipeline_builder import DynamicPipelineBuilder, TrainingConfig

logger = logging.getLogger(__name__)


class TrainingResult:
    """Encapsulates the output of a completed model training job."""

    def __init__(
        self,
        experiment_id: str,
        task: str,
        algorithm_id: str,
        display_name: str,
        metrics: Dict[str, Any],
        artifact_path: str,
        model_size_bytes: int,
        feature_columns: List[str],
        numeric_columns: List[str],
        categorical_columns: List[str],
        target_column: Optional[str],
        target_classes: Optional[List[str]],
        train_samples: int,
        test_samples: int,
        duration_seconds: float,
        logs: List[Dict[str, Any]],
        hyperparameters: Dict[str, Any],
    ):
        self.experiment_id = experiment_id
        self.task = task
        self.algorithm_id = algorithm_id
        self.display_name = display_name
        self.metrics = metrics
        self.artifact_path = artifact_path
        self.model_size_bytes = model_size_bytes
        self.feature_columns = feature_columns
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.target_classes = target_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.duration_seconds = duration_seconds
        self.logs = logs
        self.hyperparameters = hyperparameters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "task": self.task,
            "algorithm_id": self.algorithm_id,
            "display_name": self.display_name,
            "metrics": self.metrics,
            "artifact_path": self.artifact_path,
            "model_size_bytes": self.model_size_bytes,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "target_column": self.target_column,
            "target_classes": self.target_classes,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "duration_seconds": self.duration_seconds,
            "logs": self.logs,
            "hyperparameters": self.hyperparameters,
        }


class MLTrainingEngine:
    """
    Executes actual Scikit-learn model training workflows without blocking.
    """

    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        pipeline_builder: Optional[DynamicPipelineBuilder] = None,
    ):
        self.artifacts_dir = Path(artifacts_dir or "data/experiments")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.builder = pipeline_builder or DynamicPipelineBuilder()

    def train(
        self,
        experiment_id: str,
        df: pd.DataFrame,
        config: TrainingConfig,
        log_callback: Optional[Callable[[str, str], None]] = None,
    ) -> TrainingResult:
        """
        Executes the training workflow based on configuration.
        """
        start_time = time.time()
        logs: List[Dict[str, Any]] = []

        def log(level: str, msg: str) -> None:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level.upper(),
                "message": msg,
            }
            logs.append(entry)
            logger.info(f"[{experiment_id}] {level.upper()}: {msg}")
            if log_callback:
                try:
                    log_callback(level, msg)
                except Exception:
                    pass

        log("info", f"Initializing model training pipeline for task: '{config.task}' with algorithm '{config.algorithm_id}'.")
        log("info", f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")

        # Build dynamic pipeline
        build_res = self.builder.build_pipeline(df, config)
        algo_info = build_res.algorithm_info
        estimator_name = build_res.pipeline.named_steps["model"].__class__.__name__
        log("info", f"Constructed Scikit-learn Pipeline with estimator: {estimator_name} ({algo_info.display_name}).")
        log("info", f"Selected features ({len(build_res.feature_columns)}): {', '.join(build_res.feature_columns[:5])}{'...' if len(build_res.feature_columns) > 5 else ''}.")
        log("info", f"Numeric features: {len(build_res.numeric_columns)}, Categorical features: {len(build_res.categorical_columns)}.")

        # Experiment artifact directory
        exp_dir = self.artifacts_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = str(exp_dir / "model.joblib")

        target_classes: Optional[List[str]] = None
        metrics: Dict[str, Any] = {}
        train_samples = 0
        test_samples = 0

        if config.task in ("classification", "regression"):
            target_col = build_res.target_column
            assert target_col is not None

            # Prepare X and y
            valid_mask = df[target_col].notna()
            dropped_rows = len(df) - int(valid_mask.sum())
            if dropped_rows > 0:
                log("warning", f"Dropped {dropped_rows} rows with missing target values.")

            df_clean = df[valid_mask].copy()
            if len(df_clean) < 5:
                raise ValidationError(f"Dataset has too few rows ({len(df_clean)}) after dropping missing targets.")

            X = df_clean[build_res.feature_columns]
            y = df_clean[target_col]

            stratify_target = None
            if config.task == "classification":
                # Check for label encoding if target is object or string
                if not pd.api.types.is_numeric_dtype(y) or str(y.dtype) in ("object", "category", "string"):
                    label_enc = LabelEncoder()
                    y_encoded = label_enc.fit_transform(y.astype(str))
                    target_classes = [str(c) for c in label_enc.classes_]
                    log("info", f"Applied target label encoding. Identified classes: {target_classes}")
                    y = pd.Series(y_encoded, index=y.index)
                else:
                    target_classes = [str(c) for c in np.unique(y.dropna())]

                # Stratify check
                counts = y.value_counts()
                if (counts >= 2).all() and len(counts) > 1 and config.stratify:
                    stratify_target = y
                    log("info", "Using stratified train-test split preserving class distribution.")
                else:
                    log("info", "Class count insufficient for stratification; using random split.")

            # Train / test split
            log("info", f"Splitting data: {(1 - config.test_size) * 100:.0f}% train / {config.test_size * 100:.0f}% test (random_state={config.random_state}).")
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=stratify_target,
            )

            train_samples = len(X_train)
            test_samples = len(X_test)
            log("info", f"Training samples: {train_samples}, Testing samples: {test_samples}.")

            # Fit pipeline
            log("info", f"Fitting pipeline on {train_samples} training samples...")
            fit_start = time.time()
            build_res.pipeline.fit(X_train, y_train)
            fit_duration = time.time() - fit_start
            log("info", f"Pipeline successfully fitted in {fit_duration:.2f} seconds.")

            # Predict on test set
            log("info", "Evaluating model performance on unseen test split...")
            y_pred = build_res.pipeline.predict(X_test)

            if config.task == "classification":
                acc = float(accuracy_score(y_test, y_pred))
                prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
                f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                conf_mat = confusion_matrix(y_test, y_pred).tolist()
                clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

                metrics = {
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1_w, 4),
                    "macro_f1": round(f1_m, 4),
                    "confusion_matrix": conf_mat,
                    "classification_report": clf_report,
                }
                log("info", f"Test Results -> Accuracy: {acc:.4f}, Weighted F1: {f1_w:.4f}, Macro F1: {f1_m:.4f}.")

            elif config.task == "regression":
                mse = float(mean_squared_error(y_test, y_pred))
                rmse = float(np.sqrt(mse))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))

                metrics = {
                    "rmse": round(rmse, 4),
                    "mae": round(mae, 4),
                    "mse": round(mse, 4),
                    "r2": round(r2, 4),
                }
                log("info", f"Test Results -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}.")

            # Optional Cross Validation
            if config.cv_folds and config.cv_folds >= 2:
                log("info", f"Running {config.cv_folds}-fold cross validation on training data...")
                cv_metric = "f1_weighted" if config.task == "classification" else "neg_root_mean_squared_error"
                cv_scores = cross_val_score(build_res.pipeline, X_train, y_train, cv=config.cv_folds, scoring=cv_metric)
                if config.task == "regression":
                    cv_scores = -cv_scores
                metrics["cv_scores"] = [round(float(s), 4) for s in cv_scores]
                metrics["cv_mean"] = round(float(np.mean(cv_scores)), 4)
                metrics["cv_std"] = round(float(np.std(cv_scores)), 4)
                log("info", f"Cross Validation ({config.cv_folds}-fold) -> Mean: {metrics['cv_mean']}, Std: {metrics['cv_std']}.")

        elif config.task == "clustering":
            # Unsupervised Workflow
            X = df[build_res.feature_columns].copy()
            train_samples = len(X)
            test_samples = 0
            log("info", f"Starting unsupervised clustering on {train_samples} samples.")

            fit_start = time.time()
            build_res.pipeline.fit(X)
            fit_duration = time.time() - fit_start
            log("info", f"Clustering pipeline fitted in {fit_duration:.2f} seconds.")

            model = build_res.pipeline.named_steps["model"]
            labels: Optional[np.ndarray] = getattr(model, "labels_", None)

            if labels is None and hasattr(build_res.pipeline, "predict"):
                labels = build_res.pipeline.predict(X)

            if labels is not None:
                labels_series = pd.Series(labels)
                unique_labels = [int(x) for x in np.unique(labels)]
                non_noise_labels = [l for l in unique_labels if l != -1]
                num_clusters = len(non_noise_labels)
                noise_points = int((labels == -1).sum()) if -1 in unique_labels else 0

                cluster_dist = {str(k): int(v) for k, v in labels_series.value_counts().items()}

                metrics = {
                    "num_clusters": num_clusters,
                    "noise_points": noise_points,
                    "cluster_distribution": cluster_dist,
                }

                # Calculate Silhouette Score and Davies-Bouldin if valid cluster count
                if 1 < num_clusters < len(X):
                    try:
                        # Extract transformed numerical features for silhouette evaluation
                        preprocessor = build_res.pipeline.named_steps["preprocessor"]
                        X_transformed = preprocessor.transform(X)
                        if hasattr(X_transformed, "toarray"):
                            X_transformed = X_transformed.toarray()

                        valid_mask = labels != -1
                        if valid_mask.sum() > 2 and len(np.unique(labels[valid_mask])) > 1:
                            sil = float(silhouette_score(X_transformed[valid_mask], labels[valid_mask]))
                            db = float(davies_bouldin_score(X_transformed[valid_mask], labels[valid_mask]))
                            metrics["silhouette"] = round(sil, 4)
                            metrics["davies_bouldin"] = round(db, 4)
                            log("info", f"Clustering metrics -> Silhouette Score: {sil:.4f}, Davies-Bouldin: {db:.4f}.")
                    except Exception as e:
                        log("warning", f"Could not calculate silhouette score: {e}")

                log("info", f"Identified {num_clusters} clusters across {train_samples} instances.")

        # Save fitted pipeline artifact
        log("info", f"Serializing fitted model pipeline to {artifact_path}...")
        joblib.dump(build_res.pipeline, artifact_path)
        model_size_bytes = os.path.getsize(artifact_path)
        log("info", f"Serialized model size: {model_size_bytes / 1024:.2f} KB.")

        total_duration = time.time() - start_time
        log("info", f"Training completed successfully in {total_duration:.2f} seconds.")

        return TrainingResult(
            experiment_id=experiment_id,
            task=config.task,
            algorithm_id=config.algorithm_id,
            display_name=algo_info.display_name,
            metrics=metrics,
            artifact_path=artifact_path,
            model_size_bytes=model_size_bytes,
            feature_columns=build_res.feature_columns,
            numeric_columns=build_res.numeric_columns,
            categorical_columns=build_res.categorical_columns,
            target_column=build_res.target_column,
            target_classes=target_classes,
            train_samples=train_samples,
            test_samples=test_samples,
            duration_seconds=round(total_duration, 3),
            logs=logs,
            hyperparameters=build_res.sanitized_hyperparameters,
        )
