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
    from core.config import get_settings
    from core.exceptions import AppValidationError as ValidationError
    from services.ml.pipeline_builder import DynamicPipelineBuilder, TrainingConfig
    from services.ml.evaluator import ModelEvaluator
except ImportError:
    from backend.core.config import get_settings
    from backend.core.exceptions import AppValidationError as ValidationError
    from backend.services.ml.pipeline_builder import DynamicPipelineBuilder, TrainingConfig
    from backend.services.ml.evaluator import ModelEvaluator

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
        if artifacts_dir:
            self.artifacts_dir = Path(artifacts_dir).resolve()
        else:
            try:
                self.artifacts_dir = Path(get_settings().experiments_dir).resolve()
            except Exception:
                self.artifacts_dir = (Path(__file__).resolve().parent.parent.parent / "data" / "experiments").resolve()
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
        artifact_path = str((exp_dir / "model.joblib").resolve())

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
                metrics = ModelEvaluator.evaluate_classification(
                    build_res.pipeline, X_test, y_test, target_classes=target_classes
                )
                log(
                    "info",
                    f"Test Results -> Accuracy: {metrics['accuracy']:.4f}, Weighted F1: {metrics['f1_weighted']:.4f}, Macro F1: {metrics['f1_macro']:.4f}."
                )
                if metrics.get("roc_auc") is not None:
                    log("info", f"ROC-AUC (Weighted/OVR): {metrics['roc_auc']:.4f}")
                elif metrics.get("roc_auc_note"):
                    log("info", f"ROC-AUC Note: {metrics['roc_auc_note']}")

                # Feature Importance
                feat_imp = ModelEvaluator.extract_feature_importance(
                    build_res.pipeline, X_train, y_train, build_res.feature_columns
                )
                metrics["feature_importance"] = feat_imp
                if feat_imp.get("available"):
                    top_feats = [f"{item['feature']} ({item['importance']})" for item in feat_imp['importances'][:3]]
                    log("info", f"Top feature importances ({feat_imp['method']}): {', '.join(top_feats)}")

                # Learning curve
                lc = ModelEvaluator.compute_learning_curve(build_res.pipeline, X_train, y_train, "classification")
                if lc:
                    metrics["learning_curve"] = lc
                    log("info", f"Computed learning curve across {len(lc['curve_points'])} training increments.")

            elif config.task == "regression":
                metrics = ModelEvaluator.evaluate_regression(
                    build_res.pipeline, X_test, y_test, X_train=X_train, y_train=y_train
                )
                log(
                    "info",
                    f"Test Results -> RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}, Explained Var: {metrics['explained_variance']:.4f}."
                )
                if metrics.get("dummy_baseline") and metrics["dummy_baseline"].get("rmse") is not None:
                    db = metrics["dummy_baseline"]
                    log("info", f"DummyRegressor Baseline -> RMSE: {db['rmse']:.4f}, MAE: {db['mae']:.4f}, Improvement over baseline: {db['improvement_pct']:.2f}%.")
                if metrics.get("overfitting") and metrics["overfitting"].get("train_r2") is not None:
                    of = metrics["overfitting"]
                    log("info", f"Train vs Test R² -> Train: {of['train_r2']:.4f}, Test: {of['test_r2']:.4f}, Gap: {of['gap']:.4f} (Overfitting: {of['is_overfitting']}).")
                if metrics.get("data_leakage") and metrics["data_leakage"].get("warning"):
                    log("warning", f"Data Leakage Alert: {metrics['data_leakage']['warning']}")

                # Feature Importance
                feat_imp = ModelEvaluator.extract_feature_importance(
                    build_res.pipeline, X_train, y_train, build_res.feature_columns
                )
                metrics["feature_importance"] = feat_imp
                if feat_imp.get("available"):
                    top_feats = [f"{item['feature']} ({item['importance']})" for item in feat_imp['importances'][:3]]
                    log("info", f"Top feature importances ({feat_imp['method']}): {', '.join(top_feats)}")

                # Learning curve
                lc = ModelEvaluator.compute_learning_curve(build_res.pipeline, X_train, y_train, "regression")
                if lc:
                    metrics["learning_curve"] = lc
                    log("info", f"Computed learning curve across {len(lc['curve_points'])} training increments.")

            # Cross Validation for supervised tasks (default 5 folds if not explicitly configured)
            cv_folds = config.cv_folds if config.cv_folds is not None else 5
            if cv_folds and cv_folds >= 2 and len(X_train) >= cv_folds * 2:
                log("info", f"Running {cv_folds}-fold cross validation on training data...")
                cv_metric = "f1_weighted" if config.task == "classification" else "neg_root_mean_squared_error"
                cv_scores = cross_val_score(build_res.pipeline, X_train, y_train, cv=cv_folds, scoring=cv_metric)
                if config.task == "regression":
                    cv_scores = -cv_scores
                metrics["cv_scores"] = [round(float(s), 4) for s in cv_scores]
                metrics["cv_mean"] = round(float(np.mean(cv_scores)), 4)
                metrics["cv_std"] = round(float(np.std(cv_scores)), 4)
                metrics["cv_mean_score"] = metrics["cv_mean"]  # convenient alias
                log("info", f"Cross Validation ({cv_folds}-fold) -> Mean: {metrics['cv_mean']}, Std: {metrics['cv_std']}.")

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
                metrics = ModelEvaluator.evaluate_clustering(build_res.pipeline, X, labels)
                log("info", f"Identified {metrics['num_clusters']} clusters across {train_samples} instances.")
                if metrics.get("silhouette_score") is not None:
                    log("info", f"Clustering Metrics -> Silhouette: {metrics['silhouette_score']}, Davies-Bouldin: {metrics['davies_bouldin']}, Calinski-Harabasz: {metrics.get('calinski_harabasz')}")
                elif metrics.get("evaluation_notes"):
                    log("info", f"Evaluation Note: {metrics['evaluation_notes']}")

                metrics["feature_importance"] = {
                    "available": False,
                    "method": "none",
                    "importances": [],
                    "explanation": "Unsupervised clustering groups geometric density without target labels; feature importance is not defined.",
                }
            else:
                metrics = {
                    "num_clusters": 0,
                    "noise_points": 0,
                    "cluster_distribution": {},
                    "evaluation_notes": "Estimator did not assign cluster labels.",
                }

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
