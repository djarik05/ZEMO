"""
Zemo Model Evaluation and Comparison Engine.

Calculates real model evaluation metrics, feature importance, prediction residuals,
ROC-AUC, learning curves, and multi-model experiment comparison.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation engine for Classification, Regression, and Clustering pipelines.
    Guarantees all metrics are calculated from real predictions and fitted estimators.
    """

    @classmethod
    def evaluate_classification(
        cls,
        pipeline: Pipeline,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        target_classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate classification pipeline on unseen test data.
        Calculates Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report,
        and ROC-AUC where probability or decision scores are supported.
        """
        y_true = np.asarray(y_test)
        y_pred = pipeline.predict(X_test)
        y_pred = np.asarray(y_pred)

        # Unique classes
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        if target_classes is None or len(target_classes) != len(unique_classes):
            class_labels = [str(c) for c in unique_classes]
        else:
            class_labels = [str(c) for c in target_classes]

        # Basic core metrics
        acc = float(accuracy_score(y_true, y_pred))
        prec_w = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        prec_m = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_w = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_m = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Confusion matrix
        conf_mat = confusion_matrix(y_true, y_pred).tolist()

        # Classification report
        raw_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # ROC-AUC calculation
        roc_auc: Optional[float] = None
        roc_auc_note: Optional[str] = None

        try:
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)
                if len(unique_classes) == 2:
                    # Binary classification
                    # Take probability for the positive class
                    prob_col = 1 if y_prob.shape[1] > 1 else 0
                    roc_auc = float(roc_auc_score(y_true, y_prob[:, prob_col]))
                elif len(unique_classes) > 2:
                    # Multiclass classification
                    roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
            elif hasattr(pipeline, "decision_function"):
                y_scores = pipeline.decision_function(X_test)
                if len(unique_classes) == 2:
                    roc_auc = float(roc_auc_score(y_true, y_scores))
                elif len(unique_classes) > 2:
                    roc_auc = float(roc_auc_score(y_true, y_scores, multi_class="ovr", average="weighted"))
            else:
                roc_auc_note = "Estimator does not provide probability estimates or decision scores for ROC-AUC."
        except Exception as e:
            roc_auc_note = f"ROC-AUC could not be computed: {e}"

        # Predictions breakdown / sample points (up to 50 for interactive UI inspection)
        sample_size = min(50, len(y_true))
        prediction_samples = []
        for a, p in zip(y_true[:sample_size], y_pred[:sample_size]):
            prediction_samples.append({
                "actual": str(a),
                "predicted": str(p),
                "correct": bool(a == p),
            })

        actual_counts = {str(k): int(v) for k, v in pd.Series(y_true).value_counts().items()}
        predicted_counts = {str(k): int(v) for k, v in pd.Series(y_pred).value_counts().items()}

        return {
            "accuracy": round(acc, 4),
            "precision_weighted": round(prec_w, 4),
            "precision_macro": round(prec_m, 4),
            "precision": round(prec_w, 4),  # backwards compatibility
            "recall_weighted": round(rec_w, 4),
            "recall_macro": round(rec_m, 4),
            "recall": round(rec_w, 4),  # backwards compatibility
            "f1_weighted": round(f1_w, 4),
            "f1_macro": round(f1_m, 4),
            "f1": round(f1_w, 4),  # backwards compatibility
            "confusion_matrix": conf_mat,
            "classes": class_labels,
            "classification_report": raw_report,
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "roc_auc_note": roc_auc_note,
            "actual_distribution": actual_counts,
            "predicted_distribution": predicted_counts,
            "prediction_samples": prediction_samples,
        }

    @classmethod
    def evaluate_regression(
        cls,
        pipeline: Pipeline,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Evaluate regression pipeline on unseen test data.
        Calculates MAE, MSE, RMSE, R², Explained Variance, and residual diagnostics.
        """
        y_true = np.asarray(y_test, dtype=float)
        y_pred = np.asarray(pipeline.predict(X_test), dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        exp_var = float(explained_variance_score(y_true, y_pred))

        residuals = y_true - y_pred
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))

        # Prediction points sample (up to 100 points for frontend actual vs predicted scatter plot)
        sample_size = min(100, len(y_true))
        prediction_points = []
        for a, p, r in zip(y_true[:sample_size], y_pred[:sample_size], residuals[:sample_size]):
            prediction_points.append({
                "actual": round(float(a), 4),
                "predicted": round(float(p), 4),
                "residual": round(float(r), 4),
            })

        # Residual histogram distribution (10 bins)
        hist_counts, bin_edges = np.histogram(residuals, bins=10)
        residual_hist = {
            "counts": [int(c) for c in hist_counts],
            "bin_edges": [round(float(b), 4) for b in bin_edges],
        }

        return {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "explained_variance": round(exp_var, 4),
            "mean_residual": round(mean_res, 4),
            "std_residual": round(std_res, 4),
            "prediction_points": prediction_points,
            "residual_distribution": residual_hist,
        }

    @classmethod
    def evaluate_clustering(
        cls,
        pipeline: Pipeline,
        X: Union[pd.DataFrame, np.ndarray],
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate clustering pipeline without ground truth labels.
        Calculates Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score,
        and cluster distributions. Explains if metrics cannot be computed.
        """
        labels_arr = np.asarray(labels)
        unique_labels = [int(x) for x in np.unique(labels_arr)]
        non_noise = [l for l in unique_labels if l != -1]
        num_clusters = len(non_noise)
        noise_points = int((labels_arr == -1).sum()) if -1 in unique_labels else 0

        cluster_counts = {str(k): int(v) for k, v in pd.Series(labels_arr).value_counts().items()}

        silhouette: Optional[float] = None
        davies_bouldin: Optional[float] = None
        calinski_harabasz: Optional[float] = None
        eval_notes: Optional[str] = None

        # Transform features to numeric array for cluster metric geometry
        try:
            preprocessor = pipeline.named_steps.get("preprocessor")
            if preprocessor and hasattr(preprocessor, "transform"):
                X_trans = preprocessor.transform(X)
            else:
                X_trans = X

            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            elif isinstance(X_trans, pd.DataFrame):
                X_trans = X_trans.to_numpy()

            # Metric eligibility: at least 2 non-noise clusters and >= 3 valid samples
            valid_mask = labels_arr != -1 if -1 in unique_labels else np.ones(len(labels_arr), dtype=bool)
            valid_count = int(valid_mask.sum())
            valid_clusters = len(np.unique(labels_arr[valid_mask]))

            if valid_clusters >= 2 and valid_count > valid_clusters:
                X_valid = X_trans[valid_mask]
                labels_valid = labels_arr[valid_mask]
                silhouette = float(silhouette_score(X_valid, labels_valid))
                davies_bouldin = float(davies_bouldin_score(X_valid, labels_valid))
                calinski_harabasz = float(calinski_harabasz_score(X_valid, labels_valid))
            else:
                eval_notes = (
                    f"Clustering generated {valid_clusters} non-noise cluster(s) with {valid_count} sample(s). "
                    "Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics require at least 2 distinct clusters."
                )
        except Exception as e:
            eval_notes = f"Could not calculate geometric clustering scores: {e}"

        return {
            "num_clusters": num_clusters,
            "noise_points": noise_points,
            "total_samples": len(labels_arr),
            "cluster_distribution": cluster_counts,
            "silhouette_score": round(silhouette, 4) if silhouette is not None else None,
            "silhouette": round(silhouette, 4) if silhouette is not None else None,  # backwards compatibility
            "davies_bouldin": round(davies_bouldin, 4) if davies_bouldin is not None else None,
            "calinski_harabasz": round(calinski_harabasz, 4) if calinski_harabasz is not None else None,
            "evaluation_notes": eval_notes,
        }

    @classmethod
    def extract_feature_importance(
        cls,
        pipeline: Pipeline,
        X_sample: Union[pd.DataFrame, np.ndarray],
        y_sample: Optional[Union[pd.Series, np.ndarray]] = None,
        raw_feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract or compute feature importances from a fitted pipeline.
        Supports tree-based feature_importances_, linear model coef_, and permutation importance.
        """
        model = pipeline.named_steps.get("model")
        preprocessor = pipeline.named_steps.get("preprocessor")

        if model is None:
            return {
                "available": False,
                "method": "none",
                "importances": [],
                "explanation": "Pipeline does not contain a named estimator step.",
            }

        # Determine feature names after ColumnTransformer
        feature_names: List[str] = []
        if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
            try:
                raw_names = preprocessor.get_feature_names_out()
                feature_names = [
                    str(n).replace("numeric__", "").replace("categorical__", "").replace("remainder__", "")
                    for n in raw_names
                ]
            except Exception:
                feature_names = []

        if not feature_names and raw_feature_names:
            feature_names = list(raw_feature_names)
        elif not feature_names and isinstance(X_sample, pd.DataFrame):
            feature_names = list(X_sample.columns)

        # 1. Native Tree Feature Importances
        if hasattr(model, "feature_importances_"):
            raw_imp = model.feature_importances_
            if len(feature_names) == len(raw_imp):
                pairs = sorted(
                    [{"feature": name, "importance": round(float(imp), 4)} for name, imp in zip(feature_names, raw_imp)],
                    key=lambda x: x["importance"],
                    reverse=True,
                )
                return {
                    "available": True,
                    "method": "tree_feature_importances",
                    "importances": pairs,
                    "explanation": "Calculated directly from tree node purity splits (Gini / MSE reduction).",
                }

        # 2. Linear Model Coefficients
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            if coef.ndim > 1:
                coef_mag = np.mean(np.abs(coef), axis=0)
            else:
                coef_mag = np.abs(coef)

            # Normalize to 0-1
            total = float(np.sum(coef_mag))
            norm_imp = coef_mag / total if total > 0 else coef_mag

            if len(feature_names) == len(norm_imp):
                pairs = sorted(
                    [{"feature": name, "importance": round(float(imp), 4)} for name, imp in zip(feature_names, norm_imp)],
                    key=lambda x: x["importance"],
                    reverse=True,
                )
                return {
                    "available": True,
                    "method": "linear_coefficients",
                    "importances": pairs,
                    "explanation": "Derived from absolute standardized linear model regression / decision boundary coefficients.",
                }

        # 3. Permutation Feature Importance Fallback for estimators without native weights (KNN, Naive Bayes, etc.)
        if y_sample is not None and len(X_sample) >= 10:
            try:
                # Use a lightweight sample for responsiveness
                sample_n = min(150, len(X_sample))
                if isinstance(X_sample, pd.DataFrame):
                    X_sub = X_sample.iloc[:sample_n]
                else:
                    X_sub = X_sample[:sample_n]

                if isinstance(y_sample, (pd.Series, pd.DataFrame)):
                    y_sub = y_sample.iloc[:sample_n]
                else:
                    y_sub = y_sample[:sample_n]

                perm = permutation_importance(pipeline, X_sub, y_sub, n_repeats=3, random_state=42)
                perm_imp = np.maximum(0, perm.importances_mean)
                total = float(np.sum(perm_imp))
                norm_perm = perm_imp / total if total > 0 else perm_imp

                cols = list(X_sample.columns) if isinstance(X_sample, pd.DataFrame) else feature_names[:len(norm_perm)]
                if len(cols) == len(norm_perm):
                    pairs = sorted(
                        [{"feature": name, "importance": round(float(imp), 4)} for name, imp in zip(cols, norm_perm)],
                        key=lambda x: x["importance"],
                        reverse=True,
                    )
                    return {
                        "available": True,
                        "method": "permutation_importance",
                        "importances": pairs,
                        "explanation": (
                            "This algorithm does not produce native split weights or linear coefficients. "
                            "Feature importance was computed via permutation shuffling on validation data."
                        ),
                    }
            except Exception as e:
                logger.warning("Permutation importance calculation failed: %s", e)

        return {
            "available": False,
            "method": "none",
            "importances": [],
            "explanation": (
                "Feature importance is not natively supported by this algorithm and could not be estimated via permutation."
            ),
        }

    @classmethod
    def compute_learning_curve(
        cls,
        pipeline: Pipeline,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        task: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute empirical learning curve progression across training set sizes.
        Requires at least 20 training samples.
        """
        if len(X_train) < 20 or task not in ("classification", "regression"):
            return None

        scoring = "f1_weighted" if task == "classification" else "r2"

        try:
            train_sizes, train_scores, val_scores = learning_curve(
                pipeline,
                X_train,
                y_train,
                train_sizes=np.linspace(0.25, 1.0, 4),
                cv=3,
                scoring=scoring,
                random_state=42,
            )

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            curve_points = []
            for sz, tm, ts, vm, vs in zip(train_sizes, train_mean, train_std, val_mean, val_std):
                curve_points.append({
                    "train_size": int(sz),
                    "train_score_mean": round(float(tm), 4),
                    "train_score_std": round(float(ts), 4),
                    "val_score_mean": round(float(vm), 4),
                    "val_score_std": round(float(vs), 4),
                })

            return {
                "scoring_metric": scoring,
                "curve_points": curve_points,
            }
        except Exception as e:
            logger.warning("Learning curve calculation failed: %s", e)
            return None


class ExperimentComparator:
    """
    Compares multiple trained experiments evaluating compatibility, metrics,
    hyperparameters, preprocessing architectures, and runtime performance.
    """

    @classmethod
    def compare_experiments(cls, experiments: List[Any]) -> Dict[str, Any]:
        """
        Validate compatibility and construct a unified comparison matrix.
        """
        if not experiments:
            raise ValueError("At least two experiments are required for comparison.")

        if len(experiments) < 2:
            raise ValueError("Select at least two experiments to perform a comparative benchmark.")

        # Compatibility check: all experiments must share the same dataset and task
        first_exp = experiments[0]
        ref_dataset_id = first_exp.dataset_id
        ref_task = first_exp.task

        for exp in experiments[1:]:
            if exp.dataset_id != ref_dataset_id:
                raise ValueError(
                    f"Incompatible datasets: experiment '{exp.name}' was trained on dataset '{exp.dataset_id}', "
                    f"whereas '{first_exp.name}' was trained on '{ref_dataset_id}'. Experiments must evaluate equivalent datasets."
                )
            if exp.task != ref_task:
                raise ValueError(
                    f"Incompatible tasks: experiment '{exp.name}' is task '{exp.task}', "
                    f"whereas '{first_exp.name}' is task '{ref_task}'. Cannot compare disparate ML task types."
                )

        exp_summaries = []
        for exp in experiments:
            cfg = exp.config or {}
            metrics = exp.metrics or {}
            summary = {
                "id": exp.id,
                "name": exp.name,
                "algorithm_id": exp.algorithm_id,
                "status": exp.status,
                "duration_seconds": exp.duration_seconds,
                "model_size_bytes": exp.model_size_bytes,
                "created_at": exp.created_at.isoformat() if hasattr(exp.created_at, "isoformat") else str(exp.created_at),
                "preprocessing": {
                    "missing_numeric_strategy": cfg.get("missing_numeric_strategy", "mean"),
                    "encoding_strategy": cfg.get("encoding_strategy", "auto"),
                    "scaling_strategy": cfg.get("scaling_strategy", "standard"),
                    "test_size": cfg.get("test_size", 0.2),
                    "features_count": len(cfg.get("selected_features") or []),
                },
                "hyperparameters": exp.hyperparameters or {},
                "metrics": metrics,
            }
            exp_summaries.append(summary)

        # Determine best performing model according to task primary metric
        primary_metric = "f1_weighted" if ref_task == "classification" else ("r2" if ref_task == "regression" else "silhouette_score")
        higher_is_better = True

        best_exp = None
        best_metric_val = -float("inf") if higher_is_better else float("inf")

        for s in exp_summaries:
            val = s["metrics"].get(primary_metric)
            if val is None:
                # Fallback metric
                val = s["metrics"].get("accuracy" if ref_task == "classification" else ("rmse" if ref_task == "regression" else "silhouette"))

            if val is not None:
                if higher_is_better and val > best_metric_val:
                    best_metric_val = val
                    best_exp = s
                elif not higher_is_better and val < best_metric_val:
                    best_metric_val = val
                    best_exp = s

        # Metric comparison grid
        all_metric_keys = set()
        for s in exp_summaries:
            for k, v in s["metrics"].items():
                if isinstance(v, (int, float)):
                    all_metric_keys.add(k)

        metrics_comparison = {}
        for m_key in sorted(all_metric_keys):
            metrics_comparison[m_key] = {s["id"]: s["metrics"].get(m_key) for s in exp_summaries}

        recommendation = ""
        if best_exp:
            recommendation = (
                f"Top model '{best_exp['name']}' ({best_exp['algorithm_id']}) achieved the highest {primary_metric} "
                f"({best_metric_val}) with a training execution duration of {best_exp['duration_seconds'] or 0:.2f}s."
            )
        else:
            recommendation = "Experiments compared successfully."

        return {
            "dataset_id": ref_dataset_id,
            "task": ref_task,
            "compatible": True,
            "primary_metric": primary_metric,
            "experiments_count": len(exp_summaries),
            "experiments": exp_summaries,
            "best_model": best_exp,
            "metrics_comparison": metrics_comparison,
            "recommendation": recommendation,
        }
