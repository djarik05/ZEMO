"""
ML-based Content Classification Subsystem.

Uses TF-IDF vectorization and a scikit-learn classifier (MultinomialNB)
to classify dataset columns and text content for sensitive categories.
Gracefully degrades when no trained model is available.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from app.core.config import get_settings
from app.core.exceptions import DatasetNotFoundError, ModelNotAvailableError
from app.core.logging import get_logger
from app.core.security import generate_storage_id
from app.models.enums import (
    DetectionMethod,
    FindingCategory,
    PolicyDecision,
    RemediationStatus,
    ReviewStatus,
    Severity,
    utcnow,
)
from app.repositories.dataset_repo import DatasetRepository
from app.utils.masking import truncate_evidence

logger = get_logger("ml.classifier")

# Labels recognized as benign / safe
BENIGN_LABELS = frozenset({"clean", "benign", "public", "normal", "safe", "general"})

# Mapping of label to FindingCategory and Severity
LABEL_CATEGORY_MAP: dict[str, tuple[FindingCategory, Severity]] = {
    "email": (FindingCategory.EMAIL, Severity.HIGH),
    "phone": (FindingCategory.PHONE, Severity.HIGH),
    "address": (FindingCategory.ADDRESS, Severity.MEDIUM),
    "credit_card": (FindingCategory.CREDIT_CARD, Severity.CRITICAL),
    "personal_id": (FindingCategory.PERSONAL_ID, Severity.CRITICAL),
    "ssn": (FindingCategory.PERSONAL_ID, Severity.CRITICAL),
    "api_key": (FindingCategory.API_KEY, Severity.CRITICAL),
    "password": (FindingCategory.PASSWORD, Severity.CRITICAL),
    "private_key": (FindingCategory.PRIVATE_KEY, Severity.CRITICAL),
    "db_credential": (FindingCategory.DB_CREDENTIAL, Severity.CRITICAL),
    "prompt_injection": (FindingCategory.PROMPT_INJECTION, Severity.HIGH),
    "suspicious_content": (FindingCategory.SUSPICIOUS_CONTENT, Severity.HIGH),
    "pii": (FindingCategory.PERSONAL_ID, Severity.HIGH),
    "sensitive": (FindingCategory.SUSPICIOUS_CONTENT, Severity.HIGH),
    "confidential": (FindingCategory.SUSPICIOUS_CONTENT, Severity.HIGH),
    "malicious": (FindingCategory.SUSPICIOUS_CONTENT, Severity.HIGH),
}


class ContentClassifier:
    """
    Text content classifier using TF-IDF and Multinomial Naive Bayes.
    
    Trained models are saved to and loaded from storage/ml/classifier.joblib.
    When untrained, degrades gracefully by reporting is_available() == False
    and raising ModelNotAvailableError on prediction attempts.
    """

    def __init__(
        self,
        repo: DatasetRepository | None = None,
        model_path: Path | str | None = None,
    ) -> None:
        self.repo = repo or DatasetRepository()
        self.settings = get_settings()

        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            self.model_path = self.settings.storage_path / "ml" / "classifier.joblib"

        self.pipeline: Pipeline | None = None
        self._available: bool = False
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load a serialized model from model_path."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                self._available = True
                logger.info("Loaded ML classification model from %s", self.model_path)
            except Exception as exc:
                logger.warning("Failed to load model from %s: %s", self.model_path, exc)
                self.pipeline = None
                self._available = False
        else:
            self.pipeline = None
            self._available = False

    def is_available(self) -> bool:
        """Return True if a trained classification model is loaded and ready."""
        return self._available and self.pipeline is not None

    def train(self, texts: list[str], labels: list[str]) -> dict[str, Any]:
        """
        Train a text classification pipeline and persist the model to storage.

        Args:
            texts: List of training text samples.
            labels: Corresponding target class labels.

        Returns:
            Dictionary with training metrics (accuracy, class distribution, feature count).
        """
        if not texts or not labels:
            raise ValueError("Training data and labels cannot be empty.")
        if len(texts) != len(labels):
            raise ValueError(
                f"Mismatch in sample count: {len(texts)} texts and {len(labels)} labels."
            )

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2),
                        token_pattern=r"(?u)\b\w+\b",
                        lowercase=True,
                    ),
                ),
                ("clf", MultinomialNB(alpha=0.1)),
            ]
        )

        pipeline.fit(texts, labels)
        preds = pipeline.predict(texts)
        acc = float(accuracy_score(labels, preds))

        # Class distribution
        class_distribution: dict[str, int] = {}
        for lbl in labels:
            class_distribution[lbl] = class_distribution.get(lbl, 0) + 1

        feature_count = len(pipeline.named_steps["tfidf"].vocabulary_)

        # Persist model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.model_path)

        self.pipeline = pipeline
        self._available = True

        metrics = {
            "accuracy": round(acc, 4),
            "class_distribution": class_distribution,
            "feature_count": feature_count,
            "samples_trained": len(texts),
            "classes": sorted(list(set(labels))),
        }

        logger.info(
            "Trained ML classifier on %d samples (features=%d, acc=%.4f)",
            len(texts),
            feature_count,
            acc,
        )
        return metrics

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Predict class labels and confidence scores for a list of texts.

        Returns:
            List of dicts: [{"text_preview": str, "predicted_label": str, "confidence": float}]
        
        Raises:
            ModelNotAvailableError: If no trained model is available.
        """
        if not self.is_available() or self.pipeline is None:
            raise ModelNotAvailableError(
                "Content classification model is not available. Train the model before making predictions."
            )

        if not texts:
            return []

        clean_texts = [str(t) if t is not None else "" for t in texts]
        preds = self.pipeline.predict(clean_texts)
        probs = self.pipeline.predict_proba(clean_texts)
        classes = list(self.pipeline.classes_)

        results: list[dict[str, Any]] = []
        for i, text in enumerate(texts):
            prob_row = probs[i]
            best_idx = prob_row.argmax()
            confidence = float(prob_row[best_idx])
            pred_label = str(preds[i])

            results.append({
                "text_preview": truncate_evidence(str(text), max_len=60),
                "predicted_label": pred_label,
                "confidence": round(confidence, 4),
            })

        return results

    def classify_dataset_column(
        self, dataset_id: str, column: str
    ) -> list[dict[str, Any]]:
        """
        Load DataFrame from repo, extract values of the specified column,
        run predict() on each value, and return findings for any detected sensitive classifications.

        Raises:
            ModelNotAvailableError: If model is not trained.
            DatasetNotFoundError: If dataset_id does not exist.
        """
        if not self.is_available():
            raise ModelNotAvailableError(
                "Content classification model is not available. Train the model before classifying datasets."
            )

        df = self.repo.get_dataframe(dataset_id)
        if df is None:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' not found.",
                detail={"dataset_id": dataset_id},
            )

        if column not in df.columns:
            logger.warning("Column '%s' not found in dataset '%s'", column, dataset_id)
            return []

        series = df[column]
        findings: list[dict[str, Any]] = []
        now = utcnow()

        # Collect non-null string rows with original indices
        valid_items: list[tuple[int, str]] = []
        for row_idx, val in series.items():
            if val is None or pd.isna(val):
                continue
            val_str = str(val).strip()
            if not val_str:
                continue
            idx = int(row_idx) if isinstance(row_idx, (int, float)) else 0
            valid_items.append((idx, val_str))

        if not valid_items:
            return []

        texts_to_predict = [item[1] for item in valid_items]
        predictions = self.predict(texts_to_predict)

        for (idx, raw_val), pred in zip(valid_items, predictions):
            label = pred["predicted_label"].lower().strip()
            confidence = pred["confidence"]

            if label in BENIGN_LABELS:
                continue

            # Determine category and severity
            cat_sev = LABEL_CATEGORY_MAP.get(label)
            if cat_sev:
                category, severity = cat_sev
            else:
                category = FindingCategory.SUSPICIOUS_CONTENT
                severity = Severity.HIGH

            findings.append({
                "finding_id": generate_storage_id(),
                "category": category.value,
                "subcategory": f"ml_{label}",
                "severity": severity.value,
                "confidence": round(confidence, 4),
                "dataset_id": dataset_id,
                "record_index": idx,
                "field_name": column,
                "location": f"row:{idx},col:{column}",
                "evidence": pred["text_preview"],
                "explanation": f"ML classifier identified content as sensitive '{label}' with confidence {confidence:.2%}.",
                "detection_method": DetectionMethod.ML_CLASSIFIER.value,
                "recommended_action": PolicyDecision.REVIEW.value,
                "review_status": ReviewStatus.PENDING.value,
                "remediation_status": RemediationStatus.PENDING.value,
                "created_at": now.isoformat(),
            })

        logger.info(
            "ML classification of dataset %s column '%s' generated %d findings",
            dataset_id,
            column,
            len(findings),
        )
        return findings
