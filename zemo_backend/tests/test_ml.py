"""
Unit tests for ML ContentClassifier subsystem.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.exceptions import ModelNotAvailableError
from app.ml.classifier import ContentClassifier
from app.models.enums import FindingCategory, Severity
from app.repositories.dataset_repo import DatasetRepository


def test_classifier_untrained_state(tmp_path) -> None:
    """Untrained classifier should report is_available=False and raise on predict."""
    model_file = tmp_path / "nonexistent" / "model.joblib"
    classifier = ContentClassifier(model_path=model_file)

    assert not classifier.is_available()

    with pytest.raises(ModelNotAvailableError):
        classifier.predict(["Some sample text"])


def test_classifier_training_and_prediction(tmp_path) -> None:
    """Train classifier on sample texts, verify metrics and predictions."""
    model_file = tmp_path / "models" / "classifier.joblib"
    classifier = ContentClassifier(model_path=model_file)

    texts = [
        "Please send the report to john@example.com",
        "Reach me at alice.smith@corp.org for billing",
        "The quick brown fox jumps over the lazy dog",
        "The shipment of electronics has arrived safely",
        "My phone number is 555-123-4567 call anytime",
        "Customer phone line 555-987-6543 open 24/7",
        "Total revenue for Q3 showed steady growth",
        "Weather forecast predicts sunny skies tomorrow",
    ]
    labels = [
        "email",
        "email",
        "clean",
        "clean",
        "phone",
        "phone",
        "clean",
        "clean",
    ]

    metrics = classifier.train(texts, labels)

    assert classifier.is_available()
    assert metrics["accuracy"] >= 0.80
    assert metrics["feature_count"] > 0
    assert "email" in metrics["class_distribution"]
    assert "clean" in metrics["class_distribution"]
    assert model_file.exists()

    # Predict with trained model
    preds = classifier.predict(["Contact test@example.com immediately", "The sunny park was calm"])
    assert len(preds) == 2
    assert preds[0]["predicted_label"] == "email"
    assert preds[0]["confidence"] > 0.0
    assert "text_preview" in preds[0]


def test_classify_dataset_column(tmp_path, dataset_repo: DatasetRepository) -> None:
    """Classify dataset column and verify findings for sensitive rows."""
    model_file = tmp_path / "models" / "classifier.joblib"
    classifier = ContentClassifier(repo=dataset_repo, model_path=model_file)

    # Train
    texts = [
        "contact@company.com",
        "billing@domain.org",
        "standard public note",
        "inventory confirmed",
    ]
    labels = ["email", "email", "clean", "clean"]
    classifier.train(texts, labels)

    # Setup dataset
    dataset_id = "ds-ml-test"
    df = pd.DataFrame({
        "notes": ["public message", "user email: ceo@domain.com", "all good"]
    })
    dataset_repo.save_dataframe(dataset_id, df)
    dataset_repo.save_metadata({
        "dataset_id": dataset_id,
        "original_filename": "ml_test.csv",
        "sanitized_filename": "ml_test.csv",
        "format": "csv",
        "record_count": 3,
        "column_count": 1,
        "columns": ["notes"],
        "file_size_bytes": 100,
        "status": "parsed",
        "version": 1,
    })

    findings = classifier.classify_dataset_column(dataset_id, "notes")
    assert len(findings) >= 1
    assert any(f["category"] == FindingCategory.EMAIL.value for f in findings)
    assert any(f["detection_method"] == "ml_classifier" for f in findings)
