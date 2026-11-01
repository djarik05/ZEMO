# ZEMOO Architecture Audit & ML Platform Implementation Plan

## 1. Executive Summary

Zemo is an AI/ML model-building and training software platform. The system programmatically ingests user datasets, validates and profiles data schemas, applies user-directed preprocessing pipelines, trains machine learning models via Scikit-learn, evaluates model performance, and provides inference and export capabilities.

This document records the **Phase 0 Audit** of the existing codebase and lays out the foundational architecture for the subsequent implementation phases.

---

## 2. Complete Project Audit

### 2.1 Feature Implementation Matrix

| Module / Component | State | Description |
| :--- | :--- | :--- |
| **Dataset Ingestion API** | **Fully Implemented** | `POST /api/v1/datasets`, `POST /api/v1/datasets/{id}/versions` stores raw uploads safely via `StorageService` using SHA-256 hashing and UUID filenames. |
| **Dataset Validation & Profiling** | **Fully Implemented** | Computes descriptive stats, null counts, data types, and distributions (`/validate`, `/profile`, `/process`). |
| **Governance & Security Scans** | **Fully Implemented** | Policy rule evaluations, PII detectors (email, phone, secrets, prompt injection), and readiness gating. |
| **LLM Provider Integrations** | **Partially Implemented** | Provider registry models and adapters for OpenAI, Anthropic, Gemini, Mistral, Ollama. |
| **Database & Migrations** | **Fully Implemented** | SQLAlchemy declarative models with Alembic migrations at head revision `f5a3b2c1d4e6`. |
| **Frontend UI/UX** | **Fully Implemented (UI)** | Comprehensive design system (Calm Organic & Dark Editorial), reactive panels, forms, and charts. |
| **Frontend-Backend Connectivity** | **Demonstration Mode** | `js/app.js` currently parses CSVs client-side using `FileReader` and simulates training via timers/mock metrics. |
| **ML Training Engine** | **Not Implemented Yet** | No Scikit-learn model fitting, hyperparameter validation, pipeline construction, or metric evaluation backend exists yet. |
| **Model Registry & Inference** | **Not Implemented Yet** | No model artifact serialization (`joblib`), persistence, or prediction API endpoints exist yet. |

---

### 2.2 Frontend Controls Audit (Currently Disconnected)

1. **File Upload Dropzone**: Currently reads file client-side using JavaScript `FileReader` instead of submitting `multipart/form-data` to `POST /api/v1/datasets`.
2. **Column Inclusion/Exclusion Toggles**: Stores inclusion state in memory (`state.restrictions.columns`) but does not transmit feature selection to a backend training configuration.
3. **Data Quality & Sensitive Data Toggles**: Stored in local UI state; needs wiring to dataset policy evaluation and sanitization endpoints.
4. **Training Constraints**: Target variable dropdown, missing value strategy (`drop`, `impute_mean`, `impute_median`, `impute_mode`), encoding (`one-hot`, `label`, `auto`), and split ratio are in UI state only.
5. **Model Lab Configuration**: Algorithm selection matrix (Linear Regression, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, K-Means) and dynamic hyperparameter sliders are not wired to an execution backend.
6. **Training Queue & Live Progress**: Uses `setInterval` to fake training duration and log events tagged `[DEMO]`.
7. **Model Evaluation & Export**: Confusion matrix and ROC-AUC visualizations display placeholder figures without real test set inferences.

---

### 2.3 Reusable Backend Assets

- **Storage Engine**: `services.storage.StorageService` provides robust chunked file saving, unique file naming, SHA-256 content verification, and path resolution.
- **Data Parsers & Profilers**: `parsers.parsers` and `parsers.profiling` provide column type detection, null counting, and min/max/mean statistical summaries.
- **Database Session & Models**: `core.database.config.get_db_session()` and Base models for Datasets and DatasetVersions.
- **Exception Framework**: `core.exceptions.AppValidationError`, `NotFoundError`, and structured JSON responses.
- **Logging**: `core.logging.get_logger()` structured logging with contextual metadata.

---

### 2.4 Critical Architectural Distinction: Dataset Preview vs. ML Engine

- **Preview Parsing**: `parsers.parsers.CSVParser` intentionally loads datasets with `dtype=str, na_filter=False` to safely render raw text in HTML tables without type coercion crashes.
- **ML Engine Loading**: Machine learning requires native numeric floats, integers, boolean types, and proper `NaN` / `None` sentinel detection.
- **Solution Introduced**: `services.ml.loader.DatasetLoader` has been added. It reads directly from original disk storage (`StorageService.get_file_path`) and utilizes Pandas/Numpy type inference with multi-encoding fallback (UTF-8 / Latin-1) and configurable NA tokens.

---

## 3. Verified System Health

1. **FastAPI Application**: Starts cleanly on `0.0.0.0:8000` with CORS middleware, exception handlers, and API v1 routers.
2. **Alembic Database**: Verified at head migration (`f5a3b2c1d4e6`).
3. **Existing Tests**: 52 tests passing (47 existing dataset/health tests + 5 newly implemented ML loader tests).
4. **Dependencies**: `backend/requirements.txt` updated to include `pandas`, `scikit-learn`, `numpy`, and `joblib`.
5. **Model Mapping Fix**: Resolved `ModelCapabilityInfo.model_metadata` conflict with SQLAlchemy `Base.metadata`.

---

## 4. Phase-by-Phase Roadmap

- **Phase 0 (Current)**: Project audit, dependency resolution, architecture documentation, independent ML data loader preparation.
- **Phase 1**: Dataset connection between frontend and backend (upload, storage, real backend preview, server-side profiling).
- **Phase 2**: Preprocessing pipeline engine (ColumnTransformer, scalers, imputers, encoders, train/test split, data leakage prevention).
- **Phase 3**: Core ML training engine (Scikit-learn model builders, hyperparameter mapping, synchronous/asynchronous execution).
- **Phase 4**: Real-time training progress, metrics calculation (accuracy, precision, recall, F1, ROC-AUC, RMSE, MAE, R2), and confusion matrix generation.
- **Phase 5**: Trained model persistence, versioning, metadata serialization via `joblib`.
- **Phase 6**: Model evaluation comparison UI and inference prediction API.
- **Phase 7**: End-to-end integration polish, verification, and automated regression testing.
