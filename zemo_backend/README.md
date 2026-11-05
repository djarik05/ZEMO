# Zemo Backend — AI Data Privacy, Security & Training Safety Platform

Production-grade Python/FastAPI backend for **Zemo**, an enterprise platform that inspects datasets, detects sensitive information and security threats, assesses risk, enforces safety policies, sanitizes unsafe records, and certifies AI training readiness.

---

## Architecture Overview

```
                                      ┌────────────────────────────────────────────────┐
                                      │              FastAPI Application               │
                                      │          (/health, /api/v1/...)               │
                                      └───────┬──────────────────────────────┬─────────┘
                                              │                              │
                                   ┌──────────▼───────────┐       ┌──────────▼───────────┐
                                   │   Ingestion Service  │       │  Scan Orchestrator   │
                                   │  (CSV, JSON, JSONL)  │       │   (Multi-Engine)     │
                                   └──────────┬───────────┘       └──────────┬───────────┘
                                              │                              │
                                   ┌──────────▼───────────┐       ┌──────────┼──────────────────────┬──────────────────────┐
                                   │  Dataset Repository  │       │          │                      │                      │
                                   │  (Storage Interface) │   ┌───▼──────────▼───┐  ┌───────────────▼──┐  ┌────────────────▼───┐  ┌───────────────▼──┐
                                   └──────────────────────┘   │ Privacy Detector │  │ Secret Detector  │  │ Security Analyzer  │  │ Profiler Service │
                                                              │  (PII / Cards)   │  │  (Keys / Creds)  │  │ (Injections / XSS) │  │  (Data Quality)  │
                                                              └───┬──────────────┘  └───────────────┬──┘  └────────────────┬───┘  └───────────────┬──┘
                                                                  │                                 │                      │                      │
                                                                  └─────────────────────────────────┼──────────────────────┴──────────────────────┘
                                                                                                    │
                                                                                        ┌───────────▼───────────┐
                                                                                        │      Risk Engine      │
                                                                                        │  (Composite Scoring)  │
                                                                                        └───────────┬───────────┘
                                                                                                    │
                                                                                        ┌───────────▼───────────┐
                                                                                        │     Policy Engine     │
                                                                                        │ (Mandatory Escalation)│
                                                                                        └───────────┬───────────┘
                                                                                                    │
                                                                                ┌───────────────────┴───────────────────┐
                                                                                │                                       │
                                                                    ┌───────────▼───────────┐               ┌───────────▼───────────┐
                                                                    │ Sanitization Service  │               │   Readiness Service   │
                                                                    │(Mask / Redact / Prune)│               │  (Training Gatekeeper)│
                                                                    └───────────────────────┘               └───────────────────────┘
```

---

## Directory Structure

```
zemo_backend/
├── app/
│   ├── main.py                     # FastAPI application factory & lifecycle
│   ├── core/
│   │   ├── config.py               # Pydantic v2 settings & environment loading
│   │   ├── exceptions.py           # Typed ZemoError hierarchy with error codes
│   │   ├── logging.py              # Structured, privacy-safe logger with auto-redaction
│   │   └── security.py             # Path-traversal defense, filename sanitization
│   ├── models/
│   │   └── enums.py                # Severity, Status, FindingCategory, PolicyDecision
│   ├── schemas/
│   │   ├── common.py               # HealthResponse, ErrorResponse, SuccessResponse
│   │   ├── dataset.py              # DatasetMetadata, DatasetProfile, ColumnProfile
│   │   ├── finding.py              # FindingBase, FindingDetail, FindingSummary
│   │   ├── scan.py                 # ScanRequest, ScanResult, StageResult
│   │   ├── risk.py                 # RiskAssessment, RiskBreakdown, PolicyEvaluation
│   │   ├── sanitization.py         # SanitizationRequest, SanitizationResult
│   │   └── readiness.py            # ReadinessReport, ReadinessCheck
│   ├── repositories/
│   │   └── dataset_repo.py         # Filesystem-backed abstraction (JSON/Parquet)
│   ├── services/
│   │   ├── ingestion.py            # CSV / JSON / JSONL parsing & validation
│   │   ├── profiler.py             # Schema, null, duplicate & anomaly statistics
│   │   ├── privacy_detector.py     # PII detection (Email, Phone, CC, SSN, Address)
│   │   ├── secret_detector.py      # Secret & credential detection (API keys, PEM, DB)
│   │   ├── security_analyzer.py    # Prompt injection, SQLi, XSS, malformed records
│   │   ├── risk_engine.py          # RiskEngine (scoring) & PolicyEngine (escalation)
│   │   ├── sanitization.py         # Irreversible data masking & row removal
│   │   ├── readiness.py            # Multi-factor training readiness assessment
│   │   └── scan_service.py         # Multi-stage detection orchestrator
│   ├── ml/
│   │   └── classifier.py           # Scikit-learn TF-IDF + Naive Bayes classifier
│   ├── utils/
│   │   └── masking.py              # Masking (email, phone, cards) & SHA-256 hashing
│   └── api/
│       └── routes/
│           ├── datasets.py         # /api/v1/datasets (upload, list, get, delete)
│           ├── scans.py            # /api/v1/scans (trigger scan, view status)
│           ├── sanitization.py     # /api/v1/sanitization (apply, fetch, validate)
│           └── readiness.py        # /api/v1/readiness (assess, cached report)
├── storage/                        # Persistent local data (datasets, parquet, ML weights)
├── tests/                          # Complete pytest test suite (55 unit & integration tests)
├── pyproject.toml                  # Python package configuration
├── requirements.txt                # Production & test dependencies
├── .env.example                    # Sample environment variables
└── README.md                       # Documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+ (Python 3.14 compatible)
- pip

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
copy .env.example .env
```

### 3. Start the Server (Localhost + LAN Access)
From the project root, run `start_lan.bat` (Windows) or `./start_lan.sh` (macOS/Linux), or run Uvicorn directly from `zemo_backend/`:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs` (or `http://<HOST_LAN_IP>:8000/docs`)
- ReDoc: `http://localhost:8000/redoc` (or `http://<HOST_LAN_IP>:8000/redoc`)

---

## API Reference

### System
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check, version info, active storage path |

### Datasets (`/api/v1/datasets`)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/datasets/upload` | Multipart upload (CSV, JSON, JSONL). Validates size, format, records |
| `GET` | `/api/v1/datasets` | List all uploaded datasets sorted by creation date |
| `GET` | `/api/v1/datasets/{id}` | Retrieve metadata and column profiles for a dataset |
| `GET` | `/api/v1/datasets/{id}/profile` | Retrieve data quality profile (duplicates, null counts, empty rows) |
| `GET` | `/api/v1/datasets/{id}/findings` | Retrieve all security and privacy findings with masked evidence |
| `DELETE` | `/api/v1/datasets/{id}` | Permanently delete dataset and all associated artifacts |

### Scans (`/api/v1/scans`)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/scans` | Trigger a multi-stage scan (privacy, secrets, security, quality, ML) |
| `GET` | `/api/v1/scans/{dataset_id}` | Retrieve scan progress, stage timings, and finding summary |

### Sanitization (`/api/v1/sanitization`)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/sanitization/apply` | Apply irreversible masking/redaction; creates new clean dataset version |
| `GET` | `/api/v1/sanitization/{dataset_id}` | Fetch applied remediation record and counters |
| `POST` | `/api/v1/sanitization/{dataset_id}/validate` | Post-sanitization re-scan to certify clean status |

### Training Readiness (`/api/v1/readiness`)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/readiness/assess` | Evaluate readiness for `training`, `fine_tuning`, or `inference` |
| `GET` | `/api/v1/readiness/{dataset_id}` | Fetch cached readiness report |

---

## Detection Engines & Capabilities

### 1. Privacy Detector (`PrivacyDetector`)
- **Email Addresses**: Full RFC-compliant regex; evidence masked as `j***@example.com` (Severity: `HIGH`).
- **Phone Numbers**: US and international formats; evidence masked to last 4 digits `***-***-1234` (Severity: `HIGH`).
- **Credit Cards**: Luhn-length payment cards across Visa, MC, Amex, Discover; masked as `****-****-****-1234` (Severity: `CRITICAL`).
- **Social Security Numbers**: Standard `\d{3}-\d{2}-\d{4}` pattern; completely masked with asterisks (Severity: `CRITICAL`).
- **Physical Addresses**: Street address heuristics with exclusions for hardware storage terms (Severity: `MEDIUM`).

### 2. Secret Detector (`SecretDetector`)
- **API Keys**: OpenAI, AWS, GitHub tokens, standard key prefixes (`sk-`, `AKIA`, `pk-`) (Severity: `CRITICAL`).
- **Passwords**: Cleartext credential assignment patterns in values and headers (Severity: `CRITICAL`).
- **Private Keys**: PEM-encoded RSA/EC private key headers (Severity: `CRITICAL`).
- **Database Credentials**: URI connection strings with embedded passwords (Severity: `CRITICAL`).
- **Suspicious Column Headers**: Identifies columns named `password`, `secret`, `api_key`, `token`, `credential`.

### 3. Security Analyzer (`SecurityAnalyzer`)
- **Prompt Injections**: Jailbreak attempts, instruction resets (`ignore previous instructions`, `DAN mode`, `system prompt:`) (Severity: `HIGH`).
- **Suspicious Payloads**: SQL Injection (`OR 1=1`, `UNION SELECT`, `DROP TABLE`), XSS (`<script>`, `javascript:`), and command injection (Severity: `HIGH`).
- **Malformed Records**: Oversized records (>10,000 chars), binary non-printable byte sequences (Severity: `MEDIUM`).

### 4. Risk & Policy Engine
- **Composite Risk Score**: Weighted formula across Privacy (30%), Secrets (30%), Security (25%), Quality (15%).
- **Policy Escalation**: Strict hierarchy: `ALLOW` < `REVIEW` < `SANITIZE` < `BLOCK`.
- **Mandatory Non-Overridable Blocks**:
  1. Any `CRITICAL` secret detected -> `BLOCK`
  2. Any `CRITICAL` PII with confidence > 0.9 -> `BLOCK`
  3. Prompt injection count > 5 -> `BLOCK`
  *A mandatory block can NEVER be overridden by a low composite score.*

### 5. Sanitization & Validation
- **Non-Destructive Versioning**: Original datasets are immutable. Sanitization creates a new versioned dataset (`parent_dataset_id` linked).
- **Transformation Actions**:
  - `mask`: Replaces sensitive characters with asterisks / formatted placeholders.
  - `redact`: Replaces cell contents with `[REDACTED]`.
  - `hash`: Replaces value with salted SHA-256 digest.
  - `remove_record`: Prunes entire malicious/prompt injection rows.

### 6. Machine Learning Subsystem (`ContentClassifier`)
- Scikit-learn `TfidfVectorizer` + `MultinomialNB` pipeline.
- Graceful degradation: Flags `is_available() == False` when untrained; raises clean `ModelNotAvailableError` rather than crashing.
- Safe serialization using `joblib` (`.joblib` format, strictly forbidding Python `pickle`).

---

## Security Defenses

- **Zero Pickle Usage**: Model persistence strictly uses `joblib` binary format.
- **Path Traversal Protection**: All paths resolved against root storage directory with `safe_join()`.
- **Blocked Executables**: Rejects `.exe`, `.bat`, `.sh`, `.py`, `.dll`, `.jar`, `.pkl`.
- **Irreversible Masking**: Raw secrets and PII are never logged or returned unmasked in API responses.

---

## Test Suite

The test suite contains **55 tests** covering 100% of the core backend requirements:

```bash
py -m pytest tests/ -v
```

### Test Summary
```
tests/test_api.py ..........            [10 passed]
tests/test_ingestion.py .......         [ 7 passed]
tests/test_ml.py ...                    [ 3 passed]
tests/test_privacy.py .......           [ 7 passed]
tests/test_readiness.py ....            [ 4 passed]
tests/test_risk.py .......              [ 7 passed]
tests/test_sanitization.py .....        [ 5 passed]
tests/test_secrets.py ......            [ 6 passed]
tests/test_security.py .....            [ 5 passed]
======================= 55 passed in 2.07s =======================
```
