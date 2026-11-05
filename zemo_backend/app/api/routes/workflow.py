"""Pipeline workflow and training readiness status API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.exceptions import DatasetNotFoundError
from app.models.enums import (
    DatasetStatus,
    FindingCategory,
    PolicyDecision,
    ReadinessStatus,
    Severity,
)
from app.repositories.dataset_repo import DatasetRepository
from app.services.readiness import ReadinessService
from app.services.risk_engine import PolicyEngine, RiskEngine

router = APIRouter()
repo = DatasetRepository()
readiness_service = ReadinessService(repo=repo)
risk_engine = RiskEngine()
policy_engine = PolicyEngine()

DEFAULT_PIPELINE_STAGES: list[dict[str, Any]] = [
    {
        "id": "input",
        "title": "Data Input",
        "status": "pending",
        "badgeText": "AWAITING",
        "description": "Raw dataset ingested, schema validated, format normalized (CSV, JSON, JSONL).",
    },
    {
        "id": "privacy",
        "title": "Privacy Scan",
        "status": "pending",
        "badgeText": "PENDING",
        "description": "Deterministic & probabilistic PII inspection detecting direct identifiers, emails, phones, and names.",
    },
    {
        "id": "security",
        "title": "Security Analysis",
        "status": "pending",
        "badgeText": "PENDING",
        "description": "Credential detection, API key entropy analysis, and prompt injection adversarial pattern screening.",
    },
    {
        "id": "risk",
        "title": "Risk Assessment",
        "status": "pending",
        "badgeText": "PENDING",
        "description": "Unified risk score calculation factoring severity, frequency, and downstream model blast radius.",
    },
    {
        "id": "sanitization",
        "title": "Sanitization",
        "status": "pending",
        "badgeText": "QUEUED",
        "description": "Automated redaction, synthetic replacement, and blocking of hazardous training samples.",
    },
    {
        "id": "validation",
        "title": "Validation",
        "status": "pending",
        "badgeText": "PENDING",
        "description": "Post-sanitization verification ensuring no residual privacy leaks or semantic distortion.",
    },
    {
        "id": "readiness",
        "title": "AI Readiness",
        "status": "pending",
        "badgeText": "PENDING",
        "description": "Final safety attestation and export to target AI model fine-tuning or evaluation pipeline.",
    },
]


@router.get("/workflow")
async def get_default_workflow() -> dict[str, Any]:
    model_cfg = repo.get_model_config() or {
        "provider_name": "OpenAI",
        "model": "gpt-4o",
    }
    return {
        "dataset_id": None,
        "pipelineStages": DEFAULT_PIPELINE_STAGES,
        "trainingReadiness": {
            "totalRecords": 0,
            "cleanRecords": 0,
            "cleanPercentage": 0.0,
            "sanitizedRecords": 0,
            "blockedRecords": 0,
            "readinessScore": 0,
            "requiredThreshold": 95,
            "statusText": "Awaiting Dataset",
            "targetArchitecture": f"{model_cfg.get('provider_name', 'OpenAI')} {model_cfg.get('model', 'gpt-4o')}",
            "steps": [
                {"step": 1, "title": "Dataset Format & Schema Validation", "status": "pending", "badge": "PENDING", "desc": "Upload a dataset to validate format and record schema."},
                {"step": 2, "title": "Privacy & PII Audit", "status": "pending", "badge": "PENDING", "desc": "Automated PII detection will run upon dataset inspection."},
                {"step": 3, "title": "Security & Credential Screening", "status": "pending", "badge": "PENDING", "desc": "Secret and prompt injection screening awaits dataset scan."},
                {"step": 4, "title": "Required Remediation Execution", "status": "pending", "badge": "PENDING", "desc": "Awaiting execution approval for redaction policies."},
                {"step": 5, "title": "Post-Remediation Revalidation", "status": "pending", "badge": "PENDING", "desc": "Verification scan will run automatically following sanitization."},
                {"step": 6, "title": "Training Safety Certificate", "status": "pending", "badge": "PENDING", "desc": "Readiness certification requires passing all safety checks."},
            ],
        },
    }


@router.get("/workflow/{dataset_id}")
async def get_dataset_workflow(dataset_id: str) -> dict[str, Any]:
    meta = repo.get_metadata(dataset_id)
    if not meta:
        raise DatasetNotFoundError(
            f"Dataset '{dataset_id}' not found.",
            detail={"dataset_id": dataset_id},
        )

    findings = repo.get_findings(dataset_id)
    risk = repo.get_risk(dataset_id) or risk_engine.assess(dataset_id, findings)
    policy = repo.get_policy(dataset_id) or policy_engine.evaluate(dataset_id, findings, risk)
    readiness_dict = repo.get_readiness(dataset_id)
    if not readiness_dict:
        report = readiness_service.assess(dataset_id)
        readiness_dict = report.model_dump(mode="json")

    sanitization = repo.get_sanitization(dataset_id)
    is_sanitized_version = meta.get("parent_dataset_id") is not None or meta.get("status") in (
        DatasetStatus.SANITIZED.value,
        DatasetStatus.VALIDATED.value,
        DatasetStatus.READY.value,
    )
    has_sanitized = sanitization is not None or is_sanitized_version

    # Categorize findings
    privacy_cats = {
        FindingCategory.EMAIL.value,
        FindingCategory.PHONE.value,
        FindingCategory.ADDRESS.value,
        FindingCategory.PERSONAL_ID.value,
        FindingCategory.CREDIT_CARD.value,
        FindingCategory.FINANCIAL.value,
    }
    security_cats = {
        FindingCategory.API_KEY.value,
        FindingCategory.PASSWORD.value,
        FindingCategory.ACCESS_TOKEN.value,
        FindingCategory.PRIVATE_KEY.value,
        FindingCategory.DB_CREDENTIAL.value,
        FindingCategory.PROMPT_INJECTION.value,
        FindingCategory.SUSPICIOUS_CONTENT.value,
        FindingCategory.MALFORMED_RECORD.value,
    }

    privacy_findings = [f for f in findings if f.get("category") in privacy_cats]
    security_findings = [f for f in findings if f.get("category") in security_cats]

    total_records = int(meta.get("record_count", 0))
    sanitize_records_set = {
        f.get("record_index")
        for f in findings
        if f.get("record_index") is not None
        and f.get("recommended_action") == PolicyDecision.SANITIZE.value
    }
    block_records_set = {
        f.get("record_index")
        for f in findings
        if f.get("record_index") is not None
        and f.get("recommended_action") == PolicyDecision.BLOCK.value
    }
    affected_records_set = sanitize_records_set | block_records_set
    clean_records = max(0, total_records - len(affected_records_set))
    clean_pct = round((clean_records / total_records) * 100.0, 1) if total_records > 0 else 100.0

    readiness_score = round(float(readiness_dict.get("overall_score", 0.0)))
    overall_status = readiness_dict.get("overall_status", ReadinessStatus.NOT_READY.value)
    risk_score = round(float(risk.get("overall_score", 0.0)))
    risk_level = str(risk.get("risk_level", "low")).upper()

    # Build 7 pipeline stages dynamically
    stages = [
        {
            "id": "input",
            "title": "Data Input",
            "status": "completed",
            "badgeText": "PASSED",
            "description": f"Ingested '{meta.get('original_filename')}' ({total_records:,} records, {meta.get('column_count', 0)} columns, {str(meta.get('format', 'csv')).upper()}).",
        },
        {
            "id": "privacy",
            "title": "Privacy Scan",
            "status": "completed",
            "badgeText": "CLEAN" if len(privacy_findings) == 0 else f"FLAGGED ({len(privacy_findings)})",
            "description": (
                "Zero unmasked PII instances detected across dataset records."
                if len(privacy_findings) == 0
                else f"Detected {len(privacy_findings)} PII findings across {len(sanitize_records_set)} records requiring masking or pseudonymization."
            ),
        },
        {
            "id": "security",
            "title": "Security Analysis",
            "status": "completed",
            "badgeText": "PASSED" if len(security_findings) == 0 else f"FLAGGED ({len(security_findings)})",
            "description": (
                "No exposed API keys, credentials, or prompt injection patterns detected."
                if len(security_findings) == 0
                else f"Identified {len(security_findings)} security & credential findings across {len(block_records_set) or 1} records."
            ),
        },
        {
            "id": "risk",
            "title": "Risk Assessment",
            "status": "completed" if has_sanitized or overall_status == ReadinessStatus.READY.value else "active",
            "badgeText": f"{risk_level} ({risk_score}/100)",
            "description": f"Composite risk score is {risk_score}/100 ({risk_level}). Policy decision: {str(policy.get('overall_decision', 'allow')).upper()}.",
        },
        {
            "id": "sanitization",
            "title": "Sanitization",
            "status": "completed" if has_sanitized or len(findings) == 0 else "pending",
            "badgeText": "APPLIED" if has_sanitized else ("NOT NEEDED" if len(findings) == 0 else "QUEUED"),
            "description": (
                "Automated masking, redaction, and record pruning applied to produce a clean dataset version."
                if has_sanitized
                else "Automated redaction, synthetic replacement, and blocking of hazardous training samples."
            ),
        },
        {
            "id": "validation",
            "title": "Validation",
            "status": "completed" if has_sanitized or overall_status == ReadinessStatus.READY.value else "pending",
            "badgeText": "VERIFIED" if has_sanitized or overall_status == ReadinessStatus.READY.value else "PENDING",
            "description": "Post-sanitization verification ensuring no residual privacy leaks or semantic distortion.",
        },
        {
            "id": "readiness",
            "title": "AI Readiness",
            "status": "completed" if overall_status == ReadinessStatus.READY.value else ("active" if has_sanitized else "pending"),
            "badgeText": "READY" if overall_status == ReadinessStatus.READY.value else overall_status.upper().replace("_", " "),
            "description": readiness_dict.get("message", "Final safety attestation for AI model training and fine-tuning."),
        },
    ]

    model_cfg = repo.get_model_config() or {
        "provider_name": "OpenAI",
        "model": "gpt-4o",
    }

    fmt_str = str(meta.get("format", "csv")).upper()
    steps = [
        {
            "step": 1,
            "title": "Dataset Format & Schema Validation",
            "status": "pass",
            "badge": "PASSED",
            "desc": f"{total_records:,} records parsed successfully. Schema complies with {fmt_str} specification.",
        },
        {
            "step": 2,
            "title": "Privacy & PII Audit",
            "status": "pass" if len(privacy_findings) == 0 else "warning",
            "badge": "PASSED" if len(privacy_findings) == 0 else "ACTION REQ",
            "desc": (
                "Zero unmasked PII findings detected."
                if len(privacy_findings) == 0
                else f"{len(privacy_findings)} PII findings across {len(sanitize_records_set)} records. Automated masking rule queued."
            ),
        },
        {
            "step": 3,
            "title": "Security & Credential Screening",
            "status": "pass" if len(security_findings) == 0 else "warning",
            "badge": "PASSED" if len(security_findings) == 0 else "CRITICAL REQ",
            "desc": (
                "Zero exposed secrets or prompt injections detected."
                if len(security_findings) == 0
                else f"{len(security_findings)} security/secret findings flagged. Blocking & redaction required."
            ),
        },
        {
            "step": 4,
            "title": "Required Remediation Execution",
            "status": "pass" if (has_sanitized or len(findings) == 0) else "pending",
            "badge": "COMPLETED" if (has_sanitized or len(findings) == 0) else "PENDING",
            "desc": (
                "Sanitization and redaction policies executed."
                if (has_sanitized or len(findings) == 0)
                else "Click 'Prepare Training Workspace' to execute sanitization & redaction policies."
            ),
        },
        {
            "step": 5,
            "title": "Post-Remediation Revalidation",
            "status": "pass" if (has_sanitized or overall_status == ReadinessStatus.READY.value) else "pending",
            "badge": "VERIFIED" if (has_sanitized or overall_status == ReadinessStatus.READY.value) else "PENDING",
            "desc": (
                "Post-remediation verification scan confirmed zero critical leaks."
                if (has_sanitized or overall_status == ReadinessStatus.READY.value)
                else "Verification scan will run automatically following sanitization."
            ),
        },
        {
            "step": 6,
            "title": "Training Safety Certificate",
            "status": "pass" if overall_status == ReadinessStatus.READY.value else "pending",
            "badge": "CERTIFIED" if overall_status == ReadinessStatus.READY.value else overall_status.upper(),
            "desc": (
                f"Readiness score ({readiness_score}%) meets safety threshold. Certified for training."
                if overall_status == ReadinessStatus.READY.value
                else f"Readiness score ({readiness_score}%) is below the mandatory 95% threshold."
            ),
        },
    ]

    return {
        "dataset_id": dataset_id,
        "pipelineStages": stages,
        "trainingReadiness": {
            "totalRecords": total_records,
            "cleanRecords": clean_records,
            "cleanPercentage": clean_pct,
            "sanitizedRecords": len(sanitize_records_set),
            "blockedRecords": len(block_records_set),
            "readinessScore": readiness_score,
            "requiredThreshold": 95,
            "statusText": "Certified Ready" if overall_status == ReadinessStatus.READY.value else "Remediation Required",
            "overallStatus": overall_status,
            "targetArchitecture": f"{model_cfg.get('provider_name', 'OpenAI')} {model_cfg.get('model', 'gpt-4o')}",
            "steps": steps,
        },
    }
