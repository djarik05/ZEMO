"""Sanitization API routes."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.core.exceptions import DatasetNotFoundError
from app.models.enums import DatasetStatus, Severity, utcnow
from app.repositories.dataset_repo import DatasetRepository
from app.schemas.sanitization import (
    PostValidationResult,
    SanitizationRequest,
    SanitizationResult,
)
from app.schemas.scan import ScanRequest
from app.services.readiness import ReadinessService
from app.services.sanitization import SanitizationService
from app.services.scan_service import ScanService

router = APIRouter()
repo = DatasetRepository()
sanitization_service = SanitizationService(repo=repo)
scan_service = ScanService(repo=repo)
readiness_service = ReadinessService(repo=repo)


@router.post("/sanitization/apply", response_model=SanitizationResult, status_code=status.HTTP_200_OK)
async def apply_sanitization(request: SanitizationRequest) -> SanitizationResult:
    return sanitization_service.sanitize(
        dataset_id=request.dataset_id,
        actions=request.actions,
        auto_apply_policy=request.auto_apply_policy,
    )


@router.get("/sanitization/{dataset_id}", response_model=SanitizationResult)
async def get_sanitization(dataset_id: str) -> SanitizationResult:
    if not repo.exists(dataset_id):
        raise DatasetNotFoundError(
            f"Dataset '{dataset_id}' not found.",
            detail={"dataset_id": dataset_id},
        )
    san_data = repo.get_sanitization(dataset_id)
    if not san_data:
        return sanitization_service.sanitize(dataset_id=dataset_id, auto_apply_policy=True)
    return SanitizationResult.model_validate(san_data)


@router.post("/sanitization/{dataset_id}/validate", response_model=PostValidationResult)
async def validate_sanitization(dataset_id: str) -> PostValidationResult:
    if not repo.exists(dataset_id):
        raise DatasetNotFoundError(
            f"Dataset '{dataset_id}' not found.",
            detail={"dataset_id": dataset_id},
        )
    san_data = repo.get_sanitization(dataset_id)
    target_id = san_data["sanitized_dataset_id"] if san_data and "sanitized_dataset_id" in san_data else dataset_id

    scan_res = scan_service.run_scan(
        ScanRequest(
            dataset_id=target_id,
            enable_privacy=True,
            enable_secrets=True,
            enable_security=True,
            enable_quality=False,
        )
    )
    readiness_service.assess(dataset_id=target_id)

    actionable_remaining = scan_res.summary.critical + scan_res.summary.high
    passed = actionable_remaining == 0
    if passed:
        repo.update_status(target_id, DatasetStatus.READY)
    else:
        repo.update_status(target_id, DatasetStatus.VALIDATED)

    checks = [
        f"Privacy scan: {scan_res.stages[0].finding_count if len(scan_res.stages) > 0 else 0} findings",
        f"Secret scan: {scan_res.stages[1].finding_count if len(scan_res.stages) > 1 else 0} findings",
        f"Security scan: {scan_res.stages[2].finding_count if len(scan_res.stages) > 2 else 0} findings",
    ]

    return PostValidationResult(
        sanitized_dataset_id=target_id,
        remaining_findings=actionable_remaining,
        passed=passed,
        checks=checks,
        validated_at=utcnow(),
    )

