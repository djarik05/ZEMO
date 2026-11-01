from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from core.config import Settings, get_settings
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from services.sanitization import get_sanitization_service
from .sanitization_schemas import (
    SanitizationJobCreate,
    SanitizationJobResponse,
    SanitizationJobListResponse,
    SanitizationJobStatusResponse,
    SanitizationJobReportResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/sanitization", tags=["sanitization"])


@router.post(
    "/jobs",
    response_model=SanitizationJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new sanitization job",
)
async def create_sanitization_job(request: SanitizationJobCreate):
    service = get_sanitization_service()
    try:
        job = await service.create_sanitization_job(
            dataset_id=request.dataset_id,
            version_id=request.version_id,
            scan_id=request.scan_id,
            evaluation_id=request.evaluation_id,
            rules=[r.model_dump() for r in request.rules],
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return SanitizationJobResponse.model_validate(job)


@router.post(
    "/jobs/{job_id}/run",
    response_model=SanitizationJobResponse,
    summary="Run a sanitization job",
)
async def run_sanitization_job(job_id: str):
    service = get_sanitization_service()
    try:
        job = await service.run_sanitization(job_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return SanitizationJobResponse.model_validate(job)


@router.get(
    "/jobs",
    response_model=SanitizationJobListResponse,
    summary="List sanitization jobs with pagination",
)
async def list_sanitization_jobs(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    settings: Settings = Depends(get_settings),
):
    page_size = min(page_size, settings.max_page_size)
    service = get_sanitization_service()
    jobs, total = await service.list_jobs(
        dataset_id=dataset_id,
        page=page,
        page_size=page_size,
    )
    total_pages = (total + page_size - 1) // page_size
    return SanitizationJobListResponse(
        items=[{"id": j.id, "dataset_id": j.dataset_id, "version_id": j.version_id, "status": j.status, "created_at": j.created_at.isoformat() if j.created_at else None} for j in jobs],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=SanitizationJobResponse,
    summary="Get sanitization job details",
)
async def get_sanitization_job(job_id: str):
    service = get_sanitization_service()
    try:
        job = await service.get_job(job_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return SanitizationJobResponse.model_validate(job)


@router.get(
    "/jobs/{job_id}/status",
    response_model=SanitizationJobStatusResponse,
    summary="Get sanitization job status with details",
)
async def get_sanitization_job_status(job_id: str):
    service = get_sanitization_service()
    try:
        status_info = await service.get_job_status(job_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return status_info


@router.get(
    "/jobs/{job_id}/report",
    response_model=dict,
    summary="Get sanitization job report",
)
async def get_sanitization_job_report(job_id: str):
    service = get_sanitization_service()
    try:
        report = await service.get_job_status(job_id)  # Reusing get_job_status for now
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return report