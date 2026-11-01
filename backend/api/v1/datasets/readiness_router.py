from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from core.config import Settings, get_settings
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from services.readiness import get_readiness_service
from .readiness_schemas import (
    ReadinessAssessmentCreate,
    ReadinessAssessmentResponse,
    ReadinessAssessmentListResponse,
    ReadinessAssessmentReport,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/readiness", tags=["readiness"])


@router.post(
    "/assessments",
    response_model=ReadinessAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new readiness assessment for a dataset",
)
async def create_readiness_assessment(request: ReadinessAssessmentCreate):
    service = get_readiness_service()
    try:
        assessment = await service.assess_readiness(
            dataset_id=request.dataset_id,
            intended_use=request.intended_use,
            version_id=request.version_id,
            scan_id=request.scan_id,
            evaluation_id=request.evaluation_id,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return ReadinessAssessmentResponse.model_validate(assessment)


@router.get(
    "/assessments",
    response_model=ReadinessAssessmentListResponse,
    summary="List readiness assessments with pagination",
)
async def list_readiness_assessments(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    settings: Settings = Depends(get_settings),
):
    page_size = min(page_size, settings.max_page_size)
    service = get_readiness_service()
    assessments, total = await service.list_assessments(
        dataset_id=dataset_id,
        page=page,
        page_size=page_size,
    )
    total_pages = (total + page_size - 1) // page_size
    return ReadinessAssessmentListResponse(
        items=assessments,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/assessments/{assessment_id}",
    response_model=ReadinessAssessmentResponse,
    summary="Get readiness assessment details",
)
async def get_readiness_assessment(assessment_id: str):
    service = get_readiness_service()
    try:
        assessment = await service.get_assessment(assessment_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return ReadinessAssessmentResponse.model_validate(assessment)


@router.get(
    "/assessments/{assessment_id}/report",
    response_model=ReadinessAssessmentReport,
    summary="Get full readiness assessment report",
)
async def get_readiness_assessment_report(assessment_id: str):
    service = get_readiness_service()
    try:
        report = await service.get_assessment_report(assessment_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return report