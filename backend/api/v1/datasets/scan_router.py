from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from core.config import Settings, get_settings
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from services.scan import get_scan_service

from .scan_schemas import (
    ScanCreateRequest,
    ScanResponse,
    ScanListResponse,
    FindingResponse,
    FindingListResponse,
    ScanReportResponse,
    RetryScanResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/scans", tags=["scans"])


@router.post(
    "",
    response_model=ScanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new scan for a dataset",
)
async def create_scan(
    dataset_id: str,
    request: ScanCreateRequest = ScanCreateRequest(),
    settings: Settings = Depends(get_settings),
):
    service = get_scan_service()
    try:
        scan = await service.create_scan(
            dataset_id=dataset_id,
            detectors=request.detectors,
            version_id=request.version_id,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return ScanResponse.model_validate(scan)


@router.post(
    "/{scan_id}/run",
    response_model=ScanResponse,
    summary="Run a queued scan",
)
async def run_scan(scan_id: str):
    service = get_scan_service()
    try:
        scan = await service.run_scan(scan_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return ScanResponse.model_validate(scan)


@router.get(
    "",
    response_model=ScanListResponse,
    summary="List scans with pagination",
)
async def list_scans(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by scan status"),
    settings: Settings = Depends(get_settings),
):
    page_size = min(page_size, settings.max_page_size)

    service = get_scan_service()
    scans, total = await service.list_scans(
        dataset_id=dataset_id,
        page=page,
        page_size=page_size,
        status=status_filter,
    )

    total_pages = (total + page_size - 1) // page_size

    return ScanListResponse(
        items=[ScanResponse.model_validate(s) for s in scans],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/{scan_id}",
    response_model=ScanResponse,
    summary="Get scan details",
)
async def get_scan(scan_id: str):
    service = get_scan_service()
    try:
        scan = await service.get_scan(scan_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return ScanResponse.model_validate(scan)


@router.get(
    "/{scan_id}/findings",
    response_model=FindingListResponse,
    summary="List scan findings with pagination and filters",
)
async def list_scan_findings(
    scan_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    category: Optional[str] = Query(None, description="Filter by finding category"),
    severity: Optional[str] = Query(None, description="Filter by finding severity"),
    settings: Settings = Depends(get_settings),
):
    page_size = min(page_size, settings.max_page_size)

    service = get_scan_service()
    try:
        findings, total = await service.get_scan_findings(
            scan_id=scan_id,
            page=page,
            page_size=page_size,
            category=category,
            severity=severity,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    total_pages = (total + page_size - 1) // page_size

    return FindingListResponse(
        items=[FindingResponse.model_validate(f) for f in findings],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/{scan_id}/findings/{finding_id}",
    response_model=FindingResponse,
    summary="Get a specific finding",
)
async def get_finding(scan_id: str, finding_id: str):
    service = get_scan_service()
    try:
        finding = await service.get_finding(finding_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    if finding.scan_id != scan_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Finding not found in this scan")

    return FindingResponse.model_validate(finding)


@router.get(
    "/{scan_id}/report",
    response_model=ScanReportResponse,
    summary="Get full scan report with findings grouped by category",
)
async def get_scan_report(scan_id: str):
    service = get_scan_service()
    try:
        report = await service.get_scan_report(scan_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return report


@router.post(
    "/{scan_id}/retry",
    response_model=RetryScanResponse,
    summary="Retry a failed or partial scan",
)
async def retry_scan(scan_id: str):
    service = get_scan_service()
    try:
        scan = await service.retry_scan(scan_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return RetryScanResponse(scan=ScanResponse.model_validate(scan))