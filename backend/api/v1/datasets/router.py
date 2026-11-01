from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from core.config import Settings, get_settings
from core.database.models.dataset import DatasetFormat, DatasetStatus
from core.database.models.dataset_version import DatasetVersionStatus
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from services.dataset import get_dataset_service
from services.scan import get_scan_service
from services.policy import get_policy_evaluation_service
from services.sanitization import get_sanitization_service
from services.readiness import get_readiness_service

from .schemas import (
    AddVersionRequest,
    DatasetCreateResponse,
    DatasetListResponse,
    DatasetVersionResponse,
    PreviewResponse,
    ProfileResponse,
    ProcessDatasetRequest,
    ValidateDatasetRequest,
    ProfileDatasetRequest,
    ValidationResponse,
    DeleteDatasetResponse,
)
from .scan_schemas import (
    ScanCreateRequest,
    ScanResponse,
    ScanListResponse,
    FindingResponse,
    FindingListResponse,
    ScanReportResponse,
    RetryScanResponse,
)
from .policy_schemas import (
    PolicyCreate,
    PolicyUpdate,
    PolicyResponse,
    PolicyListResponse,
    PolicyRuleCreate,
    PolicyRuleUpdate,
    PolicyRuleResponse,
    PolicyRuleListResponse,
    PolicyEvaluationCreate,
    PolicyEvaluationResponse,
    PolicyEvaluationListResponse,
    ApprovalRecordCreate,
    ApprovalRecordResponse,
    PolicyDecisionResponse,
)
from .sanitization_schemas import (
    SanitizationJobCreate,
    SanitizationJobResponse,
    SanitizationJobListResponse,
    SanitizationJobStatusResponse,
    SanitizationJobReportResponse,
)
from .readiness_schemas import (
    ReadinessAssessmentCreate,
    ReadinessAssessmentResponse,
    ReadinessAssessmentListResponse,
    ReadinessAssessmentReport,
)

logger = get_logger(__name__)

# Include sub-routers
from .policy_router import router as policy_router
from .sanitization_router import router as sanitization_router
from .readiness_router import router as readiness_router

router = APIRouter(prefix="/datasets", tags=["datasets"])
router.include_router(policy_router, prefix="/policies", tags=["policies"])
router.include_router(sanitization_router, prefix="/sanitization", tags=["sanitization"])
router.include_router(readiness_router, prefix="/readiness", tags=["readiness"])


def validate_file_format(filename: str) -> str:
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    if ext not in ("csv", "json", "jsonl"):
        raise ValidationError(f"Unsupported file format: {ext}. Supported: csv, json, jsonl")
    return ext


@router.post(
    "",
    response_model=DatasetCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and register a new dataset",
)
async def upload_dataset(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    if not file.filename:
        raise ValidationError("No filename provided")

    format = validate_file_format(file.filename)

    content = await file.read()
    if len(content) > settings.max_file_size_bytes:
        raise ValidationError(
            f"File size exceeds maximum allowed size of {settings.max_file_size_mb}MB",
            {"max_size_mb": settings.max_file_size_mb, "received_bytes": len(content)},
        )

    service = get_dataset_service()
    try:
        dataset = await service.create_dataset(content, file.filename, format)
    except ValidationError as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        raise

    return DatasetCreateResponse.model_validate(dataset)


@router.get(
    "",
    response_model=DatasetListResponse,
    summary="List datasets with pagination",
)
async def list_datasets(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    settings: Settings = Depends(get_settings),
):
    page_size = min(page_size, settings.max_page_size)

    service = get_dataset_service()
    datasets, total = await service.list_datasets(page=page, page_size=page_size, status=status_filter)

    total_pages = (total + page_size - 1) // page_size

    return DatasetListResponse(
        items=[DatasetCreateResponse.model_validate(d) for d in datasets],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/{dataset_id}",
    response_model=DatasetCreateResponse,
    summary="Get dataset metadata",
)
async def get_dataset(dataset_id: str):
    service = get_dataset_service()
    try:
        dataset = await service.get_dataset(dataset_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return DatasetCreateResponse.model_validate(dataset)


@router.post(
    "/{dataset_id}/validate",
    response_model=DatasetVersionResponse,
    summary="Validate a dataset (check format, structure)",
)
async def validate_dataset(dataset_id: str, request: ValidateDatasetRequest = ValidateDatasetRequest()):
    service = get_dataset_service()
    try:
        version = await service.validate_dataset(dataset_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return DatasetVersionResponse.model_validate(version)


@router.post(
    "/{dataset_id}/profile",
    response_model=DatasetVersionResponse,
    summary="Profile a validated dataset",
)
async def profile_dataset(dataset_id: str, request: ProfileDatasetRequest = ProfileDatasetRequest()):
    service = get_dataset_service()
    try:
        version = await service.profile_dataset(dataset_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return DatasetVersionResponse.model_validate(version)


@router.post(
    "/{dataset_id}/process",
    response_model=DatasetVersionResponse,
    summary="Process (validate and profile) a dataset",
)
async def process_dataset(dataset_id: str, request: ProcessDatasetRequest = ProcessDatasetRequest()):
    service = get_dataset_service()
    try:
        version = await service.process_dataset(dataset_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return DatasetVersionResponse.model_validate(version)


@router.get(
    "/{dataset_id}/validation",
    response_model=ValidationResponse,
    summary="Get dataset validation results",
)
async def get_dataset_validation(
    dataset_id: str,
    version_id: Optional[str] = Query(None, description="Specific version ID (defaults to latest)"),
):
    service = get_dataset_service()
    try:
        validation = await service.get_validation(dataset_id, version_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return ValidationResponse(**validation)


@router.get(
    "/{dataset_id}/profile",
    response_model=ProfileResponse,
    summary="Get dataset profile",
)
async def get_dataset_profile(
    dataset_id: str,
    version_id: Optional[str] = Query(None, description="Specific version ID (defaults to latest)"),
):
    service = get_dataset_service()
    try:
        profile = await service.get_profile(dataset_id, version_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return ProfileResponse.model_validate(profile)


@router.get(
    "/{dataset_id}/preview",
    response_model=PreviewResponse,
    summary="Get bounded dataset preview",
)
async def get_dataset_preview(
    dataset_id: str,
    version_id: Optional[str] = Query(None, description="Specific version ID (defaults to latest)"),
    rows: int = Query(100, ge=1, le=1000, description="Number of rows to preview"),
):
    service = get_dataset_service()
    try:
        preview = await service.get_preview(dataset_id, version_id, rows)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return PreviewResponse.model_validate(preview)


@router.get(
    "/{dataset_id}/versions",
    response_model=list[DatasetVersionResponse],
    summary="List dataset versions",
)
async def list_dataset_versions(dataset_id: str):
    service = get_dataset_service()
    try:
        versions = await service.get_dataset_versions(dataset_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return [DatasetVersionResponse.model_validate(v) for v in versions]


@router.get(
    "/{dataset_id}/versions/{version_id}",
    response_model=DatasetVersionResponse,
    summary="Get version metadata",
)
async def get_dataset_version(dataset_id: str, version_id: str):
    service = get_dataset_service()
    try:
        version = await service.get_version(dataset_id, version_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return DatasetVersionResponse.model_validate(version)


@router.post(
    "/{dataset_id}/versions",
    response_model=DatasetVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new version to an existing dataset",
)
async def add_dataset_version(
    dataset_id: str,
    file: UploadFile = File(...),
    format: str = Form(...),
):
    if not file.filename:
        raise ValidationError("No filename provided")

    format = format.lower()
    if format not in ("csv", "json", "jsonl"):
        raise ValidationError(f"Unsupported format: {format}")

    settings = get_settings()
    content = await file.read()
    if len(content) > settings.max_file_size_bytes:
        raise ValidationError(
            f"File size exceeds maximum allowed size of {settings.max_file_size_mb}MB",
            {"max_size_mb": settings.max_file_size_mb, "received_bytes": len(content)},
        )

    service = get_dataset_service()
    try:
        version = await service.add_version(dataset_id, content, file.filename, format)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return DatasetVersionResponse.model_validate(version)


@router.delete(
    "/{dataset_id}",
    response_model=DeleteDatasetResponse,
    summary="Delete a dataset and all its versions and files",
)
async def delete_dataset(dataset_id: str):
    service = get_dataset_service()
    try:
        await service.delete_dataset(dataset_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return DeleteDatasetResponse(success=True, message="Dataset deleted successfully")


# Scan endpoints

@router.post(
    "/{dataset_id}/scans",
    response_model=ScanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new scan for a dataset",
)
async def create_scan(
    dataset_id: str,
    request: ScanCreateRequest = ScanCreateRequest(),
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
    "/{dataset_id}/scans/{scan_id}/run",
    response_model=ScanResponse,
    summary="Run a queued scan",
)
async def run_scan(dataset_id: str, scan_id: str):
    service = get_scan_service()
    try:
        scan = await service.run_scan(scan_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return ScanResponse.model_validate(scan)


@router.get(
    "/{dataset_id}/scans",
    response_model=ScanListResponse,
    summary="List scans for a dataset with pagination",
)
async def list_scans(
    dataset_id: str,
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
    "/{dataset_id}/scans/{scan_id}",
    response_model=ScanResponse,
    summary="Get scan details",
)
async def get_scan(dataset_id: str, scan_id: str):
    service = get_scan_service()
    try:
        scan = await service.get_scan(scan_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    if scan.dataset_id != dataset_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found in this dataset")
    return ScanResponse.model_validate(scan)


@router.get(
    "/{dataset_id}/scans/{scan_id}/findings",
    response_model=FindingListResponse,
    summary="List scan findings with pagination and filters",
)
async def list_scan_findings(
    dataset_id: str,
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
    "/{dataset_id}/scans/{scan_id}/findings/{finding_id}",
    response_model=FindingResponse,
    summary="Get a specific finding",
)
async def get_finding(dataset_id: str, scan_id: str, finding_id: str):
    service = get_scan_service()
    try:
        finding = await service.get_finding(finding_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    if finding.scan_id != scan_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Finding not found in this scan")

    return FindingResponse.model_validate(finding)


@router.get(
    "/{dataset_id}/scans/{scan_id}/report",
    summary="Get full scan report with findings grouped by category",
)
async def get_scan_report(dataset_id: str, scan_id: str):
    service = get_scan_service()
    try:
        report = await service.get_scan_report(scan_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return report


@router.post(
    "/{dataset_id}/scans/{scan_id}/retry",
    summary="Retry a failed or partial scan",
)
async def retry_scan(dataset_id: str, scan_id: str):
    service = get_scan_service()
    try:
        scan = await service.retry_scan(scan_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return {"scan": ScanResponse.model_validate(scan)}