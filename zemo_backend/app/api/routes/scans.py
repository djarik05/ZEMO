"""Scan execution API routes."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.core.exceptions import DatasetNotFoundError
from app.repositories.dataset_repo import DatasetRepository
from app.schemas.scan import ScanRequest, ScanResult
from app.services.scan_service import ScanService

router = APIRouter()
repo = DatasetRepository()
scan_service = ScanService(repo=repo)


@router.post("/scans", response_model=ScanResult, status_code=status.HTTP_200_OK)
async def run_scan(request: ScanRequest) -> ScanResult:
    return scan_service.run_scan(request)


@router.get("/scans/{dataset_id}", response_model=ScanResult)
async def get_scan(dataset_id: str) -> ScanResult:
    if not repo.exists(dataset_id):
        raise DatasetNotFoundError(
            f"Dataset '{dataset_id}' not found.",
            detail={"dataset_id": dataset_id},
        )
    scan_data = repo.get_scan(dataset_id)
    if not scan_data:
        return scan_service.run_scan(ScanRequest(dataset_id=dataset_id))
    return ScanResult.model_validate(scan_data)

