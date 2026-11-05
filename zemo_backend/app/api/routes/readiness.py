"""Readiness evaluation API routes."""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel

from app.core.exceptions import DatasetNotFoundError
from app.models.enums import IntendedOperation
from app.repositories.dataset_repo import DatasetRepository
from app.schemas.readiness import ReadinessReport
from app.services.readiness import ReadinessService

router = APIRouter()
repo = DatasetRepository()
readiness_service = ReadinessService(repo=repo)


class ReadinessAssessRequest(BaseModel):
    dataset_id: str
    intended_operation: IntendedOperation = IntendedOperation.TRAINING


@router.post("/readiness/assess", response_model=ReadinessReport, status_code=status.HTTP_200_OK)
async def assess_readiness(request: ReadinessAssessRequest) -> ReadinessReport:
    return readiness_service.assess(
        dataset_id=request.dataset_id,
        intended_operation=request.intended_operation,
    )


@router.get("/readiness/{dataset_id}", response_model=ReadinessReport)
async def get_readiness(dataset_id: str) -> ReadinessReport:
    if not repo.exists(dataset_id):
        raise DatasetNotFoundError(
            f"Dataset '{dataset_id}' not found.",
            detail={"dataset_id": dataset_id},
        )
    cached = repo.get_readiness(dataset_id)
    if cached:
        return ReadinessReport.model_validate(cached)
    return readiness_service.assess(dataset_id=dataset_id)

