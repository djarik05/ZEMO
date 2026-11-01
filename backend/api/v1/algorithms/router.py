from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

try:
    from core.exceptions import AppValidationError as ValidationError, NotFoundError
except ImportError:
    from backend.core.exceptions import AppValidationError as ValidationError, NotFoundError

from services.ml.registry import get_algorithm_registry
from .schemas import (
    AlgorithmCompatibilityRequest,
    AlgorithmCompatibilityResponse,
    AlgorithmListResponse,
    AlgorithmResponse,
    ValidateHyperparametersRequest,
    ValidateHyperparametersResponse,
)

router = APIRouter(prefix="/algorithms", tags=["algorithms"])


@router.get(
    "",
    response_model=AlgorithmListResponse,
    summary="List all registered machine learning algorithms",
)
async def list_algorithms(
    task: Optional[str] = Query(
        None,
        description="Filter algorithms by task: 'classification', 'regression', or 'clustering'",
    ),
):
    registry = get_algorithm_registry()
    algos = registry.list_algorithms(task=task)
    return AlgorithmListResponse(
        total=len(algos),
        task_filter=task,
        algorithms=[AlgorithmResponse.model_validate(a.model_dump()) for a in algos],
    )


@router.get(
    "/{algorithm_id}",
    response_model=AlgorithmResponse,
    summary="Get algorithm specifications and hyperparameter constraints",
)
async def get_algorithm(
    algorithm_id: str,
    task: Optional[str] = Query(None, description="Optional task context for resolving ambiguous aliases"),
):
    registry = get_algorithm_registry()
    try:
        info = registry.get(algorithm_id, task=task)
        return AlgorithmResponse.model_validate(info.model_dump())
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/{algorithm_id}/validate",
    response_model=ValidateHyperparametersResponse,
    summary="Validate hyperparameter values against algorithm constraints",
)
async def validate_hyperparameters(
    algorithm_id: str,
    request: ValidateHyperparametersRequest,
):
    registry = get_algorithm_registry()
    try:
        info = registry.get(algorithm_id, task=request.task)
        is_valid, sanitized, errors = registry.validate_hyperparameters(
            algorithm_id,
            request.hyperparameters,
            task=request.task,
        )
        return ValidateHyperparametersResponse(
            valid=is_valid,
            algorithm_id=info.id,
            display_name=info.display_name,
            sanitized_hyperparameters=sanitized,
            errors=errors,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/{algorithm_id}/compatibility",
    response_model=AlgorithmCompatibilityResponse,
    summary="Check algorithm compatibility with a dataset task and target column",
)
async def check_compatibility(
    algorithm_id: str,
    request: AlgorithmCompatibilityRequest,
):
    registry = get_algorithm_registry()
    try:
        info = registry.get(algorithm_id, task=request.task)
        is_compatible, errors = registry.validate_compatibility(
            algorithm_id,
            task=request.task,
            target_type=request.target_type,
        )
        return AlgorithmCompatibilityResponse(
            compatible=is_compatible,
            algorithm_id=info.id,
            display_name=info.display_name,
            task=request.task,
            errors=errors,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
