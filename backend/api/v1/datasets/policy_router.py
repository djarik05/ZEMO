from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from core.config import Settings, get_settings
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from services.policy import get_policy_evaluation_service
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

logger = get_logger(__name__)

router = APIRouter(prefix="/policies", tags=["policies"])


@router.post(
    "",
    response_model=PolicyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new policy",
)
async def create_policy(request: PolicyCreate):
    service = get_policy_evaluation_service()
    try:
        policy = await service.create_policy(
            name=request.name,
            description=request.description,
            intended_uses=request.intended_uses,
            default_action=request.default_action,
            created_by=request.created_by,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return PolicyResponse.model_validate(policy)


@router.get(
    "",
    response_model=PolicyListResponse,
    summary="List policies with pagination",
)
async def list_policies(
    active_only: bool = Query(True, description="Filter by active policies only"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    settings: Settings = Depends(get_settings),
):
    page_size = min(page_size, settings.max_page_size)
    service = get_policy_evaluation_service()
    policies = await service.list_policies(active_only=active_only)

    # Apply pagination
    total = len(policies)
    total_pages = (total + page_size - 1) // page_size
    start = (page - 1) * page_size
    end = start + page_size
    paginated = policies[start:end]

    return PolicyListResponse(
        items=[PolicyResponse.model_validate(p) for p in paginated],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/{policy_id}",
    response_model=PolicyResponse,
    summary="Get policy details",
)
async def get_policy(policy_id: str):
    service = get_policy_evaluation_service()
    try:
        policy = await service.get_policy(policy_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return PolicyResponse.model_validate(policy)


@router.patch(
    "/{policy_id}",
    response_model=PolicyResponse,
    summary="Update a policy",
)
async def update_policy(policy_id: str, request: PolicyUpdate):
    service = get_policy_evaluation_service()
    try:
        policy = await service.update_policy(
            policy_id=policy_id,
            name=request.name,
            description=request.description,
            intended_uses=request.intended_uses,
            default_action=request.default_action,
            is_active=request.is_active,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return PolicyResponse.model_validate(policy)


@router.delete(
    "/{policy_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a policy",
)
async def delete_policy(policy_id: str):
    service = get_policy_evaluation_service()
    try:
        await service.delete_policy(policy_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


# Policy Rule endpoints

@router.post(
    "/{policy_id}/rules",
    response_model=PolicyRuleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new policy rule",
)
async def create_policy_rule(policy_id: str, request: PolicyRuleCreate):
    service = get_policy_evaluation_service()
    try:
        rule = await service.create_policy_rule(
            policy_id=policy_id,
            name=request.name,
            description=request.description,
            category=request.category.value,
            condition=request.condition,
            action=request.action,
            priority=request.priority,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return PolicyRuleResponse.model_validate(rule)


@router.get(
    "/{policy_id}/rules",
    response_model=PolicyRuleListResponse,
    summary="List policy rules with pagination",
)
async def list_policy_rules(
    policy_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    # TODO: Implement list policy rules with pagination
    # For now, return empty list
    return PolicyRuleListResponse(
        items=[],
        total=0,
        page=page,
        page_size=page_size,
        total_pages=0,
    )


@router.get(
    "/{policy_id}/rules/{rule_id}",
    response_model=PolicyRuleResponse,
    summary="Get policy rule details",
)
async def get_policy_rule(policy_id: str, rule_id: str):
    # TODO: Implement get policy rule
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


@router.patch(
    "/{policy_id}/rules/{rule_id}",
    response_model=PolicyRuleResponse,
    summary="Update a policy rule",
)
async def update_policy_rule(policy_id: str, rule_id: str, request: PolicyRuleUpdate):
    # TODO: Implement update policy rule
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


@router.delete(
    "/{policy_id}/rules/{rule_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a policy rule",
)
async def delete_policy_rule(policy_id: str, rule_id: str):
    # TODO: Implement delete policy rule
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


# Policy Evaluation endpoints

@router.post(
    "/{policy_id}/evaluations",
    response_model=PolicyEvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new policy evaluation for a dataset",
)
async def create_policy_evaluation(policy_id: str, request: PolicyEvaluationCreate):
    service = get_policy_evaluation_service()
    try:
        evaluation = await service.evaluate_dataset(
            dataset_id=request.dataset_id,
            policy_id=policy_id,
            version_id=request.version_id,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return PolicyEvaluationResponse.model_validate(evaluation)


@router.get(
    "/{policy_id}/evaluations",
    response_model=PolicyEvaluationListResponse,
    summary="List policy evaluations with pagination",
)
async def list_policy_evaluations(
    policy_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    settings: Settings = Depends(get_settings),
):
    page_size = min(page_size, settings.max_page_size)
    service = get_policy_evaluation_service()
    evaluations, total = await service.list_evaluations(
        policy_id=policy_id,
        page=page,
        page_size=page_size,
    )
    total_pages = (total + page_size - 1) // page_size
    return PolicyEvaluationListResponse(
        items=[PolicyEvaluationResponse.model_validate(e) for e in evaluations],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/{policy_id}/evaluations/{evaluation_id}",
    response_model=PolicyEvaluationResponse,
    summary="Get policy evaluation details",
)
async def get_policy_evaluation(policy_id: str, evaluation_id: str):
    service = get_policy_evaluation_service()
    try:
        evaluation = await service.get_evaluation(evaluation_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    if evaluation.policy_id != policy_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation not found in this policy")
    return PolicyEvaluationResponse.model_validate(evaluation)


@router.post(
    "/{policy_id}/evaluations/{evaluation_id}/approve",
    response_model=PolicyEvaluationResponse,
    summary="Approve a policy evaluation",
)
async def approve_policy_evaluation(
    policy_id: str,
    evaluation_id: str,
    request: ApprovalRecordCreate,
):
    service = get_policy_evaluation_service()
    try:
        evaluation = await service.approve_evaluation(
            evaluation_id=evaluation_id,
            approved_by=request.approved_by,
            decision=request.decision,
            comment=request.comment,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return PolicyEvaluationResponse.model_validate(evaluation)


# Policy Decision endpoints

@router.get(
    "/{policy_id}/evaluations/{evaluation_id}/decisions",
    response_model=List[PolicyDecisionResponse],
    summary="List policy decisions for an evaluation",
)
async def list_policy_decisions(policy_id: str, evaluation_id: str):
    # TODO: Implement list policy decisions
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


@router.get(
    "/{policy_id}/decisions/{decision_id}",
    response_model=PolicyDecisionResponse,
    summary="Get policy decision details",
)
async def get_policy_decision(policy_id: str, decision_id: str):
    # TODO: Implement get policy decision
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


# Approval Record endpoints

@router.get(
    "/{policy_id}/approvals",
    response_model=List[ApprovalRecordResponse],
    summary="List approval records for a policy",
)
async def list_approval_records(policy_id: str):
    # TODO: Implement list approval records
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


@router.get(
    "/{policy_id}/approvals/{approval_id}",
    response_model=ApprovalRecordResponse,
    summary="Get approval record details",
)
async def get_approval_record(policy_id: str, approval_id: str):
    # TODO: Implement get approval record
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")