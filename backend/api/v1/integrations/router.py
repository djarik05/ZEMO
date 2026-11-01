from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from core.config import Settings, get_settings
from core.database.models.integration import (
    ProviderCategory,
    ModelIntegrationStatus,
    AgentStatus,
    AgentFramework,
    AuthenticationMethod,
    ProviderType,
    ProviderStatus,
)
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from services.integrations.registry import get_integration_registry_service
from services.integrations.verification import get_verification_service
from services.integrations.provider_registry import get_provider_registry_service
from services.integrations.adapters import AdapterRegistry

from .schemas import (
    ModelIntegrationCreate,
    ModelIntegrationUpdate,
    ModelIntegrationResponse,
    ModelIntegrationListResponse,
    AgentIntegrationCreate,
    AgentIntegrationUpdate,
    AgentIntegrationResponse,
    AgentIntegrationListResponse,
    CapabilityCheckRequest,
    CapabilityCheckResponse,
    VerificationResponse,
    DatasetAccessGrantCreate,
    DatasetAccessGrantResponse,
    AdapterInfoResponse,
    ProviderCreate,
    ProviderUpdate,
    ProviderResponse,
    ProviderListResponse,
    ProviderTestResponse,
    ProviderSmokeTestRequest,
    ProviderSmokeTestResponse,
    ModelCapabilityCreate,
    ModelCapabilityUpdate,
    ModelCapabilityResponse,
    ModelCapabilityListResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/integrations", tags=["integrations"])


def validate_provider_create(request: ProviderCreate) -> None:
    try:
        ProviderType(request.provider_type)
    except ValueError:
        raise ValidationError(f"Invalid provider type: {request.provider_type}")

    try:
        AuthenticationMethod(request.auth_method)
    except ValueError:
        raise ValidationError(f"Invalid auth method: {request.auth_method}")

    if request.auth_method != "none" and not request.credential:
        raise ValidationError("Credential required for non-none auth methods")


def validate_model_integration_create(request: ModelIntegrationCreate) -> None:
    try:
        ProviderCategory(request.provider_category)
    except ValueError:
        raise ValidationError(f"Invalid provider category: {request.provider_category}")

    try:
        AuthenticationMethod(request.auth_method)
    except ValueError:
        raise ValidationError(f"Invalid auth method: {request.auth_method}")

    if request.auth_method != "none" and not request.credential:
        raise ValidationError("Credential required for non-none auth methods")

    available_adapters = AdapterRegistry.list_adapters()
    if request.adapter_type not in available_adapters:
        raise ValidationError(f"Unknown adapter type: {request.adapter_type}. Available: {available_adapters}")


def validate_agent_integration_create(request: AgentIntegrationCreate) -> None:
    try:
        AgentFramework(request.framework)
    except ValueError:
        raise ValidationError(f"Invalid agent framework: {request.framework}")

    try:
        AuthenticationMethod(request.auth_method)
    except ValueError:
        raise ValidationError(f"Invalid auth method: {request.auth_method}")

    if request.auth_method != "none" and not request.credential:
        raise ValidationError("Credential required for non-none auth methods")

    available_adapters = AdapterRegistry.list_adapters()
    if request.adapter_type not in available_adapters:
        raise ValidationError(f"Unknown adapter type: {request.adapter_type}. Available: {available_adapters}")


@router.get(
    "/adapters",
    response_model=List[AdapterInfoResponse],
    summary="List available adapter types",
)
async def list_adapters():
    adapters = [
        {"adapter_type": "openai", "category": "cloud", "description": "OpenAI API", "supported_capabilities": ["text_generation", "embeddings", "function_calling", "fine_tuning"]},
        {"adapter_type": "anthropic", "category": "cloud", "description": "Anthropic API", "supported_capabilities": ["text_generation", "function_calling"]},
        {"adapter_type": "google_gemini", "category": "cloud", "description": "Google Gemini API", "supported_capabilities": ["text_generation", "embeddings", "function_calling"]},
        {"adapter_type": "mistral", "category": "cloud", "description": "Mistral API", "supported_capabilities": ["text_generation", "embeddings", "function_calling", "fine_tuning"]},
        {"adapter_type": "ollama", "category": "local", "description": "Ollama local models", "supported_capabilities": ["text_generation", "embeddings"]},
        {"adapter_type": "openai_compatible", "category": "local", "description": "OpenAI-compatible local inference", "supported_capabilities": ["text_generation", "embeddings", "function_calling", "fine_tuning", "training"]},
        {"adapter_type": "custom_rest", "category": "custom", "description": "Custom REST API", "supported_capabilities": ["text_generation", "embeddings", "function_calling", "training"]},
        {"adapter_type": "custom_rest_agent", "category": "agent", "description": "Custom REST agent", "supported_capabilities": ["task_execution", "tool_use", "multi_step_reasoning", "data_analysis"]},
        {"adapter_type": "openai_compatible_agent", "category": "agent", "description": "OpenAI-compatible agent", "supported_capabilities": ["task_execution", "tool_use", "multi_step_reasoning", "data_analysis"]},
        {"adapter_type": "local_agent", "category": "agent", "description": "Local agent endpoint", "supported_capabilities": ["task_execution", "tool_use", "multi_step_reasoning", "data_analysis"]},
        {"adapter_type": "webhook_agent", "category": "agent", "description": "Webhook-based agent", "supported_capabilities": ["task_execution", "tool_use"]},
    ]
    return [AdapterInfoResponse(**a) for a in adapters]


# Provider endpoints
@router.get(
    "/providers",
    response_model=ProviderListResponse,
    summary="List AI providers",
)
async def list_providers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    provider_type: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    enabled: Optional[bool] = Query(None),
    settings: Settings = Depends(get_settings),
):
    page_size = min(page_size, settings.max_page_size)

    registry = get_provider_registry_service()
    providers, total = await registry.list_providers(
        page=page,
        page_size=page_size,
        provider_type=provider_type,
        status=status_filter,
        enabled=enabled,
    )

    total_pages = (total + page_size - 1) // page_size

    return ProviderListResponse(
        items=[ProviderResponse.model_validate(p) for p in providers],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.post(
    "/providers",
    response_model=ProviderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new AI provider",
)
async def create_provider(request: ProviderCreate):
    validate_provider_create(request)

    registry = get_provider_registry_service()
    try:
        provider = await registry.create_provider(
            name=request.name,
            display_name=request.display_name,
            provider_type=request.provider_type,
            base_url=str(request.base_url),
            auth_method=request.auth_method,
            credential=request.credential,
            default_headers=request.default_headers,
            extra_config=request.extra_config,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return ProviderResponse.model_validate(provider)


@router.get(
    "/providers/{provider_id}",
    response_model=ProviderResponse,
    summary="Get AI provider details",
)
async def get_provider(provider_id: str):
    registry = get_provider_registry_service()
    try:
        provider = await registry.get_provider(provider_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return ProviderResponse.model_validate(provider)


@router.patch(
    "/providers/{provider_id}",
    response_model=ProviderResponse,
    summary="Update AI provider",
)
async def update_provider(provider_id: str, request: ProviderUpdate):
    registry = get_provider_registry_service()
    try:
        provider = await registry.update_provider(
            provider_id=provider_id,
            display_name=request.display_name,
            base_url=str(request.base_url) if request.base_url else None,
            auth_method=request.auth_method,
            credential=request.credential,
            default_headers=request.default_headers,
            extra_config=request.extra_config,
            enabled=request.enabled,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return ProviderResponse.model_validate(provider)


@router.post(
    "/providers/{provider_id}/test",
    response_model=ProviderTestResponse,
    summary="Test AI provider connection",
)
async def test_provider_connection(provider_id: str):
    registry = get_provider_registry_service()
    try:
        result = await registry.test_provider_connection(provider_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return ProviderTestResponse(**result)


@router.post(
    "/providers/{provider_id}/smoke-test",
    response_model=ProviderSmokeTestResponse,
    summary="Run smoke test on AI provider with a specific model",
)
async def smoke_test_provider(provider_id: str, request: ProviderSmokeTestRequest):
    registry = get_provider_registry_service()
    try:
        result = await registry.smoke_test_provider(provider_id, request.model)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return ProviderSmokeTestResponse(**result)


@router.get(
    "/providers/{provider_id}/models",
    response_model=ModelCapabilityListResponse,
    summary="List models available from an AI provider",
)
async def list_provider_models(
    provider_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    settings: Settings = Depends(get_settings),
):
    page_size = min(page_size, settings.max_page_size)

    registry = get_provider_registry_service()
    try:
        models = await registry.list_provider_models(provider_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    # Apply pagination manually since we're getting a list
    total = len(models)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_models = models[start:end]
    total_pages = (total + page_size - 1) // page_size

    return ModelCapabilityListResponse(
        items=[ModelCapabilityResponse.model_validate(m) for m in paginated_models],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/models/{model_id}/capabilities",
    response_model=ModelCapabilityResponse,
    summary="Get model capabilities",
)
async def get_model_capabilities(model_id: str, provider_id: str = Query(...)):
    registry = get_provider_registry_service()
    try:
        capability = await registry.get_model_capabilities(provider_id, model_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return ModelCapabilityResponse.model_validate(capability)


@router.post(
    "/providers/{provider_id}/models",
    response_model=ModelCapabilityResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add or update model capability for a provider",
)
async def upsert_model_capability(provider_id: str, request: ModelCapabilityCreate):
    registry = get_provider_registry_service()
    try:
        capability = await registry.upsert_model_capability(
            provider_id=provider_id,
            model_identifier=request.model_identifier,
            display_name=request.display_name,
            capabilities=request.capabilities,
            max_context_tokens=request.max_context_tokens,
            max_output_tokens=request.max_output_tokens,
            supports_streaming=request.supports_streaming,
            supports_system_prompt=request.supports_system_prompt,
            supports_parallel_tool_calls=request.supports_parallel_tool_calls,
            input_modalities=request.input_modalities,
            output_modalities=request.output_modalities,
            pricing_input_per_1k=request.pricing_input_per_1k,
            pricing_output_per_1k=request.pricing_output_per_1k,
            is_deprecated=request.is_deprecated,
            metadata=request.metadata,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return ModelCapabilityResponse.model_validate(capability)


@router.patch(
    "/providers/{provider_id}/models/{model_identifier}",
    response_model=ModelCapabilityResponse,
    summary="Update model capability for a provider",
)
async def update_model_capability(
    provider_id: str,
    model_identifier: str,
    request: ModelCapabilityUpdate,
):
    registry = get_provider_registry_service()
    # Get existing capability first
    try:
        existing = await registry.get_model_capabilities(provider_id, model_identifier)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    # Prepare update data
    update_data = {
        "display_name": request.display_name if request.display_name is not None else existing.display_name,
        "capabilities": request.capabilities if request.capabilities is not None else existing.capabilities,
        "max_context_tokens": request.max_context_tokens if request.max_context_tokens is not None else existing.max_context_tokens,
        "max_output_tokens": request.max_output_tokens if request.max_output_tokens is not None else existing.max_output_tokens,
        "supports_streaming": request.supports_streaming if request.supports_streaming is not None else existing.supports_streaming,
        "supports_system_prompt": request.supports_system_prompt if request.supports_system_prompt is not None else existing.supports_system_prompt,
        "supports_parallel_tool_calls": request.supports_parallel_tool_calls if request.supports_parallel_tool_calls is not None else existing.supports_parallel_tool_calls,
        "input_modalities": request.input_modalities if request.input_modalities is not None else existing.input_modalities,
        "output_modalities": request.output_modalities if request.output_modalities is not None else existing.output_modalities,
        "pricing_input_per_1k": request.pricing_input_per_1k if request.pricing_input_per_1k is not None else existing.pricing_input_per_1k,
        "pricing_output_per_1k": request.pricing_output_per_1k if request.pricing_output_per_1k is not None else existing.pricing_output_per_1k,
        "is_deprecated": request.is_deprecated if request.is_deprecated is not None else existing.is_deprecated,
        "metadata": request.metadata if request.metadata is not None else existing.metadata,
    }

    try:
        capability = await registry.upsert_model_capability(
            provider_id=provider_id,
            model_identifier=model_identifier,
            **update_data,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return ModelCapabilityResponse.model_validate(capability)


@router.delete(
    "/providers/{provider_id}/models/{model_identifier}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete model capability for a provider",
)
async def delete_model_capability(provider_id: str, model_identifier: str):
    registry = get_provider_registry_service()
    try:
        await registry.delete_model_capability(provider_id, model_identifier)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.post(
    "/providers/{provider_id}/disable",
    response_model=ProviderResponse,
    summary="Disable AI provider",
)
async def disable_provider(provider_id: str):
    registry = get_provider_registry_service()
    try:
        provider = await registry.update_provider(provider_id, enabled=False)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return ProviderResponse.model_validate(provider)


@router.delete(
    "/providers/{provider_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete AI provider",
)
async def delete_provider(provider_id: str):
    registry = get_provider_registry_service()
    try:
        await registry.delete_provider(provider_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.post(
    "/models",
    response_model=ModelIntegrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new model integration",
)
async def create_model_integration(request: ModelIntegrationCreate):
    validate_model_integration_create(request)

    registry = get_integration_registry_service()
    try:
        integration = await registry.create_model_integration(
            name=request.name,
            display_name=request.display_name,
            provider_category=request.provider_category,
            model_identifier=request.model_identifier,
            adapter_type=request.adapter_type,
            base_url=str(request.base_url),
            auth_method=request.auth_method,
            declared_capabilities=request.declared_capabilities,
            credential=request.credential,
            extra=request.extra,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return ModelIntegrationResponse.model_validate(integration)


@router.get(
    "/models",
    response_model=ModelIntegrationListResponse,
    summary="List model integrations",
)
async def list_model_integrations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    provider_category: Optional[str] = Query(None),
    enabled: Optional[bool] = Query(None),
    settings: Settings = Depends(get_settings),
):
    page_size = min(page_size, settings.max_page_size)

    registry = get_integration_registry_service()
    integrations, total = await registry.list_model_integrations(
        page=page,
        page_size=page_size,
        status=status_filter,
        provider_category=provider_category,
        enabled=enabled,
    )

    total_pages = (total + page_size - 1) // page_size

    return ModelIntegrationListResponse(
        items=[ModelIntegrationResponse.model_validate(i) for i in integrations],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/models/{integration_id}",
    response_model=ModelIntegrationResponse,
    summary="Get model integration details",
)
async def get_model_integration(integration_id: str):
    registry = get_integration_registry_service()
    try:
        integration = await registry.get_model_integration(integration_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return ModelIntegrationResponse.model_validate(integration)


@router.patch(
    "/models/{integration_id}",
    response_model=ModelIntegrationResponse,
    summary="Update model integration",
)
async def update_model_integration(integration_id: str, request: ModelIntegrationUpdate):
    registry = get_integration_registry_service()
    try:
        integration = await registry.update_model_integration(
            integration_id=integration_id,
            display_name=request.display_name,
            model_identifier=request.model_identifier,
            base_url=str(request.base_url) if request.base_url else None,
            auth_method=request.auth_method,
            declared_capabilities=request.declared_capabilities,
            credential=request.credential,
            enabled=request.enabled,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return ModelIntegrationResponse.model_validate(integration)


@router.post(
    "/models/{integration_id}/verify",
    response_model=VerificationResponse,
    summary="Verify model integration connectivity",
)
async def verify_model_integration(integration_id: str):
    verification = get_verification_service()
    try:
        integration = await verification.verify_model_integration(integration_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return VerificationResponse(
        success=integration.status in (ModelIntegrationStatus.CONNECTED, ModelIntegrationStatus.VERIFIED),
        status=integration.status.value,
        diagnostic=integration.verification_diagnostic or "",
        capabilities=integration.verified_capabilities,
    )


@router.post(
    "/models/{integration_id}/capabilities/check",
    response_model=CapabilityCheckResponse,
    summary="Check a specific model capability",
)
async def check_model_capability(integration_id: str, request: CapabilityCheckRequest):
    verification = get_verification_service()
    try:
        result = await verification.check_model_capability(integration_id, request.capability)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return CapabilityCheckResponse(**result)


@router.post(
    "/models/{integration_id}/capabilities/check-all",
    response_model=List[CapabilityCheckResponse],
    summary="Check all declared model capabilities",
)
async def check_all_model_capabilities(integration_id: str):
    verification = get_verification_service()
    try:
        results = await verification.check_all_model_capabilities(integration_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return [CapabilityCheckResponse(**r) for r in results]


@router.post(
    "/models/{integration_id}/disable",
    response_model=ModelIntegrationResponse,
    summary="Disable model integration",
)
async def disable_model_integration(integration_id: str):
    registry = get_integration_registry_service()
    try:
        integration = await registry.update_model_integration(integration_id, enabled=False)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return ModelIntegrationResponse.model_validate(integration)


@router.delete(
    "/models/{integration_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete model integration",
)
async def delete_model_integration(integration_id: str):
    registry = get_integration_registry_service()
    try:
        await registry.delete_model_integration(integration_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.post(
    "/agents",
    response_model=AgentIntegrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new agent integration",
)
async def create_agent_integration(request: AgentIntegrationCreate):
    validate_agent_integration_create(request)

    registry = get_integration_registry_service()
    try:
        integration = await registry.create_agent_integration(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            framework=request.framework,
            protocol=request.protocol,
            adapter_type=request.adapter_type,
            endpoint=str(request.endpoint),
            auth_method=request.auth_method,
            declared_capabilities=request.declared_capabilities,
            intended_operation=request.intended_operation,
            credential=request.credential,
        )
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return AgentIntegrationResponse.model_validate(integration)


@router.get(
    "/agents",
    response_model=AgentIntegrationListResponse,
    summary="List agent integrations",
)
async def list_agent_integrations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    framework: Optional[str] = Query(None),
    enabled: Optional[bool] = Query(None),
    settings: Settings = Depends(get_settings),
):
    page_size = min(page_size, settings.max_page_size)

    registry = get_integration_registry_service()
    integrations, total = await registry.list_agent_integrations(
        page=page,
        page_size=page_size,
        status=status_filter,
        framework=framework,
        enabled=enabled,
    )

    total_pages = (total + page_size - 1) // page_size

    return AgentIntegrationListResponse(
        items=[AgentIntegrationResponse.model_validate(i) for i in integrations],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/agents/{integration_id}",
    response_model=AgentIntegrationResponse,
    summary="Get agent integration details",
)
async def get_agent_integration(integration_id: str):
    registry = get_integration_registry_service()
    try:
        integration = await registry.get_agent_integration(integration_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return AgentIntegrationResponse.model_validate(integration)


@router.patch(
    "/agents/{integration_id}",
    response_model=AgentIntegrationResponse,
    summary="Update agent integration",
)
async def update_agent_integration(integration_id: str, request: AgentIntegrationUpdate):
    registry = get_integration_registry_service()
    try:
        integration = await registry.update_agent_integration(
            integration_id=integration_id,
            display_name=request.display_name,
            description=request.description,
            endpoint=str(request.endpoint) if request.endpoint else None,
            auth_method=request.auth_method,
            declared_capabilities=request.declared_capabilities,
            credential=request.credential,
            enabled=request.enabled,
            intended_operation=request.intended_operation,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return AgentIntegrationResponse.model_validate(integration)


@router.post(
    "/agents/{integration_id}/verify",
    response_model=VerificationResponse,
    summary="Verify agent integration connectivity",
)
async def verify_agent_integration(integration_id: str):
    verification = get_verification_service()
    try:
        integration = await verification.verify_agent_integration(integration_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return VerificationResponse(
        success=integration.status in (AgentStatus.CONNECTED, AgentStatus.VERIFIED),
        status=integration.status.value,
        diagnostic=integration.verification_diagnostic or "",
        capabilities=integration.verified_capabilities,
    )


@router.post(
    "/agents/{integration_id}/capabilities/check",
    response_model=CapabilityCheckResponse,
    summary="Check a specific agent capability",
)
async def check_agent_capability(integration_id: str, request: CapabilityCheckRequest):
    verification = get_verification_service()
    try:
        result = await verification.check_agent_capability(integration_id, request.capability)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return CapabilityCheckResponse(**result)


@router.post(
    "/agents/{integration_id}/capabilities/check-all",
    response_model=List[CapabilityCheckResponse],
    summary="Check all declared agent capabilities",
)
async def check_all_agent_capabilities(integration_id: str):
    verification = get_verification_service()
    try:
        results = await verification.check_all_agent_capabilities(integration_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return [CapabilityCheckResponse(**r) for r in results]


@router.post(
    "/agents/{integration_id}/disable",
    response_model=AgentIntegrationResponse,
    summary="Disable agent integration",
)
async def disable_agent_integration(integration_id: str):
    registry = get_integration_registry_service()
    try:
        integration = await registry.update_agent_integration(integration_id, enabled=False)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return AgentIntegrationResponse.model_validate(integration)


@router.delete(
    "/agents/{integration_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete agent integration",
)
async def delete_agent_integration(integration_id: str):
    registry = get_integration_registry_service()
    try:
        await registry.delete_agent_integration(integration_id)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.post(
    "/access-grants",
    response_model=DatasetAccessGrantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Grant dataset access to integration",
)
async def grant_dataset_access(request: DatasetAccessGrantCreate):
    registry = get_integration_registry_service()
    try:
        grant = await registry.grant_dataset_access(
            integration_type=request.integration_type,
            integration_id=request.integration_id,
            dataset_id=request.dataset_id,
            dataset_version_id=request.dataset_version_id,
            operation=request.operation,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return DatasetAccessGrantResponse.model_validate(grant)


@router.get(
    "/access-grants",
    response_model=List[DatasetAccessGrantResponse],
    summary="List dataset access grants",
)
async def list_dataset_access_grants(
    integration_type: Optional[str] = Query(None),
    integration_id: Optional[str] = Query(None),
    dataset_id: Optional[str] = Query(None),
):
    registry = get_integration_registry_service()
    grants = await registry.list_dataset_access_grants(
        integration_type=integration_type,
        integration_id=integration_id,
        dataset_id=dataset_id,
    )
    return [DatasetAccessGrantResponse.model_validate(g) for g in grants]


@router.delete(
    "/access-grants",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke dataset access grant",
)
async def revoke_dataset_access(
    integration_type: str = Query(...),
    integration_id: str = Query(...),
    dataset_id: str = Query(...),
    dataset_version_id: Optional[str] = Query(None),
    operation: str = Query(...),
):
    registry = get_integration_registry_service()
    try:
        await registry.revoke_dataset_access(
            integration_type=integration_type,
            integration_id=integration_id,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            operation=operation,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))