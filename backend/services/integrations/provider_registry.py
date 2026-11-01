from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

from sqlalchemy import select, func, delete, or_

from core.config import get_settings
from core.database.config import get_db_session
from core.database.models.integration import (
    Provider,
    ModelCapabilityInfo,
    ProviderType,
    ProviderStatus,
    AuthenticationMethod,
    ModelCapability,
)
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from services.integrations.secrets import get_secret_store
from services.integrations.security import get_ssrf_protector
from services.integrations.adapters import (
    AdapterConfig,
    AdapterRegistry,
    AdapterError,
)

logger = get_logger(__name__)


class ProviderRegistryService:
    def __init__(self):
        self.settings = get_settings()
        self.secret_store = get_secret_store()
        self.ssrf_protector = get_ssrf_protector()

    def _validate_provider_type(self, provider_type: str) -> ProviderType:
        try:
            return ProviderType(provider_type)
        except ValueError:
            raise ValidationError(f"Invalid provider type: {provider_type}")

    def _validate_auth_method(self, auth_method: str) -> AuthenticationMethod:
        try:
            return AuthenticationMethod(auth_method)
        except ValueError:
            raise ValidationError(f"Invalid auth method: {auth_method}")

    async def create_provider(
        self,
        name: str,
        display_name: str,
        provider_type: str,
        base_url: str,
        auth_method: str,
        credential: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        extra_config: Optional[Dict[str, str]] = None,
    ) -> Provider:
        provider_type_enum = self._validate_provider_type(provider_type)
        auth_method_enum = self._validate_auth_method(auth_method)

        if auth_method != "none" and not credential:
            raise ValidationError("Credential required for non-none auth methods")

        self.ssrf_protector.validate_url(base_url)

        credential_ref = None
        if credential:
            credential_ref = self.secret_store.generate_ref()
            self.secret_store.store(credential_ref, credential)

        async for session in get_db_session():
            existing = await session.execute(
                select(Provider).where(Provider.name == name)
            )
            if existing.scalar_one_or_none():
                raise ValidationError(f"Provider with name '{name}' already exists")

            provider = Provider(
                name=name,
                display_name=display_name,
                provider_type=provider_type_enum,
                base_url=base_url,
                auth_method=auth_method_enum,
                credential_ref=credential_ref,
                default_headers=default_headers or {},
                extra_config=extra_config or {},
                status=ProviderStatus.INACTIVE,
            )
            session.add(provider)
            await session.commit()
            await session.refresh(provider)

            logger.info("provider_created", provider_id=provider.id, name=name, type=provider_type)
            return provider

    async def update_provider(
        self,
        provider_id: str,
        display_name: Optional[str] = None,
        base_url: Optional[str] = None,
        auth_method: Optional[str] = None,
        credential: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        extra_config: Optional[Dict[str, str]] = None,
        enabled: Optional[bool] = None,
    ) -> Provider:
        async for session in get_db_session():
            provider = await session.get(Provider, provider_id)
            if not provider:
                raise NotFoundError(f"Provider not found: {provider_id}")

            if display_name is not None:
                provider.display_name = display_name
            if base_url is not None:
                self.ssrf_protector.validate_url(base_url)
                provider.base_url = base_url
            if auth_method is not None:
                provider.auth_method = self._validate_auth_method(auth_method)
            if credential is not None:
                if provider.credential_ref:
                    self.secret_store.store(provider.credential_ref, credential)
                else:
                    provider.credential_ref = self.secret_store.generate_ref()
                    self.secret_store.store(provider.credential_ref, credential)
            if default_headers is not None:
                provider.default_headers = default_headers
            if extra_config is not None:
                provider.extra_config = extra_config
            if enabled is not None:
                provider.enabled = enabled
                if enabled:
                    provider.status = ProviderStatus.INACTIVE
                else:
                    provider.status = ProviderStatus.INACTIVE

            provider.updated_at = datetime.utcnow()
            await session.commit()
            await session.refresh(provider)

            logger.info("provider_updated", provider_id=provider_id)
            return provider

    async def get_provider(self, provider_id: str) -> Provider:
        async for session in get_db_session():
            provider = await session.get(Provider, provider_id)
            if not provider:
                raise NotFoundError(f"Provider not found: {provider_id}")
            return provider

    async def list_providers(
        self,
        page: int = 1,
        page_size: int = 20,
        provider_type: Optional[str] = None,
        status: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Tuple[List[Provider], int]:
        async for session in get_db_session():
            query = select(Provider)

            if provider_type:
                query = query.where(Provider.provider_type == provider_type)
            if status:
                query = query.where(Provider.status == status)
            if enabled is not None:
                query = query.where(Provider.enabled == enabled)

            total_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(total_query)
            total = total_result.scalar() or 0

            query = query.order_by(Provider.created_at.desc())
            query = query.offset((page - 1) * page_size).limit(page_size)
            result = await session.execute(query)
            providers = list(result.scalars().all())

            return providers, total

    async def delete_provider(self, provider_id: str) -> bool:
        async for session in get_db_session():
            provider = await session.get(Provider, provider_id)
            if not provider:
                raise NotFoundError(f"Provider not found: {provider_id}")

            if provider.credential_ref:
                self.secret_store.delete(provider.credential_ref)

            await session.execute(
                delete(ModelCapabilityInfo).where(
                    ModelCapabilityInfo.provider_id == provider_id,
                )
            )

            await session.delete(provider)
            await session.commit()

            logger.info("provider_deleted", provider_id=provider_id)
            return True

    def _create_adapter_config(self, provider: Provider) -> AdapterConfig:
        credential = None
        if provider.credential_ref:
            credential = self.secret_store.retrieve(provider.credential_ref)

        return AdapterConfig(
            base_url=provider.base_url,
            model_identifier="",
            auth_method=provider.auth_method.value,
            credential=credential,
            extra={
                "default_headers": provider.default_headers,
                **provider.extra_config,
            },
        )

    def _get_adapter_class(self, provider_type: ProviderType):
        adapter_type_map = {
            ProviderType.OPENAI: "openai",
            ProviderType.ANTHROPIC: "anthropic",
            ProviderType.GOOGLE_GEMINI: "google_gemini",
            ProviderType.MISTRAL: "mistral",
            ProviderType.OLLAMA: "ollama",
            ProviderType.OPENAI_COMPATIBLE: "openai_compatible",
            ProviderType.CUSTOM: "custom_rest",
        }
        adapter_type = adapter_type_map.get(provider_type)
        if not adapter_type:
            raise ValidationError(f"No adapter mapping for provider type: {provider_type}")
        try:
            return AdapterRegistry.get(adapter_type)
        except AdapterError as e:
            raise ValidationError(f"Unknown adapter type: {adapter_type}")

    async def test_provider_connection(self, provider_id: str) -> Dict[str, Any]:
        async for session in get_db_session():
            provider = await session.get(Provider, provider_id)
            if not provider:
                raise NotFoundError(f"Provider not found: {provider_id}")

            provider.status = ProviderStatus.TESTING
            await session.commit()

            try:
                self.ssrf_protector.validate_url(provider.base_url)

                adapter_class = self._get_adapter_class(provider.provider_type)
                config = self._create_adapter_config(provider)
                adapter = adapter_class(
                    config,
                    timeout_seconds=self.settings.integration_timeout_seconds,
                    max_response_size=self.settings.integration_max_response_size,
                )

                result = await adapter.verify_connection()

                provider.status = ProviderStatus.ACTIVE if result.success else ProviderStatus.ERROR
                provider.last_test_at = datetime.utcnow()
                provider.last_test_result = result.diagnostic

                await session.commit()
                await session.refresh(provider)

                logger.info(
                    "provider_connection_tested",
                    provider_id=provider_id,
                    success=result.success,
                    status=provider.status.value,
                )
                return {
                    "success": result.success,
                    "status": provider.status.value,
                    "diagnostic": result.diagnostic,
                    "capabilities": result.capabilities,
                }

            except Exception as e:
                provider.status = ProviderStatus.ERROR
                provider.last_test_at = datetime.utcnow()
                provider.last_test_result = str(e)
                await session.commit()
                logger.error("provider_connection_test_failed", provider_id=provider_id, error=str(e))
                raise ValidationError(f"Connection test failed: {str(e)}")

    async def smoke_test_provider(self, provider_id: str, model: Optional[str] = None) -> Dict[str, Any]:
        async for session in get_db_session():
            provider = await session.get(Provider, provider_id)
            if not provider:
                raise NotFoundError(f"Provider not found: {provider_id}")

            provider.status = ProviderStatus.TESTING
            await session.commit()

            try:
                self.ssrf_protector.validate_url(provider.base_url)

                adapter_class = self._get_adapter_class(provider.provider_type)
                config = self._create_adapter_config(provider)
                adapter = adapter_class(
                    config,
                    timeout_seconds=self.settings.integration_timeout_seconds,
                    max_response_size=self.settings.integration_max_response_size,
                )

                test_model = model or "gpt-3.5-turbo"
                if hasattr(config, 'model_identifier'):
                    config.model_identifier = test_model

                result = await adapter.verify_connection()

                provider.status = ProviderStatus.ACTIVE if result.success else ProviderStatus.ERROR
                provider.last_test_at = datetime.utcnow()
                provider.last_test_result = result.diagnostic

                await session.commit()
                await session.refresh(provider)

                logger.info(
                    "provider_smoke_tested",
                    provider_id=provider_id,
                    model=test_model,
                    success=result.success,
                )
                return {
                    "success": result.success,
                    "model_tested": test_model,
                    "status": provider.status.value,
                    "diagnostic": result.diagnostic,
                    "capabilities": result.capabilities,
                }

            except Exception as e:
                provider.status = ProviderStatus.ERROR
                provider.last_test_at = datetime.utcnow()
                provider.last_test_result = str(e)
                await session.commit()
                logger.error("provider_smoke_test_failed", provider_id=provider_id, error=str(e))
                raise ValidationError(f"Smoke test failed: {str(e)}")

    async def list_provider_models(self, provider_id: str) -> List[ModelCapabilityInfo]:
        async for session in get_db_session():
            provider = await session.get(Provider, provider_id)
            if not provider:
                raise NotFoundError(f"Provider not found: {provider_id}")

            query = select(ModelCapabilityInfo).where(
                ModelCapabilityInfo.provider_id == provider_id,
                ModelCapabilityInfo.is_deprecated == False,
            ).order_by(ModelCapabilityInfo.model_identifier)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_model_capabilities(self, provider_id: str, model_identifier: str) -> ModelCapabilityInfo:
        async for session in get_db_session():
            query = select(ModelCapabilityInfo).where(
                ModelCapabilityInfo.provider_id == provider_id,
                ModelCapabilityInfo.model_identifier == model_identifier,
            )
            result = await session.execute(query)
            capability = result.scalar_one_or_none()
            if not capability:
                raise NotFoundError(f"Model capability not found: {model_identifier} for provider {provider_id}")
            return capability

    async def upsert_model_capability(
        self,
        provider_id: str,
        model_identifier: str,
        display_name: str,
        capabilities: List[str],
        max_context_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        supports_streaming: bool = False,
        supports_system_prompt: bool = True,
        supports_parallel_tool_calls: bool = False,
        input_modalities: Optional[List[str]] = None,
        output_modalities: Optional[List[str]] = None,
        pricing_input_per_1k: Optional[float] = None,
        pricing_output_per_1k: Optional[float] = None,
        is_deprecated: bool = False,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ModelCapabilityInfo:
        async for session in get_db_session():
            provider = await session.get(Provider, provider_id)
            if not provider:
                raise NotFoundError(f"Provider not found: {provider_id}")

            query = select(ModelCapabilityInfo).where(
                ModelCapabilityInfo.provider_id == provider_id,
                ModelCapabilityInfo.model_identifier == model_identifier,
            )
            result = await session.execute(query)
            capability = result.scalar_one_or_none()

            if capability:
                capability.display_name = display_name
                capability.capabilities = capabilities
                capability.max_context_tokens = max_context_tokens
                capability.max_output_tokens = max_output_tokens
                capability.supports_streaming = supports_streaming
                capability.supports_system_prompt = supports_system_prompt
                capability.supports_parallel_tool_calls = supports_parallel_tool_calls
                capability.input_modalities = input_modalities or []
                capability.output_modalities = output_modalities or []
                capability.pricing_input_per_1k = pricing_input_per_1k
                capability.pricing_output_per_1k = pricing_output_per_1k
                capability.is_deprecated = is_deprecated
                capability.model_metadata = metadata or {}
                capability.updated_at = datetime.utcnow()
            else:
                capability = ModelCapabilityInfo(
                    provider_id=provider_id,
                    model_identifier=model_identifier,
                    display_name=display_name,
                    capabilities=capabilities,
                    max_context_tokens=max_context_tokens,
                    max_output_tokens=max_output_tokens,
                    supports_streaming=supports_streaming,
                    supports_system_prompt=supports_system_prompt,
                    supports_parallel_tool_calls=supports_parallel_tool_calls,
                    input_modalities=input_modalities or [],
                    output_modalities=output_modalities or [],
                    pricing_input_per_1k=pricing_input_per_1k,
                    pricing_output_per_1k=pricing_output_per_1k,
                    is_deprecated=is_deprecated,
                    metadata=metadata or {},
                )
                session.add(capability)

            await session.commit()
            await session.refresh(capability)

            logger.info("model_capability_upserted", provider_id=provider_id, model=model_identifier)
            return capability

    async def delete_model_capability(self, provider_id: str, model_identifier: str) -> bool:
        async for session in get_db_session():
            query = select(ModelCapabilityInfo).where(
                ModelCapabilityInfo.provider_id == provider_id,
                ModelCapabilityInfo.model_identifier == model_identifier,
            )
            result = await session.execute(query)
            capability = result.scalar_one_or_none()
            if not capability:
                raise NotFoundError(f"Model capability not found: {model_identifier} for provider {provider_id}")

            await session.delete(capability)
            await session.commit()

            logger.info("model_capability_deleted", provider_id=provider_id, model=model_identifier)
            return True

    async def discover_provider_models(self, provider_id: str) -> List[Dict[str, Any]]:
        async for session in get_db_session():
            provider = await session.get(Provider, provider_id)
            if not provider:
                raise NotFoundError(f"Provider not found: {provider_id}")

            adapter_class = self._get_adapter_class(provider.provider_type)
            config = self._create_adapter_config(provider)
            adapter = adapter_class(
                config,
                timeout_seconds=self.settings.integration_timeout_seconds,
                max_response_size=self.settings.integration_max_response_size,
            )

            if hasattr(adapter, 'list_models'):
                models = await adapter.list_models()
                return models
            else:
                return []


def get_provider_registry_service() -> ProviderRegistryService:
    return ProviderRegistryService()