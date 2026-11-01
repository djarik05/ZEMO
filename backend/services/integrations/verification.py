from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import select

from core.config import get_settings
from core.database.config import get_db_session
from core.database.models.integration import (
    ModelIntegration,
    AgentIntegration,
    ModelIntegrationStatus,
    AgentStatus,
    CapabilityStatus,
)
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from services.integrations.adapters import (
    AdapterConfig,
    AdapterRegistry,
    AdapterError,
)
from services.integrations.registry import get_integration_registry_service
from services.integrations.secrets import get_secret_store
from services.integrations.security import get_ssrf_protector

logger = get_logger(__name__)


class VerificationService:
    def __init__(self):
        self.settings = get_settings()
        self.registry = get_integration_registry_service()
        self.secret_store = get_secret_store()
        self.ssrf_protector = get_ssrf_protector()

    def _create_adapter_config(self, integration) -> AdapterConfig:
        credential = None
        if integration.credential_ref:
            credential = self.secret_store.retrieve(integration.credential_ref)

        extra = {}
        if hasattr(integration, 'framework'):
            extra["health_path"] = "/health"
        elif hasattr(integration, 'intended_operation'):
            extra["health_path"] = "/health"

        return AdapterConfig(
            base_url=integration.base_url if hasattr(integration, 'base_url') else integration.endpoint,
            model_identifier=integration.model_identifier if hasattr(integration, 'model_identifier') else "",
            auth_method=integration.auth_method.value,
            credential=credential,
            extra=extra,
        )

    def _get_adapter_class(self, adapter_type: str):
        try:
            return AdapterRegistry.get(adapter_type)
        except AdapterError as e:
            raise ValidationError(f"Unknown adapter type: {adapter_type}")

    async def verify_model_integration(self, integration_id: str) -> ModelIntegration:
        async for session in get_db_session():
            integration = await session.get(ModelIntegration, integration_id)
            if not integration:
                raise NotFoundError(f"Model integration not found: {integration_id}")

            integration.status = ModelIntegrationStatus.CONNECTING
            await session.commit()

            try:
                self.ssrf_protector.validate_url(integration.base_url)

                adapter_class = self._get_adapter_class(integration.adapter_type)
                config = self._create_adapter_config(integration)
                adapter = adapter_class(
                    config,
                    timeout_seconds=self.settings.integration_timeout_seconds,
                    max_response_size=self.settings.integration_max_response_size,
                )

                result = await adapter.verify_connection()

                integration.status = ModelIntegrationStatus.CONNECTED if result.success else ModelIntegrationStatus.FAILED
                integration.verification_diagnostic = result.diagnostic
                integration.last_verification_at = datetime.utcnow()

                if result.success:
                    integration.verified_capabilities = {}
                    for cap in integration.declared_capabilities:
                        if cap in result.capabilities:
                            integration.verified_capabilities[cap] = result.capabilities[cap]
                        else:
                            integration.verified_capabilities[cap] = CapabilityStatus.UNKNOWN.value

                    if all(v == CapabilityStatus.VERIFIED.value for v in integration.verified_capabilities.values()):
                        integration.status = ModelIntegrationStatus.VERIFIED

                await session.commit()
                await session.refresh(integration)

                logger.info(
                    "model_integration_verified",
                    integration_id=integration_id,
                    success=result.success,
                    status=integration.status.value,
                )
                return integration

            except Exception as e:
                integration.status = ModelIntegrationStatus.FAILED
                integration.verification_diagnostic = str(e)
                integration.last_verification_at = datetime.utcnow()
                await session.commit()
                logger.error("model_integration_verification_failed", integration_id=integration_id, error=str(e))
                raise ValidationError(f"Verification failed: {str(e)}")

    async def verify_agent_integration(self, integration_id: str) -> AgentIntegration:
        async for session in get_db_session():
            integration = await session.get(AgentIntegration, integration_id)
            if not integration:
                raise NotFoundError(f"Agent integration not found: {integration_id}")

            integration.status = AgentStatus.CONNECTING
            await session.commit()

            try:
                self.ssrf_protector.validate_url(integration.endpoint)

                adapter_class = self._get_adapter_class(integration.adapter_type)
                config = self._create_adapter_config(integration)
                adapter = adapter_class(
                    config,
                    timeout_seconds=self.settings.integration_timeout_seconds,
                    max_response_size=self.settings.integration_max_response_size,
                )

                result = await adapter.verify_connection()

                integration.status = AgentStatus.CONNECTED if result.success else AgentStatus.FAILED
                integration.verification_diagnostic = result.diagnostic
                integration.last_verification_at = datetime.utcnow()

                if result.success:
                    integration.verified_capabilities = {}
                    for cap in integration.declared_capabilities:
                        if cap in result.capabilities:
                            integration.verified_capabilities[cap] = result.capabilities[cap]
                        else:
                            integration.verified_capabilities[cap] = CapabilityStatus.UNKNOWN.value

                    if all(v == CapabilityStatus.VERIFIED.value for v in integration.verified_capabilities.values()):
                        integration.status = AgentStatus.VERIFIED

                await session.commit()
                await session.refresh(integration)

                logger.info(
                    "agent_integration_verified",
                    integration_id=integration_id,
                    success=result.success,
                    status=integration.status.value,
                )
                return integration

            except Exception as e:
                integration.status = AgentStatus.FAILED
                integration.verification_diagnostic = str(e)
                integration.last_verification_at = datetime.utcnow()
                await session.commit()
                logger.error("agent_integration_verification_failed", integration_id=integration_id, error=str(e))
                raise ValidationError(f"Verification failed: {str(e)}")

    async def check_model_capability(
        self,
        integration_id: str,
        capability: str,
    ) -> Dict[str, Any]:
        async for session in get_db_session():
            integration = await session.get(ModelIntegration, integration_id)
            if not integration:
                raise NotFoundError(f"Model integration not found: {integration_id}")

            adapter_class = self._get_adapter_class(integration.adapter_type)
            config = self._create_adapter_config(integration)
            adapter = adapter_class(
                config,
                timeout_seconds=self.settings.integration_timeout_seconds,
                max_response_size=self.settings.integration_max_response_size,
            )

            result = await adapter.check_capability(capability)

            if result.verified:
                integration.verified_capabilities[capability] = CapabilityStatus.VERIFIED.value
                await session.commit()

            return {
                "capability": result.capability,
                "status": result.status,
                "verified": result.verified,
                "diagnostic": result.diagnostic,
                "metadata": result.metadata,
            }

    async def check_agent_capability(
        self,
        integration_id: str,
        capability: str,
    ) -> Dict[str, Any]:
        async for session in get_db_session():
            integration = await session.get(AgentIntegration, integration_id)
            if not integration:
                raise NotFoundError(f"Agent integration not found: {integration_id}")

            adapter_class = self._get_adapter_class(integration.adapter_type)
            config = self._create_adapter_config(integration)
            adapter = adapter_class(
                config,
                timeout_seconds=self.settings.integration_timeout_seconds,
                max_response_size=self.settings.integration_max_response_size,
            )

            result = await adapter.check_capability(capability)

            if result.verified:
                integration.verified_capabilities[capability] = CapabilityStatus.VERIFIED.value
                await session.commit()

            return {
                "capability": result.capability,
                "status": result.status,
                "verified": result.verified,
                "diagnostic": result.diagnostic,
                "metadata": result.metadata,
            }

    async def check_all_model_capabilities(self, integration_id: str) -> List[Dict[str, Any]]:
        integration = await self.registry.get_model_integration(integration_id)
        results = []

        for capability in integration.declared_capabilities:
            result = await self.check_model_capability(integration_id, capability)
            results.append(result)

        return results

    async def check_all_agent_capabilities(self, integration_id: str) -> List[Dict[str, Any]]:
        integration = await self.registry.get_agent_integration(integration_id)
        results = []

        for capability in integration.declared_capabilities:
            result = await self.check_agent_capability(integration_id, capability)
            results.append(result)

        return results


def get_verification_service() -> VerificationService:
    return VerificationService()