from datetime import datetime
from typing import Optional, List, Tuple

from sqlalchemy import select, func, delete, or_

from core.config import get_settings
from core.database.config import get_db_session
from core.database.models.integration import (
    ModelIntegration,
    AgentIntegration,
    DatasetAccessGrant,
    ProviderCategory,
    ModelIntegrationStatus,
    AgentStatus,
    AgentFramework,
    AuthenticationMethod,
    CapabilityStatus,
)
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from services.integrations.secrets import get_secret_store

logger = get_logger(__name__)


class IntegrationRegistryService:
    def __init__(self):
        self.settings = get_settings()
        self.secret_store = get_secret_store()

    async def create_model_integration(
        self,
        name: str,
        display_name: str,
        provider_category: str,
        model_identifier: str,
        adapter_type: str,
        base_url: str,
        auth_method: str,
        declared_capabilities: list[str],
        credential: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> ModelIntegration:
        from services.integrations.adapters import AdapterRegistry
        AdapterRegistry.get(adapter_type)

        try:
            ProviderCategory(provider_category)
        except ValueError:
            raise ValidationError(f"Invalid provider category: {provider_category}")

        try:
            AuthenticationMethod(auth_method)
        except ValueError:
            raise ValidationError(f"Invalid auth method: {auth_method}")

        if auth_method != "none" and not credential:
            raise ValidationError("Credential required for non-none auth methods")

        credential_ref = None
        if credential:
            credential_ref = self.secret_store.generate_ref()
            self.secret_store.store(credential_ref, credential)

        async for session in get_db_session():
            existing = await session.execute(
                select(ModelIntegration).where(ModelIntegration.name == name)
            )
            if existing.scalar_one_or_none():
                raise ValidationError(f"Model integration with name '{name}' already exists")

            integration = ModelIntegration(
                name=name,
                display_name=display_name,
                provider_category=ProviderCategory(provider_category),
                model_identifier=model_identifier,
                adapter_type=adapter_type,
                base_url=base_url,
                auth_method=AuthenticationMethod(auth_method),
                declared_capabilities=declared_capabilities,
                verified_capabilities={cap: CapabilityStatus.DECLARED.value for cap in declared_capabilities},
                status=ModelIntegrationStatus.REGISTERED,
                credential_ref=credential_ref,
            )
            session.add(integration)
            await session.commit()
            await session.refresh(integration)

            logger.info("model_integration_created", integration_id=integration.id, name=name)
            return integration

    async def update_model_integration(
        self,
        integration_id: str,
        display_name: Optional[str] = None,
        model_identifier: Optional[str] = None,
        base_url: Optional[str] = None,
        auth_method: Optional[str] = None,
        declared_capabilities: Optional[list[str]] = None,
        credential: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> ModelIntegration:
        async for session in get_db_session():
            integration = await session.get(ModelIntegration, integration_id)
            if not integration:
                raise NotFoundError(f"Model integration not found: {integration_id}")

            if display_name is not None:
                integration.display_name = display_name
            if model_identifier is not None:
                integration.model_identifier = model_identifier
            if base_url is not None:
                integration.base_url = base_url
            if auth_method is not None:
                try:
                    integration.auth_method = AuthenticationMethod(auth_method)
                except ValueError:
                    raise ValidationError(f"Invalid auth method: {auth_method}")
            if declared_capabilities is not None:
                integration.declared_capabilities = declared_capabilities
                integration.verified_capabilities = {
                    cap: CapabilityStatus.DECLARED.value for cap in declared_capabilities
                }
            if credential is not None:
                if integration.credential_ref:
                    self.secret_store.store(integration.credential_ref, credential)
                else:
                    integration.credential_ref = self.secret_store.generate_ref()
                    self.secret_store.store(integration.credential_ref, credential)
            if enabled is not None:
                integration.enabled = enabled
                if enabled:
                    integration.status = ModelIntegrationStatus.REGISTERED
                else:
                    integration.status = ModelIntegrationStatus.DISABLED

            integration.updated_at = datetime.utcnow()
            await session.commit()
            await session.refresh(integration)

            logger.info("model_integration_updated", integration_id=integration_id)
            return integration

    async def get_model_integration(self, integration_id: str) -> ModelIntegration:
        async for session in get_db_session():
            integration = await session.get(ModelIntegration, integration_id)
            if not integration:
                raise NotFoundError(f"Model integration not found: {integration_id}")
            return integration

    async def list_model_integrations(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        provider_category: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Tuple[List[ModelIntegration], int]:
        async for session in get_db_session():
            query = select(ModelIntegration)

            if status:
                query = query.where(ModelIntegration.status == status)
            if provider_category:
                query = query.where(ModelIntegration.provider_category == provider_category)
            if enabled is not None:
                query = query.where(ModelIntegration.enabled == enabled)

            total_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(total_query)
            total = total_result.scalar() or 0

            query = query.order_by(ModelIntegration.created_at.desc())
            query = query.offset((page - 1) * page_size).limit(page_size)
            result = await session.execute(query)
            integrations = list(result.scalars().all())

            return integrations, total

    async def delete_model_integration(self, integration_id: str) -> bool:
        async for session in get_db_session():
            integration = await session.get(ModelIntegration, integration_id)
            if not integration:
                raise NotFoundError(f"Model integration not found: {integration_id}")

            if integration.credential_ref:
                self.secret_store.delete(integration.credential_ref)

            await session.execute(
                delete(DatasetAccessGrant).where(
                    DatasetAccessGrant.integration_type == "model",
                    DatasetAccessGrant.integration_id == integration_id,
                )
            )

            await session.delete(integration)
            await session.commit()

            logger.info("model_integration_deleted", integration_id=integration_id)
            return True

    async def create_agent_integration(
        self,
        name: str,
        display_name: str,
        description: Optional[str],
        framework: str,
        protocol: str,
        adapter_type: str,
        endpoint: str,
        auth_method: str,
        declared_capabilities: list[str],
        intended_operation: str,
        credential: Optional[str] = None,
    ) -> AgentIntegration:
        from services.integrations.adapters import AdapterRegistry
        AdapterRegistry.get(adapter_type)

        try:
            AgentFramework(framework)
        except ValueError:
            raise ValidationError(f"Invalid agent framework: {framework}")

        try:
            AuthenticationMethod(auth_method)
        except ValueError:
            raise ValidationError(f"Invalid auth method: {auth_method}")

        if auth_method != "none" and not credential:
            raise ValidationError("Credential required for non-none auth methods")

        credential_ref = None
        if credential:
            credential_ref = self.secret_store.generate_ref()
            self.secret_store.store(credential_ref, credential)

        async for session in get_db_session():
            existing = await session.execute(
                select(AgentIntegration).where(AgentIntegration.name == name)
            )
            if existing.scalar_one_or_none():
                raise ValidationError(f"Agent integration with name '{name}' already exists")

            integration = AgentIntegration(
                name=name,
                display_name=display_name,
                description=description,
                framework=AgentFramework(framework),
                protocol=protocol,
                adapter_type=adapter_type,
                endpoint=endpoint,
                auth_method=AuthenticationMethod(auth_method),
                declared_capabilities=declared_capabilities,
                verified_capabilities={cap: CapabilityStatus.DECLARED.value for cap in declared_capabilities},
                intended_operation=intended_operation,
                status=AgentStatus.REGISTERED,
                credential_ref=credential_ref,
            )
            session.add(integration)
            await session.commit()
            await session.refresh(integration)

            logger.info("agent_integration_created", integration_id=integration.id, name=name)
            return integration

    async def update_agent_integration(
        self,
        integration_id: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        endpoint: Optional[str] = None,
        auth_method: Optional[str] = None,
        declared_capabilities: Optional[list[str]] = None,
        credential: Optional[str] = None,
        enabled: Optional[bool] = None,
        intended_operation: Optional[str] = None,
    ) -> AgentIntegration:
        async for session in get_db_session():
            integration = await session.get(AgentIntegration, integration_id)
            if not integration:
                raise NotFoundError(f"Agent integration not found: {integration_id}")

            if display_name is not None:
                integration.display_name = display_name
            if description is not None:
                integration.description = description
            if endpoint is not None:
                integration.endpoint = endpoint
            if auth_method is not None:
                try:
                    integration.auth_method = AuthenticationMethod(auth_method)
                except ValueError:
                    raise ValidationError(f"Invalid auth method: {auth_method}")
            if declared_capabilities is not None:
                integration.declared_capabilities = declared_capabilities
                integration.verified_capabilities = {
                    cap: CapabilityStatus.DECLARED.value for cap in declared_capabilities
                }
            if credential is not None:
                if integration.credential_ref:
                    self.secret_store.store(integration.credential_ref, credential)
                else:
                    integration.credential_ref = self.secret_store.generate_ref()
                    self.secret_store.store(integration.credential_ref, credential)
            if enabled is not None:
                integration.enabled = enabled
                if enabled:
                    integration.status = AgentStatus.REGISTERED
                else:
                    integration.status = AgentStatus.DISABLED
            if intended_operation is not None:
                integration.intended_operation = intended_operation

            integration.updated_at = datetime.utcnow()
            await session.commit()
            await session.refresh(integration)

            logger.info("agent_integration_updated", integration_id=integration_id)
            return integration

    async def get_agent_integration(self, integration_id: str) -> AgentIntegration:
        async for session in get_db_session():
            integration = await session.get(AgentIntegration, integration_id)
            if not integration:
                raise NotFoundError(f"Agent integration not found: {integration_id}")
            return integration

    async def list_agent_integrations(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        framework: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Tuple[List[AgentIntegration], int]:
        async for session in get_db_session():
            query = select(AgentIntegration)

            if status:
                query = query.where(AgentIntegration.status == status)
            if framework:
                query = query.where(AgentIntegration.framework == framework)
            if enabled is not None:
                query = query.where(AgentIntegration.enabled == enabled)

            total_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(total_query)
            total = total_result.scalar() or 0

            query = query.order_by(AgentIntegration.created_at.desc())
            query = query.offset((page - 1) * page_size).limit(page_size)
            result = await session.execute(query)
            integrations = list(result.scalars().all())

            return integrations, total

    async def delete_agent_integration(self, integration_id: str) -> bool:
        async for session in get_db_session():
            integration = await session.get(AgentIntegration, integration_id)
            if not integration:
                raise NotFoundError(f"Agent integration not found: {integration_id}")

            if integration.credential_ref:
                self.secret_store.delete(integration.credential_ref)

            await session.execute(
                delete(DatasetAccessGrant).where(
                    DatasetAccessGrant.integration_type == "agent",
                    DatasetAccessGrant.integration_id == integration_id,
                )
            )

            await session.delete(integration)
            await session.commit()

            logger.info("agent_integration_deleted", integration_id=integration_id)
            return True

    async def grant_dataset_access(
        self,
        integration_type: str,
        integration_id: str,
        dataset_id: str,
        dataset_version_id: Optional[str],
        operation: str,
        granted_by: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> DatasetAccessGrant:
        async for session in get_db_session():
            existing = await session.execute(
                select(DatasetAccessGrant).where(
                    DatasetAccessGrant.integration_type == integration_type,
                    DatasetAccessGrant.integration_id == integration_id,
                    DatasetAccessGrant.dataset_id == dataset_id,
                    DatasetAccessGrant.dataset_version_id == dataset_version_id,
                    DatasetAccessGrant.operation == operation,
                    DatasetAccessGrant.revoked_at.is_(None),
                )
            )
            if existing.scalar_one_or_none():
                raise ValidationError("Access grant already exists")

            grant = DatasetAccessGrant(
                integration_type=integration_type,
                integration_id=integration_id,
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
                operation=operation,
                granted_by=granted_by,
                expires_at=expires_at,
            )
            session.add(grant)
            await session.commit()
            await session.refresh(grant)

            logger.info(
                "dataset_access_granted",
                integration_type=integration_type,
                integration_id=integration_id,
                dataset_id=dataset_id,
                operation=operation,
            )
            return grant

    async def revoke_dataset_access(
        self,
        integration_type: str,
        integration_id: str,
        dataset_id: str,
        dataset_version_id: Optional[str],
        operation: str,
    ) -> bool:
        async for session in get_db_session():
            result = await session.execute(
                select(DatasetAccessGrant).where(
                    DatasetAccessGrant.integration_type == integration_type,
                    DatasetAccessGrant.integration_id == integration_id,
                    DatasetAccessGrant.dataset_id == dataset_id,
                    DatasetAccessGrant.dataset_version_id == dataset_version_id,
                    DatasetAccessGrant.operation == operation,
                    DatasetAccessGrant.revoked_at.is_(None),
                )
            )
            grant = result.scalar_one_or_none()
            if not grant:
                raise NotFoundError("Access grant not found")

            grant.revoked_at = datetime.utcnow()
            await session.commit()

            logger.info(
                "dataset_access_revoked",
                integration_type=integration_type,
                integration_id=integration_id,
                dataset_id=dataset_id,
                operation=operation,
            )
            return True

    async def list_dataset_access_grants(
        self,
        integration_type: Optional[str] = None,
        integration_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> List[DatasetAccessGrant]:
        async for session in get_db_session():
            query = select(DatasetAccessGrant).where(DatasetAccessGrant.revoked_at.is_(None))

            if integration_type:
                query = query.where(DatasetAccessGrant.integration_type == integration_type)
            if integration_id:
                query = query.where(DatasetAccessGrant.integration_id == integration_id)
            if dataset_id:
                query = query.where(DatasetAccessGrant.dataset_id == dataset_id)

            query = query.order_by(DatasetAccessGrant.created_at.desc())
            result = await session.execute(query)
            return list(result.scalars().all())


def get_integration_registry_service() -> IntegrationRegistryService:
    return IntegrationRegistryService()