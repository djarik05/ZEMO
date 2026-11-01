import pytest
import sys
import os

# Set test environment before any imports
os.environ["ENVIRONMENT"] = "test"

from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import Settings, get_settings
from core.database.config import get_database, Base
from core.exceptions import register_exception_handlers
from core.logging import setup_logging
from api.v1.health import router as health_router
from api.v1.integrations.router import router as integrations_router
from core.database.models.integration import Provider, ModelCapabilityInfo, ProviderType, ProviderStatus, AuthenticationMethod


def create_test_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Zemo - AI/ML Model Training & Experimentation Platform API",
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(integrations_router, prefix=settings.api_prefix)

    return app


@pytest.fixture
def app():
    return create_test_app()


@pytest.fixture
async def async_client(app):
    import tempfile
    import core.database.config as db_config
    
    # Use a temporary file-based database for testing
    test_db_fd, test_db_path = tempfile.mkstemp(suffix='.db')
    os.close(test_db_fd)
    test_db_url = f"sqlite+aiosqlite:///{test_db_path}"
    
    # Set environment variable before any get_settings() call
    os.environ["DATABASE_URL"] = test_db_url
    
    # Clear lru_cache on get_settings
    get_settings.cache_clear()
    
    # Reset cached database
    db_config._db = None
    
    settings = get_settings()
    setup_logging()
    app.state.settings = settings
    
    db = get_database()
    # Drop and recreate all tables for this test database
    await db.drop_tables()
    await db.create_tables()
    
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    try:
        yield client
    finally:
        await client.aclose()
        await db.close()
        # Ensure all connections are closed before deleting the file
        await db.engine.dispose()
        if os.path.exists(test_db_path):
            try:
                os.unlink(test_db_path)
            except PermissionError:
                # Windows file locking - ignore
                pass
        # Restore original DATABASE_URL
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        get_settings.cache_clear()
        db_config._db = None


class TestProviderAPI:
    """Test AI Provider API endpoints."""

    @pytest.mark.asyncio
    async def test_create_provider(self, async_client: AsyncClient):
        """Test creating a new AI provider."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            
            mock_provider = Provider(
                id="test-provider-id",
                name="test-openai",
                display_name="Test OpenAI",
                provider_type=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                auth_method=AuthenticationMethod.API_KEY,
                credential_ref="cred-ref-123",
                default_headers={},
                extra_config={},
                status=ProviderStatus.INACTIVE,
                enabled=True,
            )
            mock_registry.create_provider.return_value = mock_provider

            response = await async_client.post(
                "/api/v1/integrations/providers",
                json={
                    "name": "test-openai",
                    "display_name": "Test OpenAI",
                    "provider_type": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "auth_method": "api_key",
                    "credential": "sk-test-key",
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test-openai"
        assert data["display_name"] == "Test OpenAI"
        assert data["provider_type"] == "openai"
        assert data["status"] == "inactive"

    @pytest.mark.asyncio
    async def test_create_provider_invalid_type(self, async_client: AsyncClient):
        """Test creating provider with invalid type."""
        response = await async_client.post(
            "/api/v1/integrations/providers",
            json={
                "name": "test-provider",
                "display_name": "Test Provider",
                "provider_type": "invalid_type",
                "base_url": "https://api.example.com/v1",
                "auth_method": "api_key",
                "credential": "test-key",
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_provider_missing_credential(self, async_client: AsyncClient):
        """Test creating provider without credential for non-none auth."""
        response = await async_client.post(
            "/api/v1/integrations/providers",
            json={
                "name": "test-provider",
                "display_name": "Test Provider",
                "provider_type": "openai",
                "base_url": "https://api.openai.com/v1",
                "auth_method": "api_key",
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_providers(self, async_client: AsyncClient):
        """Test listing AI providers."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            
            mock_providers = [
                Provider(
                    id="provider-1",
                    name="openai",
                    display_name="OpenAI",
                    provider_type=ProviderType.OPENAI,
                    base_url="https://api.openai.com/v1",
                    auth_method=AuthenticationMethod.API_KEY,
                    credential_ref="cred-1",
                    default_headers={},
                    extra_config={},
                    status=ProviderStatus.ACTIVE,
                    enabled=True,
                ),
                Provider(
                    id="provider-2",
                    name="anthropic",
                    display_name="Anthropic",
                    provider_type=ProviderType.ANTHROPIC,
                    base_url="https://api.anthropic.com",
                    auth_method=AuthenticationMethod.API_KEY,
                    credential_ref="cred-2",
                    default_headers={},
                    extra_config={},
                    status=ProviderStatus.INACTIVE,
                    enabled=True,
                ),
            ]
            mock_registry.list_providers.return_value = (mock_providers, 2)

            response = await async_client.get("/api/v1/integrations/providers")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        assert data["items"][0]["name"] == "openai"
        assert data["items"][1]["name"] == "anthropic"

    @pytest.mark.asyncio
    async def test_get_provider(self, async_client: AsyncClient):
        """Test getting a specific provider."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            
            mock_provider = Provider(
                id="test-provider-id",
                name="test-openai",
                display_name="Test OpenAI",
                provider_type=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                auth_method=AuthenticationMethod.API_KEY,
                credential_ref="cred-ref-123",
                default_headers={},
                extra_config={},
                status=ProviderStatus.ACTIVE,
                enabled=True,
            )
            mock_registry.get_provider.return_value = mock_provider

            response = await async_client.get("/api/v1/integrations/providers/test-provider-id")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-provider-id"
        assert data["name"] == "test-openai"

    @pytest.mark.asyncio
    async def test_get_provider_not_found(self, async_client: AsyncClient):
        """Test getting a non-existent provider."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            mock_registry.get_provider.side_effect = Exception("Not found")

            response = await async_client.get("/api/v1/integrations/providers/nonexistent")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_provider(self, async_client: AsyncClient):
        """Test updating a provider."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            
            mock_provider = Provider(
                id="test-provider-id",
                name="test-openai",
                display_name="Updated OpenAI",
                provider_type=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                auth_method=AuthenticationMethod.API_KEY,
                credential_ref="cred-ref-123",
                default_headers={},
                extra_config={},
                status=ProviderStatus.ACTIVE,
                enabled=True,
            )
            mock_registry.update_provider.return_value = mock_provider

            response = await async_client.patch(
                "/api/v1/integrations/providers/test-provider-id",
                json={
                    "display_name": "Updated OpenAI",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "Updated OpenAI"

    @pytest.mark.asyncio
    async def test_test_provider_connection(self, async_client: AsyncClient):
        """Test provider connection testing."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            
            mock_registry.test_provider_connection.return_value = {
                "success": True,
                "status": "active",
                "diagnostic": "Connection successful",
                "capabilities": {"text_generation": "VERIFIED", "embeddings": "VERIFIED"},
            }

            response = await async_client.post("/api/v1/integrations/providers/test-provider-id/test")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_smoke_test_provider(self, async_client: AsyncClient):
        """Test provider smoke test."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            
            mock_registry.smoke_test_provider.return_value = {
                "success": True,
                "model_tested": "gpt-3.5-turbo",
                "status": "active",
                "diagnostic": "Smoke test successful",
                "capabilities": {"text_generation": "VERIFIED"},
            }

            response = await async_client.post(
                "/api/v1/integrations/providers/test-provider-id/smoke-test",
                json={"model": "gpt-3.5-turbo"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_tested"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_list_provider_models(self, async_client: AsyncClient):
        """Test listing models for a provider."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            
            mock_models = [
                ModelCapabilityInfo(
                    id="model-1",
                    provider_id="test-provider-id",
                    model_identifier="gpt-4",
                    display_name="GPT-4",
                    capabilities=["text_generation", "function_calling", "vision"],
                    max_context_tokens=128000,
                    max_output_tokens=4096,
                    supports_streaming=True,
                    supports_system_prompt=True,
                    supports_parallel_tool_calls=True,
                    input_modalities=["text", "image"],
                    output_modalities=["text"],
                    pricing_input_per_1k=0.03,
                    pricing_output_per_1k=0.06,
                    is_deprecated=False,
                    metadata={},
                ),
                ModelCapabilityInfo(
                    id="model-2",
                    provider_id="test-provider-id",
                    model_identifier="gpt-3.5-turbo",
                    display_name="GPT-3.5 Turbo",
                    capabilities=["text_generation", "function_calling"],
                    max_context_tokens=16384,
                    max_output_tokens=4096,
                    supports_streaming=True,
                    supports_system_prompt=True,
                    supports_parallel_tool_calls=True,
                    input_modalities=["text"],
                    output_modalities=["text"],
                    pricing_input_per_1k=0.0015,
                    pricing_output_per_1k=0.002,
                    is_deprecated=False,
                    metadata={},
                ),
            ]
            mock_registry.list_provider_models.return_value = mock_models

            response = await async_client.get("/api/v1/integrations/providers/test-provider-id/models")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        assert data["items"][0]["model_identifier"] == "gpt-4"
        assert data["items"][1]["model_identifier"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_get_model_capabilities(self, async_client: AsyncClient):
        """Test getting model capabilities."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            
            mock_capability = ModelCapabilityInfo(
                id="model-1",
                provider_id="test-provider-id",
                model_identifier="gpt-4",
                display_name="GPT-4",
                capabilities=["text_generation", "function_calling", "vision"],
                max_context_tokens=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_system_prompt=True,
                supports_parallel_tool_calls=True,
                input_modalities=["text", "image"],
                output_modalities=["text"],
                pricing_input_per_1k=0.03,
                pricing_output_per_1k=0.06,
                is_deprecated=False,
                metadata={},
            )
            mock_registry.get_model_capabilities.return_value = mock_capability

            response = await async_client.get(
                "/api/v1/integrations/models/gpt-4/capabilities?provider_id=test-provider-id"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model_identifier"] == "gpt-4"
        assert data["display_name"] == "GPT-4"
        assert "text_generation" in data["capabilities"]

    @pytest.mark.asyncio
    async def test_upsert_model_capability(self, async_client: AsyncClient):
        """Test adding/updating model capability."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            
            mock_capability = ModelCapabilityInfo(
                id="model-new",
                provider_id="test-provider-id",
                model_identifier="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                capabilities=["text_generation", "function_calling", "vision"],
                max_context_tokens=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_system_prompt=True,
                supports_parallel_tool_calls=True,
                input_modalities=["text", "image"],
                output_modalities=["text"],
                pricing_input_per_1k=0.01,
                pricing_output_per_1k=0.03,
                is_deprecated=False,
                metadata={},
            )
            mock_registry.upsert_model_capability.return_value = mock_capability

            response = await async_client.post(
                "/api/v1/integrations/providers/test-provider-id/models",
                json={
                    "model_identifier": "gpt-4-turbo",
                    "display_name": "GPT-4 Turbo",
                    "capabilities": ["text_generation", "function_calling", "vision"],
                    "max_context_tokens": 128000,
                    "max_output_tokens": 4096,
                    "supports_streaming": True,
                    "supports_system_prompt": True,
                    "supports_parallel_tool_calls": True,
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "pricing_input_per_1k": 0.01,
                    "pricing_output_per_1k": 0.03,
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["model_identifier"] == "gpt-4-turbo"
        assert data["display_name"] == "GPT-4 Turbo"

    @pytest.mark.asyncio
    async def test_update_model_capability(self, async_client: AsyncClient):
        """Test updating model capability."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            
            # Mock existing capability
            mock_existing = ModelCapabilityInfo(
                id="model-1",
                provider_id="test-provider-id",
                model_identifier="gpt-4",
                display_name="GPT-4",
                capabilities=["text_generation", "function_calling", "vision"],
                max_context_tokens=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_system_prompt=True,
                supports_parallel_tool_calls=True,
                input_modalities=["text", "image"],
                output_modalities=["text"],
                pricing_input_per_1k=0.03,
                pricing_output_per_1k=0.06,
                is_deprecated=False,
                metadata={},
            )
            # Mock updated capability
            mock_updated = ModelCapabilityInfo(
                id="model-1",
                provider_id="test-provider-id",
                model_identifier="gpt-4",
                display_name="GPT-4 Updated",
                capabilities=["text_generation", "function_calling", "vision", "audio"],
                max_context_tokens=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_system_prompt=True,
                supports_parallel_tool_calls=True,
                input_modalities=["text", "image", "audio"],
                output_modalities=["text"],
                pricing_input_per_1k=0.03,
                pricing_output_per_1k=0.06,
                is_deprecated=False,
                metadata={},
            )
            mock_registry.get_model_capabilities.return_value = mock_existing
            mock_registry.upsert_model_capability.return_value = mock_updated

            response = await async_client.patch(
                "/api/v1/integrations/providers/test-provider-id/models/gpt-4",
                json={
                    "display_name": "GPT-4 Updated",
                    "capabilities": ["text_generation", "function_calling", "vision", "audio"],
                    "input_modalities": ["text", "image", "audio"],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "GPT-4 Updated"
        assert "audio" in data["capabilities"]

    @pytest.mark.asyncio
    async def test_delete_model_capability(self, async_client: AsyncClient):
        """Test deleting model capability."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            mock_registry.delete_model_capability.return_value = True

            response = await async_client.delete(
                "/api/v1/integrations/providers/test-provider-id/models/gpt-4"
            )

        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_disable_provider(self, async_client: AsyncClient):
        """Test disabling a provider."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            
            mock_provider = Provider(
                id="test-provider-id",
                name="test-openai",
                display_name="Test OpenAI",
                provider_type=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                auth_method=AuthenticationMethod.API_KEY,
                credential_ref="cred-ref-123",
                default_headers={},
                extra_config={},
                status=ProviderStatus.INACTIVE,
                enabled=False,
            )
            mock_registry.update_provider.return_value = mock_provider

            response = await async_client.post("/api/v1/integrations/providers/test-provider-id/disable")

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_delete_provider(self, async_client: AsyncClient):
        """Test deleting a provider."""
        with patch("services.integrations.provider_registry.get_provider_registry_service") as mock_service:
            mock_registry = AsyncMock()
            mock_service.return_value = mock_registry
            mock_registry.delete_provider.return_value = True

            response = await async_client.delete("/api/v1/integrations/providers/test-provider-id")

        assert response.status_code == 204


class TestProviderRegistryService:
    """Test ProviderRegistryService methods."""

    @pytest.mark.asyncio
    async def test_create_provider_service(self):
        """Test creating provider through service."""
        from services.integrations.provider_registry import ProviderRegistryService
        
        service = ProviderRegistryService()
        
        # Mock secret store and SSRF protector
        with patch.object(service.secret_store, 'generate_ref', return_value="test-ref"), \
             patch.object(service.secret_store, 'store'), \
             patch.object(service.ssrf_protector, 'validate_url'), \
             patch("services.integrations.provider_registry.get_db_session") as mock_db:
            
            mock_session = AsyncMock()
            mock_db.return_value.__aiter__.return_value = [mock_session]
            
            mock_provider = Provider(
                id="test-id",
                name="test-provider",
                display_name="Test Provider",
                provider_type=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                auth_method=AuthenticationMethod.API_KEY,
                credential_ref="test-ref",
                default_headers={},
                extra_config={},
                status=ProviderStatus.INACTIVE,
                enabled=True,
            )
            mock_session.get.return_value = None
            mock_session.add = MagicMock()
            mock_session.commit = AsyncMock()
            mock_session.refresh = AsyncMock()
            
            result = await service.create_provider(
                name="test-provider",
                display_name="Test Provider",
                provider_type="openai",
                base_url="https://api.openai.com/v1",
                auth_method="api_key",
                credential="sk-test",
            )
            
            assert result.name == "test-provider"
            assert result.provider_type == ProviderType.OPENAI

    @pytest.mark.asyncio
    async def test_list_providers_service(self):
        """Test listing providers through service."""
        from services.integrations.provider_registry import ProviderRegistryService
        
        service = ProviderRegistryService()
        
        with patch("services.integrations.provider_registry.get_db_session") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aiter__.return_value = [mock_session]
            
            mock_providers = [
                Provider(
                    id="provider-1",
                    name="openai",
                    display_name="OpenAI",
                    provider_type=ProviderType.OPENAI,
                    base_url="https://api.openai.com/v1",
                    auth_method=AuthenticationMethod.API_KEY,
                    credential_ref="cred-1",
                    default_headers={},
                    extra_config={},
                    status=ProviderStatus.ACTIVE,
                    enabled=True,
                ),
            ]
            
            mock_result = AsyncMock()
            mock_result.scalars.return_value.all.return_value = mock_providers
            mock_session.execute.return_value = mock_result
            
            mock_count_result = AsyncMock()
            mock_count_result.scalar.return_value = 1
            mock_session.execute.side_effect = [mock_result, mock_count_result]
            
            providers, total = await service.list_providers()
            
            assert len(providers) == 1
            assert total == 1
            assert providers[0].name == "openai"

    @pytest.mark.asyncio
    async def test_get_provider_service(self):
        """Test getting provider through service."""
        from services.integrations.provider_registry import ProviderRegistryService
        
        service = ProviderRegistryService()
        
        with patch("services.integrations.provider_registry.get_db_session") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aiter__.return_value = [mock_session]
            
            mock_provider = Provider(
                id="test-id",
                name="test-provider",
                display_name="Test Provider",
                provider_type=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                auth_method=AuthenticationMethod.API_KEY,
                credential_ref="cred-1",
                default_headers={},
                extra_config={},
                status=ProviderStatus.ACTIVE,
                enabled=True,
            )
            mock_session.get.return_value = mock_provider
            
            result = await service.get_provider("test-id")
            
            assert result.id == "test-id"
            assert result.name == "test-provider"

    @pytest.mark.asyncio
    async def test_delete_provider_service(self):
        """Test deleting provider through service."""
        from services.integrations.provider_registry import ProviderRegistryService
        
        service = ProviderRegistryService()
        
        with patch("services.integrations.provider_registry.get_db_session") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aiter__.return_value = [mock_session]
            
            mock_provider = Provider(
                id="test-id",
                name="test-provider",
                display_name="Test Provider",
                provider_type=ProviderType.OPENAI,
                base_url="https://api.openai.com/v1",
                auth_method=AuthenticationMethod.API_KEY,
                credential_ref="cred-1",
                default_headers={},
                extra_config={},
                status=ProviderStatus.INACTIVE,
                enabled=True,
            )
            mock_session.get.return_value = mock_provider
            mock_session.delete = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.execute = AsyncMock()
            
            with patch.object(service.secret_store, 'delete'):
                result = await service.delete_provider("test-id")
            
            assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])