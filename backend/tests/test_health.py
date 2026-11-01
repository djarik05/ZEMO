import pytest
import sys
import os

# Set test environment before any imports
os.environ["ENVIRONMENT"] = "test"

# Clear any cached modules that might have imported integration models
modules_to_clear = [k for k in sys.modules.keys() if k.startswith('core.database.models.integration')]
for mod in modules_to_clear:
    del sys.modules[mod]

# Now import the rest
from httpx import AsyncClient, ASGITransport
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import Settings, get_settings
from core.database.config import get_database
from core.exceptions import register_exception_handlers
from core.logging import setup_logging
from api.v1.health import router as health_router


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

    return app


@pytest.fixture
def app():
    return create_test_app()


@pytest.fixture
async def async_client(app):
    # Manually run lifespan startup
    from contextlib import asynccontextmanager
    from core.config import get_settings
    from core.database.config import get_database
    from core.logging import setup_logging
    import tempfile
    import os
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
    # Drop tables first to avoid index conflicts, then create
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


@pytest.mark.asyncio
async def test_health_endpoint_returns_200(async_client):
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint_returns_expected_fields(async_client):
    response = await async_client.get("/api/v1/health")
    data = response.json()

    assert "app_name" in data
    assert "status" in data
    assert "api_version" in data
    assert "environment" in data

    assert data["app_name"] == "Zemo"
    assert data["status"] == "healthy"
    assert data["api_version"] == "v1"
    assert data["environment"] in ("development", "staging", "production", "test")


@pytest.mark.asyncio
async def test_health_endpoint_routing_works_with_configured_prefix(async_client):
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200

    response_not_found = await async_client.get("/health")
    assert response_not_found.status_code == 404


@pytest.mark.asyncio
async def test_liveness_endpoint(async_client):
    response = await async_client.get("/api/v1/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


@pytest.mark.asyncio
async def test_readiness_endpoint(async_client):
    response = await async_client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["database_connected"] is True
    assert data["database"] == "connected"


@pytest.mark.asyncio
async def test_settings_validation():
    settings = Settings(
        app_name="TestApp",
        environment="test",
        debug=True,
        api_prefix="/api/v1",
        log_level="DEBUG",
        cors_origins=["http://localhost:3000"],
    )
    assert settings.app_name == "TestApp"
    assert settings.environment == "test"
    assert settings.debug is True
    assert settings.api_prefix == "/api/v1"
    assert settings.log_level == "DEBUG"
    assert settings.cors_origins == ["http://localhost:3000"]


@pytest.mark.asyncio
async def test_invalid_environment_raises_error():
    with pytest.raises(ValueError):
        Settings(environment="invalid")


@pytest.mark.asyncio
async def test_invalid_log_level_raises_error():
    with pytest.raises(ValueError):
        Settings(log_level="INVALID")


@pytest.mark.asyncio
async def test_cors_origins_parsing():
    settings = Settings(cors_origins="http://a.com, http://b.com")
    assert settings.cors_origins == ["http://a.com", "http://b.com"]

    settings_list = Settings(cors_origins=["http://a.com", "http://b.com"])
    assert settings_list.cors_origins == ["http://a.com", "http://b.com"]