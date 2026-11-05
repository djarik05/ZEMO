import os
os.environ["ENVIRONMENT"] = "test"

import pytest
import tempfile
from httpx import AsyncClient, ASGITransport
from main import app as global_app


@pytest.fixture
def app():
    return global_app


@pytest.fixture
async def async_client(app):
    from core.config import get_settings
    from core.database.config import get_database
    from core.logging import setup_logging
    import core.database.config as db_config

    test_db_fd, test_db_path = tempfile.mkstemp(suffix=".db")
    os.close(test_db_fd)
    test_db_url = f"sqlite+aiosqlite:///{test_db_path}"

    os.environ["DATABASE_URL"] = test_db_url
    get_settings.cache_clear()
    db_config._db = None

    settings = get_settings()
    setup_logging()
    app.state.settings = settings

    db = get_database()
    await db.create_tables()

    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    try:
        yield client
    finally:
        await client.aclose()
        await db.close()
        await db.engine.dispose()
        if os.path.exists(test_db_path):
            try:
                os.unlink(test_db_path)
            except PermissionError:
                pass
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        get_settings.cache_clear()
        db_config._db = None