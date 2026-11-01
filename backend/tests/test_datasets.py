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
from io import BytesIO
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import Settings, get_settings
from core.database.config import get_database, Base
from core.database.models.dataset import Dataset, DatasetFormat, DatasetStatus
from core.database.models.dataset_version import DatasetVersion, DatasetVersionStatus
from parsers.parsers import CSVParser, JSONParser, JSONLParser, ParseError
from parsers.profiling import calculate_profile
from services.storage import get_storage_service
from core.exceptions import register_exception_handlers
from core.logging import setup_logging
from api.v1.health import router as health_router
from api.v1.datasets.router import router as datasets_router
from api.v1.algorithms.router import router as algorithms_router


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
    app.include_router(datasets_router, prefix=settings.api_prefix)
    app.include_router(algorithms_router, prefix=settings.api_prefix)

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


@pytest.fixture
async def test_db():
    import uuid
    from pathlib import Path
    from core.database.test_config import create_database
    
    # Use a unique file-based database for each test run
    test_db_path = Path(f"./data/test_zemo_{uuid.uuid4().hex}.db")
    test_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    db = create_database(f"sqlite+aiosqlite:///{test_db_path}", echo=False)
    await db.create_tables()
    yield db
    await db.close()
    # Clean up the test database file
    if test_db_path.exists():
        test_db_path.unlink()


class TestCSVParser:
    def test_parse_valid_csv(self):
        parser = CSVParser()
        csv_content = b"name,age,city\nJohn,30,NYC\nJane,25,LA\n"
        file = BytesIO(csv_content)
        result = parser.parse(file)

        assert result.record_count == 2
        assert result.column_count == 3
        assert result.columns == ["name", "age", "city"]
        assert len(result.records) == 2
        assert result.records[0]["name"] == "John"
        assert result.records[1]["name"] == "Jane"
        assert not result.errors

    def test_parse_empty_csv(self):
        parser = CSVParser()
        csv_content = b""
        file = BytesIO(csv_content)
        result = parser.parse(file)

        assert result.record_count == 0
        assert result.column_count == 0
        assert len(result.errors) == 1
        assert result.errors[0]["type"] == "empty_file"

    def test_parse_csv_with_duplicate_columns(self):
        parser = CSVParser()
        csv_content = b"name,age,name\nJohn,30,NYC\n"
        file = BytesIO(csv_content)
        result = parser.parse(file)

        # pandas handles duplicate columns by renaming them (e.g., "name.1")
        # The parser should detect this and report it
        assert result.column_count == 3
        assert "name.1" in result.columns or "name_1" in result.columns

    def test_parse_csv_with_missing_values(self):
        parser = CSVParser()
        csv_content = b"name,age,city\nJohn,,NYC\n,25,LA\n"
        file = BytesIO(csv_content)
        result = parser.parse(file)

        assert result.record_count == 2
        assert result.column_count == 3
        assert not result.errors


class TestJSONParser:
    def test_parse_json_array(self):
        parser = JSONParser()
        json_content = b'[{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]'
        file = BytesIO(json_content)
        result = parser.parse(file)

        assert result.record_count == 2
        assert result.column_count == 2
        assert set(result.columns) == {"name", "age"}

    def test_parse_json_object(self):
        parser = JSONParser()
        json_content = b'{"name": "John", "age": 30}'
        file = BytesIO(json_content)
        result = parser.parse(file)

        assert result.record_count == 1
        assert result.column_count == 2

    def test_parse_empty_json_array(self):
        parser = JSONParser()
        json_content = b"[]"
        file = BytesIO(json_content)
        result = parser.parse(file)

        assert result.record_count == 0
        assert len(result.errors) == 1
        assert result.errors[0]["type"] == "empty_array"

    def test_parse_invalid_json(self):
        parser = JSONParser()
        json_content = b'{"name": "John",}'
        file = BytesIO(json_content)

        with pytest.raises(ParseError):
            parser.parse(file)


class TestJSONLParser:
    def test_parse_valid_jsonl(self):
        parser = JSONLParser()
        jsonl_content = b'{"name": "John", "age": 30}\n{"name": "Jane", "age": 25}\n'
        file = BytesIO(jsonl_content)
        result = parser.parse(file)

        assert result.record_count == 2
        assert result.column_count == 2
        assert set(result.columns) == {"name", "age"}

    def test_parse_jsonl_with_empty_lines(self):
        parser = JSONLParser()
        jsonl_content = b'{"name": "John"}\n\n{"name": "Jane"}\n'
        file = BytesIO(jsonl_content)
        result = parser.parse(file)

        assert result.record_count == 2
        assert any("Empty line" in w for w in result.warnings)

    def test_parse_jsonl_with_invalid_line(self):
        parser = JSONLParser()
        jsonl_content = b'{"name": "John"}\ninvalid json\n{"name": "Jane"}\n'
        file = BytesIO(jsonl_content)
        result = parser.parse(file)

        assert result.record_count == 2
        assert len(result.errors) == 1
        assert result.errors[0]["type"] == "json_decode_error"
        assert result.errors[0]["line"] == 2


class TestProfiling:
    def test_profile_basic(self):
        records = [
            {"name": "John", "age": 30, "city": "NYC"},
            {"name": "Jane", "age": 25, "city": "LA"},
            {"name": "Bob", "age": 35, "city": "NYC"},
        ]
        columns = ["name", "age", "city"]

        profile = calculate_profile(records, columns)

        assert profile["record_count"] == 3
        assert profile["column_count"] == 3
        assert set(profile["columns"]) == {"name", "age", "city"}
        assert profile["schema"]["age"]["type"] in ("integer", "numeric")
        assert profile["schema"]["name"]["type"] == "categorical"
        assert profile["duplicate_records"] == 0

    def test_profile_with_duplicates(self):
        records = [
            {"name": "John", "age": 30},
            {"name": "John", "age": 30},
            {"name": "Jane", "age": 25},
        ]
        columns = ["name", "age"]

        profile = calculate_profile(records, columns)

        assert profile["duplicate_records"] == 1

    def test_profile_with_nulls(self):
        records = [
            {"name": "John", "age": 30},
            {"name": "Jane", "age": None},
            {"name": None, "age": 25},
        ]
        columns = ["name", "age"]

        profile = calculate_profile(records, columns)

        assert profile["schema"]["name"]["null_count"] == 1
        assert profile["schema"]["age"]["null_count"] == 1

    def test_profile_empty(self):
        records = []
        columns = []

        profile = calculate_profile(records, columns)

        assert profile["record_count"] == 0
        assert profile["column_count"] == 0


class TestDatasetAPI:
    @pytest.mark.asyncio
    async def test_upload_csv_dataset(self, async_client):
        csv_content = b"name,age,city\nJohn,30,NYC\nJane,25,LA\n"
        files = {"file": ("test.csv", BytesIO(csv_content), "text/csv")}

        response = await async_client.post("/api/v1/datasets", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["original_filename"] == "test.csv"
        assert data["format"] == "csv"
        assert data["status"] == "UPLOADED"
        assert "id" in data
        assert "content_hash" in data

    @pytest.mark.asyncio
    async def test_upload_json_dataset(self, async_client):
        json_content = b'[{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]'
        files = {"file": ("test.json", BytesIO(json_content), "application/json")}

        response = await async_client.post("/api/v1/datasets", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["format"] == "json"

    @pytest.mark.asyncio
    async def test_upload_jsonl_dataset(self, async_client):
        jsonl_content = b'{"name": "John", "age": 30}\n{"name": "Jane", "age": 25}\n'
        files = {"file": ("test.jsonl", BytesIO(jsonl_content), "application/jsonl")}

        response = await async_client.post("/api/v1/datasets", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["format"] == "jsonl"

    @pytest.mark.asyncio
    async def test_upload_unsupported_format(self, async_client):
        files = {"file": ("test.txt", BytesIO(b"hello"), "text/plain")}

        response = await async_client.post("/api/v1/datasets", files=files)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_upload_duplicate_dataset(self, async_client):
        csv_content = b"name,age\nJohn,30\n"
        files = {"file": ("test.csv", BytesIO(csv_content), "text/csv")}

        response1 = await async_client.post("/api/v1/datasets", files=files)
        assert response1.status_code == 201

        files = {"file": ("test2.csv", BytesIO(csv_content), "text/csv")}
        response2 = await async_client.post("/api/v1/datasets", files=files)

        assert response2.status_code == 409

    @pytest.mark.asyncio
    async def test_list_datasets(self, async_client):
        csv_content = b"name,age\nJohn,30\n"
        files = {"file": ("test.csv", BytesIO(csv_content), "text/csv")}
        await async_client.post("/api/v1/datasets", files=files)

        response = await async_client.get("/api/v1/datasets")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert data["page"] == 1
        assert "items" in data

    @pytest.mark.asyncio
    async def test_list_datasets_pagination(self, async_client):
        response = await async_client.get("/api/v1/datasets?page=1&page_size=10")

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 10

    @pytest.mark.asyncio
    async def test_get_dataset(self, async_client):
        csv_content = b"name,age\nJohn,30\n"
        files = {"file": ("test.csv", BytesIO(csv_content), "text/csv")}
        create_resp = await async_client.post("/api/v1/datasets", files=files)
        dataset_id = create_resp.json()["id"]

        response = await async_client.get(f"/api/v1/datasets/{dataset_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_dataset(self, async_client):
        response = await async_client.get("/api/v1/datasets/nonexistent-id")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_process_dataset(self, async_client):
        csv_content = b"name,age,city\nJohn,30,NYC\nJane,25,LA\n"
        files = {"file": ("test.csv", BytesIO(csv_content), "text/csv")}
        create_resp = await async_client.post("/api/v1/datasets", files=files)
        dataset_id = create_resp.json()["id"]

        response = await async_client.post(f"/api/v1/datasets/{dataset_id}/process", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PROFILED"
        assert data["validation_status"] == "passed"
        assert data["record_count"] == 2
        assert data["column_count"] == 3

    @pytest.mark.asyncio
    async def test_get_dataset_versions(self, async_client):
        csv_content = b"name,age\nJohn,30\n"
        files = {"file": ("test.csv", BytesIO(csv_content), "text/csv")}
        create_resp = await async_client.post("/api/v1/datasets", files=files)
        dataset_id = create_resp.json()["id"]

        await async_client.post(f"/api/v1/datasets/{dataset_id}/process", json={})

        response = await async_client.get(f"/api/v1/datasets/{dataset_id}/versions")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["version_number"] == 1

    @pytest.mark.asyncio
    async def test_get_dataset_profile(self, async_client):
        csv_content = b"name,age,city\nJohn,30,NYC\nJane,25,LA\n"
        files = {"file": ("test.csv", BytesIO(csv_content), "text/csv")}
        create_resp = await async_client.post("/api/v1/datasets", files=files)
        dataset_id = create_resp.json()["id"]

        await async_client.post(f"/api/v1/datasets/{dataset_id}/process", json={})

        response = await async_client.get(f"/api/v1/datasets/{dataset_id}/profile")

        assert response.status_code == 200
        data = response.json()
        assert data["record_count"] == 2
        assert data["column_count"] == 3
        assert "schema" in data
        assert "statistics" in data

    @pytest.mark.asyncio
    async def test_get_dataset_preview(self, async_client):
        csv_content = b"name,age,city\nJohn,30,NYC\nJane,25,LA\n"
        files = {"file": ("test.csv", BytesIO(csv_content), "text/csv")}
        create_resp = await async_client.post("/api/v1/datasets", files=files)
        dataset_id = create_resp.json()["id"]

        await async_client.post(f"/api/v1/datasets/{dataset_id}/process", json={})

        response = await async_client.get(f"/api/v1/datasets/{dataset_id}/preview?rows=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data["records"]) == 2
        assert data["preview_rows"] == 2
        assert data["total_rows"] == 2
        assert data["columns"] == ["name", "age", "city"]

    @pytest.mark.asyncio
    async def test_add_dataset_version(self, async_client):
        csv_content_v1 = b"name,age\nJohn,30\n"
        files = {"file": ("test.csv", BytesIO(csv_content_v1), "text/csv")}
        create_resp = await async_client.post("/api/v1/datasets", files=files)
        dataset_id = create_resp.json()["id"]

        csv_content_v2 = b"name,age,city\nJohn,30,NYC\nJane,25,LA\n"
        files = {
            "file": ("test_v2.csv", BytesIO(csv_content_v2), "text/csv"),
        }
        response = await async_client.post(
            f"/api/v1/datasets/{dataset_id}/versions",
            files=files,
            data={"format": "csv"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["version_number"] == 2
        # Version is created but not processed yet, so column_count is None
        assert data["column_count"] is None

    @pytest.mark.asyncio
    async def test_validate_dataset_endpoint(self, async_client):
        csv_content = b"name,age,city\nJohn,30,NYC\nJane,25,LA\n"
        files = {"file": ("test.csv", BytesIO(csv_content), "text/csv")}
        create_resp = await async_client.post("/api/v1/datasets", files=files)
        dataset_id = create_resp.json()["id"]

        # Validate the dataset
        response = await async_client.post(f"/api/v1/datasets/{dataset_id}/validate", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "VALIDATED"
        assert data["validation_status"] == "passed"

        # Check validation endpoint
        val_response = await async_client.get(f"/api/v1/datasets/{dataset_id}/validation")
        assert val_response.status_code == 200
        val_data = val_response.json()
        assert val_data["status"] == "passed"
        assert val_data["valid_records"] == 2
        assert val_data["invalid_records"] == 0

    @pytest.mark.asyncio
    async def test_profile_dataset_endpoint(self, async_client):
        csv_content = b"name,age,city\nJohn,30,NYC\nJane,25,LA\n"
        files = {"file": ("test.csv", BytesIO(csv_content), "text/csv")}
        create_resp = await async_client.post("/api/v1/datasets", files=files)
        dataset_id = create_resp.json()["id"]

        # First validate
        await async_client.post(f"/api/v1/datasets/{dataset_id}/validate", json={})

        # Then profile
        response = await async_client.post(f"/api/v1/datasets/{dataset_id}/profile", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PROFILED"

    @pytest.mark.asyncio
    async def test_delete_dataset(self, async_client):
        csv_content = b"name,age\nJohn,30\n"
        files = {"file": ("test.csv", BytesIO(csv_content), "text/csv")}
        create_resp = await async_client.post("/api/v1/datasets", files=files)
        dataset_id = create_resp.json()["id"]

        # Delete the dataset
        response = await async_client.delete(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify dataset is deleted
        get_response = await async_client.get(f"/api/v1/datasets/{dataset_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_dataset(self, async_client):
        response = await async_client.delete("/api/v1/datasets/nonexistent-id")
        assert response.status_code == 404


class TestStorageService:
    @pytest.mark.asyncio
    async def test_save_and_delete_file(self):
        storage = get_storage_service()
        content = b"test content"
        file = BytesIO(content)

        storage_name, size, content_hash = await storage.save_upload(file, "test.txt")

        assert size == len(content)
        assert len(content_hash) == 64
        assert storage.file_exists(storage_name)

        deleted = storage.delete_file(storage_name)
        assert deleted
        assert not storage.file_exists(storage_name)

    @pytest.mark.asyncio
    async def test_file_size_limit(self):
        storage = get_storage_service()
        large_content = b"x" * (storage.max_file_size + 1)
        file = BytesIO(large_content)

        with pytest.raises(Exception) as exc_info:
            await storage.save_upload(file, "large.txt")

        assert "exceeds maximum allowed size" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_filename_collision_handling(self):
        storage = get_storage_service()
        content = b"test content"

        file1 = BytesIO(content)
        name1, _, _ = await storage.save_upload(file1, "test.txt")

        file2 = BytesIO(content)
        name2, _, _ = await storage.save_upload(file2, "test.txt")

        assert name1 != name2
        assert storage.file_exists(name1)
        assert storage.file_exists(name2)

        storage.delete_file(name1)
        storage.delete_file(name2)


class TestSettings:
    @pytest.mark.asyncio
    async def test_max_file_size_setting(self):
        settings = Settings()
        assert settings.max_file_size_bytes == settings.max_file_size_mb * 1024 * 1024

    @pytest.mark.asyncio
    async def test_upload_dir_creation(self):
        settings = Settings()
        import os
        assert os.path.exists(settings.upload_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])