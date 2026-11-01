import hashlib
import os
import uuid
from pathlib import Path
from typing import BinaryIO

import aiofiles

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class StorageError(Exception):
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class StorageService:
    def __init__(self):
        settings = get_settings()
        self.upload_dir = Path(settings.upload_dir)
        self.max_file_size = settings.max_file_size_bytes
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _generate_storage_name(self, original_filename: str) -> str:
        ext = Path(original_filename).suffix.lower()
        unique_id = uuid.uuid4().hex
        return f"{unique_id}{ext}"

    def _get_storage_path(self, storage_name: str) -> Path:
        safe_name = Path(storage_name).name
        return self.upload_dir / safe_name

    async def save_upload(self, file: BinaryIO, original_filename: str) -> tuple[str, int, str]:
        storage_name = self._generate_storage_name(original_filename)
        storage_path = self._get_storage_path(storage_name)

        hasher = hashlib.sha256()
        total_size = 0

        async with aiofiles.open(storage_path, "wb") as out_file:
            while True:
                chunk = file.read(8192)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > self.max_file_size:
                    await out_file.close()
                    storage_path.unlink(missing_ok=True)
                    raise StorageError(
                        f"File size exceeds maximum allowed size of {self.max_file_size} bytes",
                        {"max_size": self.max_file_size, "received": total_size},
                    )
                hasher.update(chunk)
                await out_file.write(chunk)

        content_hash = hasher.hexdigest()
        logger.info("file_saved", storage_name=storage_name, size=total_size, hash=content_hash[:16])
        return storage_name, total_size, content_hash

    def get_file_path(self, storage_name: str) -> Path:
        return self._get_storage_path(storage_name)

    def delete_file(self, storage_name: str) -> bool:
        storage_path = self._get_storage_path(storage_name)
        if storage_path.exists():
            storage_path.unlink()
            logger.info("file_deleted", storage_name=storage_name)
            return True
        return False

    def file_exists(self, storage_name: str) -> bool:
        return self._get_storage_path(storage_name).exists()


_storage_service: StorageService | None = None


def get_storage_service() -> StorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service