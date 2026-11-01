from pathlib import Path
from typing import Optional, Union
import pandas as pd

from core.database.config import get_db_session
from core.database.models.dataset import Dataset
from core.database.models.dataset_version import DatasetVersion
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from services.storage import get_storage_service, StorageService

logger = get_logger(__name__)


class DatasetLoader:
    """
    Dedicated dataset loader for the ML engine.
    Independently loads original files from disk with native type inference,
    avoiding string-based representation used in previews.
    Supports CSV, XLSX, JSON, JSONL, and Parquet.
    """

    def __init__(self, storage_service: Optional[StorageService] = None):
        self.storage = storage_service or get_storage_service()

    def load_from_path(
        self,
        file_path: Union[str, Path],
        file_format: str,
        na_values: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load a dataset directly from a local path into a DataFrame with type inference.
        """
        path = Path(file_path)
        if not path.exists():
            raise NotFoundError(f"Dataset file not found at: {path}")

        fmt = file_format.lower().lstrip(".")
        custom_na = na_values or ["", "NA", "N/A", "null", "NULL", "None", "nan", "NaN"]

        try:
            if fmt == "csv":
                try:
                    df = pd.read_csv(
                        path,
                        encoding="utf-8",
                        na_values=custom_na,
                        keep_default_na=True,
                    )
                except UnicodeDecodeError:
                    df = pd.read_csv(
                        path,
                        encoding="latin-1",
                        na_values=custom_na,
                        keep_default_na=True,
                    )
            elif fmt in ("xlsx", "xls"):
                df = pd.read_excel(path, na_values=custom_na)
            elif fmt == "json":
                df = pd.read_json(path)
            elif fmt == "jsonl":
                df = pd.read_json(path, lines=True)
            elif fmt == "parquet":
                df = pd.read_parquet(path)
            else:
                raise ValidationError(
                    f"Unsupported format for ML loading: {fmt}. Supported: csv, xlsx, json, jsonl, parquet"
                )

            logger.info(
                "ml_dataset_loaded",
                path=str(path),
                format=fmt,
                rows=len(df),
                columns=len(df.columns),
            )
            return df
        except Exception as e:
            if isinstance(e, (NotFoundError, ValidationError)):
                raise
            logger.error("ml_dataset_load_failed", path=str(path), format=fmt, error=str(e))
            raise ValidationError(f"Failed to load dataset for ML processing: {e}") from e

    def load_by_storage_path(
        self,
        storage_path: str,
        file_format: str,
    ) -> pd.DataFrame:
        """
        Resolve storage path via StorageService and load DataFrame with native type inference.
        """
        resolved_path = self.storage.get_file_path(storage_path)
        return self.load_from_path(resolved_path, file_format)

    async def load_by_dataset_id(
        self,
        dataset_id: str,
        version_id: Optional[str] = None,
    ) -> tuple[pd.DataFrame, DatasetVersion]:
        """
        Query database for dataset version, resolve physical storage, and load into DataFrame.
        """
        from sqlalchemy import select

        async for session in get_db_session():
            dataset = await session.get(Dataset, dataset_id)
            if not dataset:
                raise NotFoundError(f"Dataset not found: {dataset_id}")

            if version_id:
                version = await session.get(DatasetVersion, version_id)
                if not version or version.dataset_id != dataset_id:
                    raise NotFoundError(f"Version not found: {version_id}")
            else:
                stmt = (
                    select(DatasetVersion)
                    .where(DatasetVersion.dataset_id == dataset_id)
                    .order_by(DatasetVersion.version_number.desc())
                    .limit(1)
                )
                res = await session.execute(stmt)
                version = res.scalar_one_or_none()
                if not version:
                    raise NotFoundError(f"No version found for dataset: {dataset_id}")

            df = self.load_by_storage_path(version.storage_path, version.format)
            return df, version


_dataset_loader: Optional[DatasetLoader] = None


def get_dataset_loader() -> DatasetLoader:
    global _dataset_loader
    if _dataset_loader is None:
        _dataset_loader = DatasetLoader()
    return _dataset_loader
