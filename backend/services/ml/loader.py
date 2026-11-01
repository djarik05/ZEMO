from pathlib import Path
from typing import Optional, Union
import pandas as pd

from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from services.storage import get_storage_service, StorageService

logger = get_logger(__name__)


class DatasetLoader:
    """
    Dedicated dataset loader for the ML engine.
    Independently loads original files from disk with native type inference,
    avoiding string-based representation used in previews.
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
            elif fmt == "json":
                df = pd.read_json(path)
            elif fmt == "jsonl":
                df = pd.read_json(path, lines=True)
            else:
                raise ValidationError(f"Unsupported format for ML loading: {fmt}. Supported: csv, json, jsonl")

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


_dataset_loader: Optional[DatasetLoader] = None


def get_dataset_loader() -> DatasetLoader:
    global _dataset_loader
    if _dataset_loader is None:
        _dataset_loader = DatasetLoader()
    return _dataset_loader
