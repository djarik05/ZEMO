from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from core.config import get_settings
from core.database.config import get_db_session
from core.database.models.dataset import Dataset, DatasetFormat, DatasetStatus, DatasetProfile, DatasetValidationResult, ValidationStatus, ProfilingStatus
from core.database.models.dataset_version import DatasetVersion, DatasetVersionStatus
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from parsers.parsers import ParseError, get_parser
from parsers.profiling import calculate_profile
from services.storage import get_storage_service

logger = get_logger(__name__)


class DatasetService:
    def __init__(self):
        self.settings = get_settings()
        self.storage = get_storage_service()

    async def create_dataset(
        self,
        file_content: bytes,
        original_filename: str,
        format: str,
    ) -> Dataset:
        format = format.lower()
        if format not in ("csv", "json", "jsonl"):
            raise ValidationError(f"Unsupported format: {format}")

        from io import BytesIO
        file_obj = BytesIO(file_content)

        storage_name, file_size, content_hash = await self.storage.save_upload(file_obj, original_filename)

        async for session in get_db_session():
            from sqlalchemy import select
            existing_result = await session.execute(select(Dataset).where(Dataset.content_hash == content_hash))
            existing = existing_result.scalar_one_or_none()
            if existing:
                raise ValidationError(
                    "Dataset with identical content already exists",
                    {"existing_dataset_id": existing.id},
                )

            dataset = Dataset(
                name=Path(original_filename).stem,
                original_filename=original_filename,
                format=DatasetFormat(format),
                file_size=file_size,
                content_hash=content_hash,
                status=DatasetStatus.UPLOADED,
                validation_status="pending",
                profiling_status="pending",
            )
            session.add(dataset)
            await session.flush()

            version = DatasetVersion(
                dataset_id=dataset.id,
                version_number=1,
                storage_path=storage_name,
                original_filename=original_filename,
                file_size=file_size,
                content_hash=content_hash,
                format=format,
                status=DatasetVersionStatus.UPLOADED,
                validation_status="pending",
                profiling_status="pending",
            )
            session.add(version)

            await session.commit()
            await session.refresh(dataset)

            logger.info("dataset_created", dataset_id=dataset.id, version_id=version.id)
            return dataset

    async def validate_dataset(self, dataset_id: str) -> DatasetVersion:
        """Validate a dataset (check format, structure, etc.) without profiling."""
        async for session in get_db_session():
            dataset = await session.get(Dataset, dataset_id)
            if not dataset:
                raise NotFoundError(f"Dataset not found: {dataset_id}")

            version = await session.execute(
                __import__("sqlalchemy").select(DatasetVersion)
                .where(DatasetVersion.dataset_id == dataset_id)
                .order_by(DatasetVersion.version_number.desc())
                .limit(1)
            )
            version = version.scalar_one_or_none()
            if not version:
                raise NotFoundError(f"No version found for dataset: {dataset_id}")

            dataset.status = DatasetStatus.VALIDATING
            dataset.validation_status = "in_progress"
            version.status = DatasetVersionStatus.VALIDATING
            version.validation_status = "in_progress"
            await session.commit()

            # Create validation result record
            validation_result = DatasetValidationResult(
                dataset_id=dataset.id,
                version_id=version.id,
                status=ValidationStatus.PENDING,
            )
            session.add(validation_result)

            try:
                file_path = self.storage.get_file_path(version.storage_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Storage file not found: {version.storage_path}")

                parser = get_parser(dataset.format.value)
                with open(file_path, "rb") as f:
                    parse_result = parser.parse(f)

                if parse_result.errors:
                    # Validation failed
                    dataset.status = DatasetStatus.INVALID
                    dataset.validation_status = "failed"
                    version.status = DatasetVersionStatus.INVALID
                    version.validation_status = "failed"
                    
                    validation_result.status = ValidationStatus.FAILED
                    validation_result.errors = parse_result.errors
                    validation_result.warnings = parse_result.warnings
                    validation_result.total_records = parse_result.record_count
                    validation_result.invalid_records = parse_result.record_count
                    validation_result.valid_records = 0
                    validation_result.partial_ingestion = False
                    validation_result.error_summary = str(parse_result.errors[:5])
                    
                    await session.commit()
                    raise ValidationError("Dataset validation failed", {"errors": parse_result.errors})

                # Validation passed
                dataset.status = DatasetStatus.VALIDATED
                dataset.validation_status = "passed"
                dataset.record_count = parse_result.record_count
                dataset.column_count = parse_result.column_count
                dataset.schema_metadata = {
                    "columns": parse_result.columns,
                    "record_count": parse_result.record_count,
                    "column_count": parse_result.column_count,
                }
                dataset.profiling_status = "pending"

                version.status = DatasetVersionStatus.VALIDATED
                version.validation_status = "passed"
                version.record_count = parse_result.record_count
                version.column_count = parse_result.column_count
                version.schema_metadata = {
                    "columns": parse_result.columns,
                    "record_count": parse_result.record_count,
                    "column_count": parse_result.column_count,
                }
                version.profiling_status = "pending"

                validation_result.status = ValidationStatus.PASSED
                validation_result.errors = []
                validation_result.warnings = parse_result.warnings
                validation_result.total_records = parse_result.record_count
                validation_result.valid_records = parse_result.record_count
                validation_result.invalid_records = 0
                validation_result.partial_ingestion = False

                await session.commit()
                await session.refresh(dataset)
                await session.refresh(version)

                logger.info("dataset_validated", dataset_id=dataset_id, version_id=version.id)
                return version

            except ValidationError:
                raise
            except Exception as e:
                dataset.status = DatasetStatus.FAILED
                dataset.validation_status = "error"
                version.status = DatasetVersionStatus.FAILED
                version.validation_status = "error"
                
                validation_result.status = ValidationStatus.ERROR
                validation_result.error_summary = str(e)[:500]
                
                await session.commit()
                logger.error("dataset_validation_failed", dataset_id=dataset_id, error=str(e))
                raise ValidationError(f"Dataset validation failed: {e}")

    async def profile_dataset(self, dataset_id: str) -> DatasetVersion:
        """Profile a validated dataset."""
        async for session in get_db_session():
            dataset = await session.get(Dataset, dataset_id)
            if not dataset:
                raise NotFoundError(f"Dataset not found: {dataset_id}")

            version = await session.execute(
                __import__("sqlalchemy").select(DatasetVersion)
                .where(DatasetVersion.dataset_id == dataset_id)
                .order_by(DatasetVersion.version_number.desc())
                .limit(1)
            )
            version = version.scalar_one_or_none()
            if not version:
                raise NotFoundError(f"No version found for dataset: {dataset_id}")

            if dataset.validation_status != "passed" and version.validation_status != "passed":
                raise ValidationError("Dataset must be validated before profiling")

            dataset.status = DatasetStatus.PROFILING
            dataset.profiling_status = "in_progress"
            version.status = DatasetVersionStatus.PROFILING
            version.profiling_status = "in_progress"
            await session.commit()

            # Create profile record
            profile = DatasetProfile(
                dataset_id=dataset.id,
                version_id=version.id,
                record_count=0,
                column_count=0,
                columns=[],
                schema={},
                statistics={},
                duplicate_records=0,
                empty_columns=[],
                status=ProfilingStatus.IN_PROGRESS,
            )
            session.add(profile)

            try:
                file_path = self.storage.get_file_path(version.storage_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Storage file not found: {version.storage_path}")

                parser = get_parser(dataset.format.value)
                with open(file_path, "rb") as f:
                    parse_result = parser.parse(f)

                if parse_result.errors:
                    # Should not happen if already validated, but handle anyway
                    profile.status = ProfilingStatus.FAILED
                    profile.error_summary = str(parse_result.errors[:5])
                    await session.commit()
                    raise ValidationError("Dataset profiling failed due to parse errors", {"errors": parse_result.errors})

                # Calculate profile
                profile_data = calculate_profile(parse_result.records, parse_result.columns)

                profile.record_count = profile_data["record_count"]
                profile.column_count = profile_data["column_count"]
                profile.columns = profile_data["columns"]
                profile.schema = profile_data["schema"]
                profile.statistics = profile_data["statistics"]
                profile.duplicate_records = profile_data["duplicate_records"]
                profile.empty_columns = profile_data["empty_columns"]
                profile.status = ProfilingStatus.COMPLETED

                dataset.status = DatasetStatus.PROFILED
                dataset.profiling_status = "completed"
                dataset.schema_metadata = profile_data

                version.status = DatasetVersionStatus.PROFILED
                version.profiling_status = "completed"
                version.schema_metadata = profile_data
                version.profile_metadata = profile_data
                version.record_count = profile_data["record_count"]
                version.column_count = profile_data["column_count"]

                await session.commit()
                await session.refresh(dataset)
                await session.refresh(version)
                await session.refresh(profile)

                logger.info("dataset_profiled", dataset_id=dataset_id, version_id=version.id, profile_id=profile.id)
                return version

            except ValidationError:
                raise
            except Exception as e:
                profile.status = ProfilingStatus.FAILED
                profile.error_summary = str(e)[:500]
                
                dataset.status = DatasetStatus.FAILED
                dataset.profiling_status = "error"
                version.status = DatasetVersionStatus.FAILED
                version.profiling_status = "error"
                
                await session.commit()
                logger.error("dataset_profiling_failed", dataset_id=dataset_id, error=str(e))
                raise ValidationError(f"Dataset profiling failed: {e}")

    async def process_dataset(self, dataset_id: str) -> DatasetVersion:
        """Process a dataset (validate + profile in one step)."""
        # First validate
        version = await self.validate_dataset(dataset_id)
        # Then profile
        version = await self.profile_dataset(dataset_id)
        return version

    async def get_dataset(self, dataset_id: str) -> Dataset:
        async for session in get_db_session():
            dataset = await session.get(Dataset, dataset_id)
            if not dataset:
                raise NotFoundError(f"Dataset not found: {dataset_id}")
            return dataset

    async def list_datasets(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
    ) -> tuple[list[Dataset], int]:
        from sqlalchemy import select, func

        async for session in get_db_session():
            query = select(Dataset)
            if status:
                query = query.where(Dataset.status == status)

            total_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(total_query)
            total = total_result.scalar() or 0

            query = query.order_by(Dataset.created_at.desc())
            query = query.offset((page - 1) * page_size).limit(page_size)
            result = await session.execute(query)
            datasets = result.scalars().all()

            return list(datasets), total

    async def get_dataset_versions(self, dataset_id: str) -> list[DatasetVersion]:
        from sqlalchemy import select

        async for session in get_db_session():
            dataset = await session.get(Dataset, dataset_id)
            if not dataset:
                raise NotFoundError(f"Dataset not found: {dataset_id}")

            result = await session.execute(
                select(DatasetVersion)
                .where(DatasetVersion.dataset_id == dataset_id)
                .order_by(DatasetVersion.version_number.desc())
            )
            return list(result.scalars().all())

    async def get_version(self, dataset_id: str, version_id: str) -> DatasetVersion:
        from sqlalchemy import select

        async for session in get_db_session():
            result = await session.execute(
                select(DatasetVersion).where(
                    DatasetVersion.id == version_id,
                    DatasetVersion.dataset_id == dataset_id,
                )
            )
            version = result.scalar_one_or_none()
            if not version:
                raise NotFoundError(f"Version not found: {version_id}")
            return version

    async def get_preview(self, dataset_id: str, version_id: Optional[str] = None, rows: int = 100) -> dict:
        async for session in get_db_session():
            if version_id:
                version = await self.get_version(dataset_id, version_id)
            else:
                version = await session.execute(
                    __import__("sqlalchemy").select(DatasetVersion)
                    .where(DatasetVersion.dataset_id == dataset_id)
                    .order_by(DatasetVersion.version_number.desc())
                    .limit(1)
                )
                version = version.scalar_one_or_none()
                if not version:
                    raise NotFoundError(f"No version found for dataset: {dataset_id}")

            if version.status not in (DatasetVersionStatus.PROFILED, DatasetVersionStatus.VALIDATED):
                raise ValidationError(f"Dataset version not ready for preview: {version.status}")

            file_path = self.storage.get_file_path(version.storage_path)
            if not file_path.exists():
                raise NotFoundError(f"Dataset file not found in storage")

            parser = get_parser(version.format)
            with open(file_path, "rb") as f:
                parse_result = parser.parse(f)

            preview_rows = min(rows, self.settings.preview_max_rows)
            preview_columns = min(len(parse_result.columns), self.settings.preview_max_columns)

            preview_records = parse_result.records[:preview_rows]
            preview_records = [{k: v for k, v in r.items() if k in parse_result.columns[:preview_columns]} for r in preview_records]

            return {
                "records": preview_records,
                "columns": parse_result.columns[:preview_columns],
                "total_rows": parse_result.record_count,
                "total_columns": parse_result.column_count,
                "preview_rows": len(preview_records),
                "preview_columns": preview_columns,
                "truncated": parse_result.record_count > preview_rows or len(parse_result.columns) > preview_columns,
            }

    async def get_profile(self, dataset_id: str, version_id: Optional[str] = None) -> dict:
        async for session in get_db_session():
            if version_id:
                version = await self.get_version(dataset_id, version_id)
            else:
                version = await session.execute(
                    __import__("sqlalchemy").select(DatasetVersion)
                    .where(DatasetVersion.dataset_id == dataset_id)
                    .order_by(DatasetVersion.version_number.desc())
                    .limit(1)
                )
                version = version.scalar_one_or_none()
                if not version:
                    raise NotFoundError(f"No version found for dataset: {dataset_id}")

            # First check if we have a stored profile
            from sqlalchemy import select
            profile_result = await session.execute(
                select(DatasetProfile)
                .where(DatasetProfile.version_id == version.id)
                .order_by(DatasetProfile.created_at.desc())
                .limit(1)
            )
            profile = profile_result.scalar_one_or_none()
            if profile and profile.status == ProfilingStatus.COMPLETED:
                return {
                    "record_count": profile.record_count,
                    "column_count": profile.column_count,
                    "columns": profile.columns,
                    "schema": profile.schema,
                    "statistics": profile.statistics,
                    "duplicate_records": profile.duplicate_records,
                    "empty_columns": profile.empty_columns,
                }

            if version.status not in (DatasetVersionStatus.PROFILED, DatasetVersionStatus.READY):
                raise ValidationError(f"Dataset version not ready for profiling: {version.status}")

            file_path = self.storage.get_file_path(version.storage_path)
            if not file_path.exists():
                raise NotFoundError(f"Dataset file not found in storage")

            parser = get_parser(version.format)
            with open(file_path, "rb") as f:
                parse_result = parser.parse(f)

            profile = calculate_profile(parse_result.records, parse_result.columns)
            return profile

    async def get_validation(self, dataset_id: str, version_id: Optional[str] = None) -> dict:
        """Get validation results for a dataset version."""
        async for session in get_db_session():
            if version_id:
                version = await self.get_version(dataset_id, version_id)
            else:
                version = await session.execute(
                    __import__("sqlalchemy").select(DatasetVersion)
                    .where(DatasetVersion.dataset_id == dataset_id)
                    .order_by(DatasetVersion.version_number.desc())
                    .limit(1)
                )
                version = version.scalar_one_or_none()
                if not version:
                    raise NotFoundError(f"No version found for dataset: {dataset_id}")

            # Get the latest validation result
            from sqlalchemy import select
            validation_result = await session.execute(
                select(DatasetValidationResult)
                .where(DatasetValidationResult.version_id == version.id)
                .order_by(DatasetValidationResult.created_at.desc())
                .limit(1)
            )
            validation_result = validation_result.scalar_one_or_none()
            
            if not validation_result:
                return {
                    "status": "pending",
                    "errors": [],
                    "warnings": [],
                    "valid_records": 0,
                    "invalid_records": 0,
                    "total_records": 0,
                    "partial_ingestion": False,
                }

            return {
                "status": validation_result.status,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "valid_records": validation_result.valid_records,
                "invalid_records": validation_result.invalid_records,
                "total_records": validation_result.total_records,
                "partial_ingestion": validation_result.partial_ingestion,
                "error_summary": validation_result.error_summary,
            }

    async def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and all its versions, profiles, validation results, and files."""
        async for session in get_db_session():
            dataset = await session.get(Dataset, dataset_id)
            if not dataset:
                raise NotFoundError(f"Dataset not found: {dataset_id}")

            # Get all versions to delete their files
            versions_result = await session.execute(
                __import__("sqlalchemy").select(DatasetVersion)
                .where(DatasetVersion.dataset_id == dataset_id)
            )
            versions = versions_result.scalars().all()

            # Delete files from storage
            for version in versions:
                try:
                    self.storage.delete_file(version.storage_path)
                except Exception as e:
                    logger.warning("file_deletion_failed", storage_path=version.storage_path, error=str(e))

            # Delete dataset (cascades to versions, profiles, validation_results)
            await session.delete(dataset)
            await session.commit()

            logger.info("dataset_deleted", dataset_id=dataset_id)
            return True

    async def add_version(self, dataset_id: str, file_content: bytes, original_filename: str, format: str) -> DatasetVersion:
        format = format.lower()
        if format not in ("csv", "json", "jsonl"):
            raise ValidationError(f"Unsupported format: {format}")

        async for session in get_db_session():
            dataset = await session.get(Dataset, dataset_id)
            if not dataset:
                raise NotFoundError(f"Dataset not found: {dataset_id}")

            from io import BytesIO
            file_obj = BytesIO(file_content)

            storage_name, file_size, content_hash = await self.storage.save_upload(file_obj, original_filename)

            max_version_result = await session.execute(
                __import__("sqlalchemy").select(__import__("sqlalchemy").func.max(DatasetVersion.version_number))
                .where(DatasetVersion.dataset_id == dataset_id)
            )
            max_version = max_version_result.scalar() or 0

            version = DatasetVersion(
                dataset_id=dataset.id,
                version_number=max_version + 1,
                storage_path=storage_name,
                original_filename=original_filename,
                file_size=file_size,
                content_hash=content_hash,
                format=format,
                status=DatasetVersionStatus.UPLOADED,
                validation_status="pending",
                profiling_status="pending",
            )
            session.add(version)

            dataset.current_version = max_version + 1
            dataset.updated_at = datetime.utcnow()

            await session.commit()
            await session.refresh(version)

            logger.info("dataset_version_added", dataset_id=dataset_id, version_id=version.id, version_number=version.version_number)
            return version


def get_dataset_service() -> DatasetService:
    return DatasetService()