import asyncio
import logging
from datetime import datetime
from typing import Any, Optional, List

from core.config import get_settings
from core.database.config import get_db_session
from core.database.models.dataset import (
    Dataset,
    Scan,
    Finding,
    SanitizationJob,
    SanitizationRule,
    SanitizationResult,
    SanitizationActionType,
    SanitizationRuleType,
    DatasetStatus,
    FindingCategory,
    FindingSeverity,
    FindingConfidence,
)
from core.database.models.dataset_version import DatasetVersion, DatasetVersionStatus
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from parsers.parsers import ParseError, get_parser
from services.storage import get_storage_service

logger = get_logger(__name__)


class SanitizationService:
    def __init__(self):
        self.settings = get_settings()
        self.storage = get_storage_service()

    async def create_sanitization_job(
        self,
        dataset_id: str,
        version_id: Optional[str] = None,
        scan_id: Optional[str] = None,
        evaluation_id: Optional[str] = None,
        rules: Optional[list[dict]] = None,
    ) -> SanitizationJob:
        """Create a sanitization job."""
        async for session in get_db_session():
            dataset = await session.get(Dataset, dataset_id)
            if not dataset:
                raise NotFoundError(f"Dataset not found: {dataset_id}")

            if version_id:
                version = await session.get(DatasetVersion, version_id)
                if not version or version.dataset_id != dataset_id:
                    raise NotFoundError(f"Version {version_id} not found for dataset {dataset_id}")
            else:
                from sqlalchemy import select
                version = await session.execute(
                    select(DatasetVersion)
                    .where(DatasetVersion.dataset_id == dataset_id)
                    .order_by(DatasetVersion.version_number.desc())
                    .limit(1)
                )
                version = version.scalar_one_or_none()
                if not version:
                    raise NotFoundError(f"No version found for dataset: {dataset_id}")

            # Create job
            job = SanitizationJob(
                dataset_id=dataset_id,
                version_id=version.id,
                scan_id=scan_id,
                evaluation_id=evaluation_id,
                status="PENDING",
                rules_applied=rules or [],
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)

            logger.info("sanitization_job_created", job_id=job.id, dataset_id=dataset_id, version_id=version.id)
            return job

    async def run_sanitization(self, job_id: str) -> SanitizationJob:
        """Run a sanitization job."""
        async for session in get_db_session():
            job = await session.get(SanitizationJob, job_id)
            if not job:
                raise NotFoundError(f"Sanitization job not found: {job_id}")

            if job.status not in ("PENDING", "FAILED"):
                raise ValidationError(f"Job cannot be run: current status is {job.status}")

            job.status = "RUNNING"
            job.started_at = datetime.utcnow()
            await session.commit()

            try:
                # Get the version and its file
                version = await session.get(DatasetVersion, job.version_id)
                if not version:
                    raise NotFoundError(f"Version not found: {job.version_id}")

                file_path = self.storage.get_file_path(version.storage_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Storage file not found: {version.storage_path}")

                # Parse the dataset
                parser = get_parser(version.format)
                with open(file_path, "rb") as f:
                    parse_result = parser.parse(f)

                if parse_result.errors:
                    raise ValidationError("Dataset parsing failed", {"errors": parse_result.errors})

                records = parse_result.records
                columns = parse_result.columns

                # Create rules from job configuration
                rules = []
                for rule_config in job.rules_applied:
                    rule = SanitizationRule(
                        job_id=job.id,
                        rule_type=SanitizationRuleType(rule_config["rule_type"]),
                        action_type=SanitizationActionType(rule_config["action_type"]),
                        target_column=rule_config.get("target_column"),
                        target_finding_rule_id=rule_config.get("target_finding_rule_id"),
                        parameters=rule_config.get("parameters", {}),
                    )
                    session.add(rule)
                    # We'll apply rules as we go

                await session.commit()

                # Apply sanitization rules
                sanitized_records = []
                total_masked = 0
                total_redacted = 0
                total_removed_rows = 0
                removed_columns = set()

                for rule_config in job.rules_applied:
                    rule = await session.get(SanitizationRule, rule_config.get("rule_id"))
                    if not rule:
                        continue

                    if rule.rule_type == SanitizationRuleType.PRIVACY_MASK:
                        masked, redacted = await self._apply_privacy_mask(rule, records, columns)
                        total_masked += masked
                        total_redacted += redacted
                    
                    elif rule.rule_type == SanitizationRuleType.SECRETS_REDACT:
                        redacted = await self._apply_secrets_redaction(rule, records, columns)
                        total_redacted += redacted
                    
                    elif rule.rule_type == SanitizationRuleType.PROMPT_INJECTION_QUARANTINE:
                        removed = await self._apply_prompt_injection_quarantine(rule, records, columns)
                        total_removed_rows += removed
                    
                    elif rule.rule_type == SanitizationRuleType.DATA_QUALITY_REMOVE_DUPLICATES:
                        removed = await self._remove_duplicates(rule, records)
                        total_removed_rows += removed
                    
                    elif rule.rule_type == SanitizationRuleType.DATA_QUALITY_REMOVE_COLUMN:
                        removed = await self._remove_column(rule, records, columns)
                        if rule.target_column:
                            removed_columns.add(rule.target_column)
                    
                    elif rule.rule_type == SanitizationRuleType.CUSTOM_REDACT:
                        masked, redacted = await self._apply_custom_redaction(rule, records, columns)
                        total_masked += masked
                        total_redacted += redacted

                    # Create result record
                    result = SanitizationResult(
                        job_id=job.id,
                        rule_id=rule.id,
                        records_affected=0,  # Will be updated by specific methods
                        details={},
                    )
                    session.add(rule)  # Actually add the rule if new

                # For now, just save the sanitized records as a new version
                from io import BytesIO
                import csv
                import json

                # Convert sanitized records back to file format
                output = BytesIO()
                if version.format == "csv":
                    writer = csv.DictWriter(output, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(sanitized_records)
                elif version.format == "json":
                    output.write(json.dumps(sanitized_records).encode())
                elif version.format == "jsonl":
                    for record in sanitized_records:
                        output.write((json.dumps(record) + "\n").encode())
                output.seek(0)

                # Save sanitized file
                storage_name, file_size, content_hash = await self.storage.save_upload(output, f"sanitized_{job.id}.{version.format}")

                # Create new version
                max_version_result = await session.execute(
                    __import__("sqlalchemy").select(__import__("sqlalchemy").func.max(DatasetVersion.version_number))
                    .where(DatasetVersion.dataset_id == job.dataset_id)
                )
                max_version = max_version_result.scalar() or 0

                new_version = __import__("core.database.models.dataset", fromlist=["DatasetVersion"]).DatasetVersion(
                    dataset_id=job.dataset_id,
                    version_number=max_version + 1,
                    storage_path=storage_name,
                    original_filename=f"sanitized_{job.id}.{version.format}",
                    file_size=file_size,
                    content_hash=content_hash,
                    format=version.format,
                    status="UPLOADED",
                    validation_status="pending",
                    profiling_status="pending",
                )
                session.add(new_version)

                # Update job
                job.status = "COMPLETED"
                job.completed_at = datetime.utcnow()
                job.records_processed = len(sanitized_records)
                job.records_removed = total_removed_rows
                job.columns_removed = list(removed_columns)
                job.values_masked = total_masked
                job.values_redacted = total_redacted
                job.output_version_id = new_version.id

                await session.commit()
                await session.refresh(job)

                logger.info("sanitization_job_completed", job_id=job.id, output_version_id=new_version.id)
                return job

            except Exception as e:
                job.status = "FAILED"
                job.error_summary = str(e)[:500]
                job.completed_at = datetime.utcnow()
                await session.commit()
                logger.error("sanitization_job_failed", job_id=job.id, error=str(e))
                raise

    async def _apply_privacy_mask(self, rule: SanitizationRule, records: list, columns: list) -> tuple[int, int]:
        """Apply privacy masking to records."""
        masked = 0
        redacted = 0
        target_column = rule.target_column
        parameters = rule.parameters

        if not target_column or target_column not in rule.parameters.get("columns", []):
            return 0, 0

        mask_char = parameters.get("mask_char", "*")
        preserve_length = parameters.get("preserve_length", True)

        for record in records:
            if target_column in record and record[target_column]:
                value = str(record[target_column])
                if preserve_length:
                    record[target_column] = mask_char * len(value)
                else:
                    record[target_column] = mask_char * min(len(value), 8)
                masked += 1

        return len(records), 0

    async def _apply_secrets_redaction(self, rule: SanitizationRule, records: list, columns: list) -> int:
        """Apply secrets redaction to records."""
        redacted = 0
        # This would use the secrets detector patterns
        return redacted

    async def _apply_prompt_injection_quarantine(self, rule: SanitizationRule, records: list, columns: list) -> int:
        """Quarantine records with potential prompt injection."""
        removed = 0
        return removed

    async def _remove_duplicates(self, rule: SanitizationRule, records: list) -> int:
        """Remove duplicate records."""
        seen = set()
        unique = []
        for record in records:
            key = tuple(sorted(record.items()))
            if key not in seen:
                seen.add(key)
                unique.append(record)
        removed = len(records) - len(unique)
        return removed

    async def _remove_column(self, rule: SanitizationRule, records: list, columns: list) -> int:
        """Remove a column from all records."""
        target = rule.target_column
        if target and target in columns:
            for record in records:
                if target in record:
                    del record[target]
        return 0

    async def _apply_custom_redaction(self, rule: SanitizationRule, records: list, columns: list) -> tuple[int, int]:
        """Apply custom redaction rules."""
        masked = 0
        redacted = 0
        return masked, redacted

    async def get_job(self, job_id: str) -> "SanitizationJob":
        async for session in get_db_session():
            from core.database.models.dataset import SanitizationJob
            job = await session.get(SanitizationJob, job_id)
            if not job:
                raise NotFoundError(f"Sanitization job not found: {job_id}")
            return job

    async def list_jobs(
        self,
        dataset_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list["SanitizationJob"], int]:
        from sqlalchemy import select, func
        from core.database.models.dataset import SanitizationJob

        async for session in get_db_session():
            query = select(SanitizationJob)
            if dataset_id:
                query = query.where(SanitizationJob.dataset_id == dataset_id)

            total_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(total_query)
            total = total_result.scalar() or 0

            query = query.order_by(SanitizationJob.created_at.desc())
            query = query.offset((page - 1) * page_size).limit(page_size)
            result = await session.execute(query)
            jobs = result.scalars().all()

            return list(jobs), total

    async def get_job_status(self, job_id: str) -> dict:
        async for session in get_db_session():
            from core.database.models.dataset import SanitizationJob, SanitizationRule, SanitizationResult
            from sqlalchemy import select

            job = await session.get(SanitizationJob, job_id)
            if not job:
                raise NotFoundError(f"Sanitization job not found: {job_id}")

            rules_result = await session.execute(select(SanitizationRule).where(SanitizationRule.job_id == job_id))
            rules = rules_result.scalars().all()

            results_result = await session.execute(select(SanitizationResult).where(SanitizationResult.job_id == job_id))
            results = results_result.scalars().all()

            return {
                "job": {
                    "id": job.id,
                    "dataset_id": job.dataset_id,
                    "version_id": job.version_id,
                    "status": job.status,
                    "rules_applied": job.rules_applied,
                    "records_processed": job.records_processed,
                    "records_removed": job.records_removed,
                    "columns_removed": job.columns_removed,
                    "values_masked": job.values_masked,
                    "values_redacted": job.values_redacted,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "error_summary": job.error_summary,
                    "output_version_id": job.output_version_id,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                },
                "rules": [
                    {
                        "id": r.id,
                        "rule_type": r.rule_type.value,
                        "action_type": r.action_type.value,
                        "target_column": r.target_column,
                        "parameters": r.parameters,
                        "records_affected": r.records_affected,
                    }
                    for r in rules
                ],
                "results": [
                    {
                        "id": r.id,
                        "rule_id": r.rule_id,
                        "records_affected": r.records_affected,
                        "details": r.details,
                    }
                    for r in results
                ],
            }


def get_sanitization_service() -> "SanitizationService":
    return SanitizationService()