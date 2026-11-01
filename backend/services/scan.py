import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from core.config import get_settings
from core.database.config import get_db_session
from core.database.models.dataset import (
    Dataset,
    DatasetStatus,
    Scan,
    ScanStatus,
    Finding,
    FindingCategory,
    FindingSeverity,
    FindingConfidence,
)
from core.database.models.dataset_version import (
    DatasetVersion,
    DatasetVersionStatus,
)
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger
from parsers.parsers import ParseError, get_parser
from services.storage import get_storage_service
from detectors import DETECTORS, DEFAULT_DETECTORS, DETECTOR_NAMES
from detectors.base import DetectionContext
from detectors.risk import RiskAssessmentService

logger = get_logger(__name__)


class ScanService:
    def __init__(self):
        self.settings = get_settings()
        self.storage = get_storage_service()
        self.risk_service = RiskAssessmentService()

    async def create_scan(
        self,
        dataset_id: str,
        detectors: list[str] | None = None,
        version_id: str | None = None,
    ) -> Scan:
        detectors = detectors or DEFAULT_DETECTORS
        for d in detectors:
            if d not in DETECTORS:
                raise ValidationError(f"Unknown detector: {d}")

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

            if version.status not in (DatasetVersionStatus.VALIDATED, DatasetVersionStatus.PROFILED, DatasetVersionStatus.READY):
                raise ValidationError(f"Dataset version must be validated/profiled before scanning. Current status: {version.status}")

            scan = Scan(
                dataset_id=dataset_id,
                version_id=version.id,
                status=ScanStatus.QUEUED,
                detectors_enabled=detectors,
                detectors_completed=[],
                finding_counts={},
                risk_assessment={},
            )
            session.add(scan)
            await session.commit()
            await session.refresh(scan)

            logger.info("scan_created", scan_id=scan.id, dataset_id=dataset_id, version_id=version.id, detectors=detectors)
            return scan

    async def run_scan(self, scan_id: str) -> Scan:
        async for session in get_db_session():
            scan = await session.get(Scan, scan_id)
            if not scan:
                raise NotFoundError(f"Scan not found: {scan_id}")

            if scan.status != ScanStatus.QUEUED:
                raise ValidationError(f"Scan cannot be run: current status is {scan.status}")

            scan.status = ScanStatus.RUNNING
            scan.started_at = datetime.utcnow()
            await session.commit()

            try:
                version = await session.get(DatasetVersion, scan.version_id)
                if not version:
                    raise NotFoundError(f"Version not found: {scan.version_id}")

                file_path = self.storage.get_file_path(version.storage_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Storage file not found: {version.storage_path}")

                parser = get_parser(version.format)
                with open(file_path, "rb") as f:
                    parse_result = parser.parse(f)

                if parse_result.errors:
                    raise ValidationError("Dataset parsing failed", {"errors": parse_result.errors})

                context = DetectionContext(
                    dataset_id=scan.dataset_id,
                    version_id=version.id,
                    records=parse_result.records,
                    columns=parse_result.columns,
                    schema_metadata=version.schema_metadata,
                    profile_metadata=version.profile_metadata,
                    file_format=version.format,
                    max_records=self.settings.max_records,
                    max_columns=self.settings.max_columns,
                )

                all_findings = []
                completed_detectors = []

                for detector_name in scan.detectors_enabled:
                    detector = DETECTORS.get(detector_name)
                    if not detector:
                        logger.warning("detector_not_found", detector_name=detector_name, scan_id=scan.id)
                        continue

                    try:
                        findings = detector.detect(context)
                        for finding in findings:
                            finding_obj = Finding(
                                scan_id=scan.id,
                                dataset_id=scan.dataset_id,
                                version_id=scan.version_id,
                                category=finding.category,
                                rule_id=finding.rule_id,
                                severity=finding.severity,
                                confidence=finding.confidence,
                                title=finding.title,
                                description=finding.description,
                                row_number=finding.row_number,
                                column_name=finding.column_name,
                                evidence=finding.evidence,
                                masked_evidence=finding.masked_evidence,
                                remediation_guidance=finding.remediation_guidance,
                                detector_name=finding.detector_name,
                                detector_version=finding.detector_version,
                            )
                            session.add(finding_obj)
                            all_findings.append(finding)

                        completed_detectors.append(detector_name)
                        scan.detectors_completed = completed_detectors
                        await session.commit()
                        logger.info("detector_completed", scan_id=scan.id, detector=detector_name, findings=len(findings))

                    except Exception as e:
                        logger.error("detector_failed", scan_id=scan.id, detector=detector_name, error=str(e))
                        scan.status = ScanStatus.PARTIAL
                        scan.error_summary = f"Detector {detector_name} failed: {str(e)[:500]}"
                        await session.commit()
                        raise

                risk_assessment = self.risk_service.assess(all_findings, completed_detectors)

                finding_counts = {}
                for finding in all_findings:
                    category = finding.category.value
                    finding_counts[category] = finding_counts.get(category, 0) + 1

                scan.status = ScanStatus.COMPLETED
                scan.completed_at = datetime.utcnow()
                scan.finding_counts = finding_counts
                scan.risk_assessment = risk_assessment.to_dict()
                scan.detectors_completed = completed_detectors

                await session.commit()
                await session.refresh(scan)

                logger.info("scan_completed", scan_id=scan.id, findings=len(all_findings), risk=risk_assessment.overall_risk.value)
                return scan

            except Exception as e:
                scan.status = ScanStatus.FAILED
                scan.error_summary = str(e)[:500]
                scan.completed_at = datetime.utcnow()
                await session.commit()
                logger.error("scan_failed", scan_id=scan.id, error=str(e))
                raise

    async def get_scan(self, scan_id: str) -> Scan:
        async for session in get_db_session():
            scan = await session.get(Scan, scan_id)
            if not scan:
                raise NotFoundError(f"Scan not found: {scan_id}")
            return scan

    async def list_scans(
        self,
        dataset_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
    ) -> tuple[list[Scan], int]:
        from sqlalchemy import select, func

        async for session in get_db_session():
            query = select(Scan)
            if dataset_id:
                query = query.where(Scan.dataset_id == dataset_id)
            if status:
                query = query.where(Scan.status == status)

            total_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(total_query)
            total = total_result.scalar() or 0

            query = query.order_by(Scan.created_at.desc())
            query = query.offset((page - 1) * page_size).limit(page_size)
            result = await session.execute(query)
            scans = result.scalars().all()

            return list(scans), total

    async def get_scan_findings(
        self,
        scan_id: str,
        page: int = 1,
        page_size: int = 50,
        category: str | None = None,
        severity: str | None = None,
    ) -> tuple[list[Finding], int]:
        from sqlalchemy import select, func

        async for session in get_db_session():
            scan = await session.get(Scan, scan_id)
            if not scan:
                raise NotFoundError(f"Scan not found: {scan_id}")

            query = select(Finding).where(Finding.scan_id == scan_id)
            if category:
                query = query.where(Finding.category == category)
            if severity:
                query = query.where(Finding.severity == severity)

            total_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(total_query)
            total = total_result.scalar() or 0

            query = query.order_by(Finding.created_at.desc())
            query = query.offset((page - 1) * page_size).limit(page_size)
            result = await session.execute(query)
            findings = result.scalars().all()

            return list(findings), total

    async def get_finding(self, finding_id: str) -> Finding:
        async for session in get_db_session():
            finding = await session.get(Finding, finding_id)
            if not finding:
                raise NotFoundError(f"Finding not found: {finding_id}")
            return finding

    async def get_scan_report(self, scan_id: str) -> dict:
        async for session in get_db_session():
            scan = await session.get(Scan, scan_id)
            if not scan:
                raise NotFoundError(f"Scan not found: {scan_id}")

            from sqlalchemy import select
            findings_result = await session.execute(
                select(Finding).where(Finding.scan_id == scan_id)
            )
            findings = findings_result.scalars().all()

            findings_by_category = {}
            for finding in findings:
                cat = finding.category.value
                if cat not in findings_by_category:
                    findings_by_category[cat] = []
                findings_by_category[cat].append({
                    "id": finding.id,
                    "rule_id": finding.rule_id,
                    "severity": finding.severity.value,
                    "confidence": finding.confidence.value,
                    "title": finding.title,
                    "description": finding.description,
                    "row_number": finding.row_number,
                    "column_name": finding.column_name,
                    "masked_evidence": finding.masked_evidence,
                    "remediation_guidance": finding.remediation_guidance,
                    "detector_name": finding.detector_name,
                    "created_at": finding.created_at.isoformat() if finding.created_at else None,
                })

            return {
                "scan": {
                    "id": scan.id,
                    "dataset_id": scan.dataset_id,
                    "version_id": scan.version_id,
                    "status": scan.status.value,
                    "detectors_enabled": scan.detectors_enabled,
                    "detectors_completed": scan.detectors_completed,
                    "finding_counts": scan.finding_counts,
                    "risk_assessment": scan.risk_assessment,
                    "created_at": scan.created_at.isoformat() if scan.created_at else None,
                    "started_at": scan.started_at.isoformat() if scan.started_at else None,
                    "completed_at": scan.completed_at.isoformat() if scan.completed_at else None,
                    "error_summary": scan.error_summary,
                },
                "findings_by_category": findings_by_category,
                "total_findings": len(findings),
            }

    async def retry_scan(self, scan_id: str) -> Scan:
        async for session in get_db_session():
            scan = await session.get(Scan, scan_id)
            if not scan:
                raise NotFoundError(f"Scan not found: {scan_id}")

            if scan.status not in (ScanStatus.FAILED, ScanStatus.PARTIAL):
                raise ValidationError(f"Scan can only be retried from FAILED or PARTIAL status, current: {scan.status}")

            from sqlalchemy import delete
            await session.execute(delete(Finding).where(Finding.scan_id == scan_id))

            scan.status = ScanStatus.QUEUED
            scan.detectors_completed = []
            scan.finding_counts = {}
            scan.risk_assessment = {}
            scan.error_summary = None
            scan.started_at = None
            scan.completed_at = None
            await session.commit()

            return await self.run_scan(scan_id)


_scan_service: ScanService | None = None


def get_scan_service() -> ScanService:
    global _scan_service
    if _scan_service is None:
        _scan_service = ScanService()
    return _scan_service