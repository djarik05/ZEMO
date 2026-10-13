"""
Scan orchestration service.

Executes detection pipelines across privacy, secrets, security, and quality,
persists findings, updates risk scores, and evaluates safety policies.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.exceptions import DatasetNotFoundError
from app.core.logging import get_logger
from app.core.security import generate_storage_id
from app.models.enums import DatasetStatus, ScanStatus, Severity, utcnow
from app.repositories.dataset_repo import DatasetRepository
from app.schemas.finding import FindingSummary
from app.schemas.scan import ScanRequest, ScanResult, StageResult
from app.services.privacy_detector import PrivacyDetector
from app.services.profiler import ProfilerService
from app.services.risk_engine import PolicyEngine, RiskEngine
from app.services.secret_detector import SecretDetector
from app.services.security_analyzer import SecurityAnalyzer

logger = get_logger("scan_service")


class ScanService:
    """Orchestrates comprehensive safety and privacy scans on datasets."""

    def __init__(self, repo: DatasetRepository | None = None) -> None:
        self.repo = repo or DatasetRepository()
        self.privacy_detector = PrivacyDetector()
        self.secret_detector = SecretDetector()
        self.security_analyzer = SecurityAnalyzer()
        self.profiler = ProfilerService(repo=self.repo)
        self.risk_engine = RiskEngine()
        self.policy_engine = PolicyEngine()

    def run_scan(self, request: ScanRequest) -> ScanResult:
        dataset_id = request.dataset_id
        df = self.repo.get_dataframe(dataset_id)
        if df is None:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' not found.",
                detail={"dataset_id": dataset_id},
            )

        scan_id = generate_storage_id()
        started_at = utcnow()
        start_time = time.time()
        stages: list[StageResult] = []
        all_findings: list[dict[str, Any]] = []

        # 1. Privacy Stage
        if request.enable_privacy:
            t0 = time.time()
            privacy_findings = self.privacy_detector.detect(dataset_id, df)
            all_findings.extend(privacy_findings)
            stages.append(StageResult(
                name="privacy",
                status="successful",
                finding_count=len(privacy_findings),
                duration_ms=int((time.time() - t0) * 1000),
                message=f"Found {len(privacy_findings)} privacy findings.",
            ))

        # 2. Secrets Stage
        if request.enable_secrets:
            t0 = time.time()
            secret_findings = self.secret_detector.detect(dataset_id, df)
            all_findings.extend(secret_findings)
            stages.append(StageResult(
                name="secrets",
                status="successful",
                finding_count=len(secret_findings),
                duration_ms=int((time.time() - t0) * 1000),
                message=f"Found {len(secret_findings)} secret findings.",
            ))

        # 3. Security Stage
        if request.enable_security:
            t0 = time.time()
            sec_findings = self.security_analyzer.analyze(dataset_id, df)
            all_findings.extend(sec_findings)
            stages.append(StageResult(
                name="security",
                status="successful",
                finding_count=len(sec_findings),
                duration_ms=int((time.time() - t0) * 1000),
                message=f"Found {len(sec_findings)} security findings.",
            ))

        # Save initial findings
        self.repo.save_findings(dataset_id, all_findings)

        # 4. Quality Stage
        if request.enable_quality:
            t0 = time.time()
            prev_count = len(all_findings)
            self.profiler.profile_dataset(dataset_id)
            all_findings = self.repo.get_findings(dataset_id)
            qual_count = max(0, len(all_findings) - prev_count)
            stages.append(StageResult(
                name="quality",
                status="successful",
                finding_count=qual_count,
                duration_ms=int((time.time() - t0) * 1000),
                message=f"Found {qual_count} data quality findings.",
            ))

        # Calculate risk and policy
        risk = self.risk_engine.assess(dataset_id, all_findings)
        self.repo.save_risk(dataset_id, risk)

        policy = self.policy_engine.evaluate(dataset_id, all_findings, risk)
        self.repo.save_policy(dataset_id, policy)

        # Update dataset status
        self.repo.update_status(dataset_id, DatasetStatus.SCANNED)

        # Compute summary
        summary = FindingSummary(total=len(all_findings))
        by_cat: dict[str, int] = {}
        for f in all_findings:
            sev = f.get("severity", "")
            cat = f.get("category", "")
            by_cat[cat] = by_cat.get(cat, 0) + 1

            if sev == Severity.CRITICAL.value:
                summary.critical += 1
            elif sev == Severity.HIGH.value:
                summary.high += 1
            elif sev == Severity.MEDIUM.value:
                summary.medium += 1
            elif sev == Severity.LOW.value:
                summary.low += 1
            elif sev == Severity.INFORMATIONAL.value:
                summary.informational += 1

        summary.by_category = by_cat
        completed_at = utcnow()
        duration_ms = int((time.time() - start_time) * 1000)

        result = ScanResult(
            scan_id=scan_id,
            dataset_id=dataset_id,
            status=ScanStatus.COMPLETED,
            stages=stages,
            summary=summary,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
        )

        self.repo.save_scan(dataset_id, result.model_dump(mode="json"))
        return result
