"""
Repository abstraction for all Zemo persistence.

Uses local JSON file storage behind a clean interface so a future
database implementation can be swapped without touching service code.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import safe_join, generate_storage_id
from app.models.enums import DatasetStatus, utcnow

logger = get_logger("repository")

# ── Helpers ──────────────────────────────────────────────────────

def _meta_path(base: Path, dataset_id: str) -> Path:
    return safe_join(base, "datasets", dataset_id, "metadata.json")


def _data_path(base: Path, dataset_id: str) -> Path:
    return safe_join(base, "datasets", dataset_id, "data.parquet")


def _raw_path(base: Path, dataset_id: str, filename: str) -> Path:
    return safe_join(base, "datasets", dataset_id, filename)


def _findings_path(base: Path, dataset_id: str) -> Path:
    return safe_join(base, "datasets", dataset_id, "findings.json")


def _scan_path(base: Path, dataset_id: str) -> Path:
    return safe_join(base, "datasets", dataset_id, "scan.json")


def _risk_path(base: Path, dataset_id: str) -> Path:
    return safe_join(base, "datasets", dataset_id, "risk.json")


def _policy_path(base: Path, dataset_id: str) -> Path:
    return safe_join(base, "datasets", dataset_id, "policy.json")


def _readiness_path(base: Path, dataset_id: str) -> Path:
    return safe_join(base, "datasets", dataset_id, "readiness.json")


def _sanitization_path(base: Path, dataset_id: str) -> Path:
    return safe_join(base, "datasets", dataset_id, "sanitization.json")


def _model_config_path(base: Path) -> Path:
    return safe_join(base, "config", "model_config.json")


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# ── Dataset Repository ──────────────────────────────────────────

class DatasetRepository:
    """Filesystem-backed dataset repository."""

    def __init__(self) -> None:
        pass

    @property
    def _base(self) -> Path:
        return get_settings().storage_path

    # ── Write ────────────────────────────────────────────────────

    def save_metadata(self, meta: dict[str, Any]) -> None:
        path = _meta_path(self._base, meta["dataset_id"])
        _ensure_dir(path)
        meta["updated_at"] = utcnow().isoformat()
        path.write_text(json.dumps(meta, default=str), encoding="utf-8")

    def save_dataframe(self, dataset_id: str, df: pd.DataFrame) -> None:
        path = _data_path(self._base, dataset_id)
        _ensure_dir(path)
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved dataframe for %s (%d rows)", dataset_id, len(df))

    def save_raw_file(self, dataset_id: str, filename: str, content: bytes) -> Path:
        path = _raw_path(self._base, dataset_id, filename)
        _ensure_dir(path)
        path.write_bytes(content)
        return path

    def update_status(self, dataset_id: str, status: DatasetStatus) -> None:
        meta = self.get_metadata(dataset_id)
        if meta is None:
            return
        meta["status"] = status.value
        self.save_metadata(meta)

    # ── Read ─────────────────────────────────────────────────────

    def get_metadata(self, dataset_id: str) -> dict[str, Any] | None:
        path = _meta_path(self._base, dataset_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def get_dataframe(self, dataset_id: str) -> pd.DataFrame | None:
        path = _data_path(self._base, dataset_id)
        if not path.exists():
            return None
        return pd.read_parquet(path, engine="pyarrow")

    def list_datasets(self) -> list[dict[str, Any]]:
        datasets_dir = self._base / "datasets"
        if not datasets_dir.exists():
            return []
        results = []
        for child in datasets_dir.iterdir():
            if child.is_dir():
                meta = self.get_metadata(child.name)
                if meta:
                    results.append(meta)
        results.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return results

    def exists(self, dataset_id: str) -> bool:
        return _meta_path(self._base, dataset_id).exists()

    def delete(self, dataset_id: str) -> bool:
        d = safe_join(self._base, "datasets", dataset_id)
        if d.exists():
            shutil.rmtree(d)
            return True
        return False

    # ── Findings ─────────────────────────────────────────────────

    def save_findings(self, dataset_id: str, findings: list[dict[str, Any]]) -> None:
        path = _findings_path(self._base, dataset_id)
        _ensure_dir(path)
        path.write_text(json.dumps(findings, default=str), encoding="utf-8")

    def get_findings(self, dataset_id: str) -> list[dict[str, Any]]:
        path = _findings_path(self._base, dataset_id)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    # ── Scan ─────────────────────────────────────────────────────

    def save_scan(self, dataset_id: str, scan: dict[str, Any]) -> None:
        path = _scan_path(self._base, dataset_id)
        _ensure_dir(path)
        path.write_text(json.dumps(scan, default=str), encoding="utf-8")

    def get_scan(self, dataset_id: str) -> dict[str, Any] | None:
        path = _scan_path(self._base, dataset_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ── Risk ─────────────────────────────────────────────────────

    def save_risk(self, dataset_id: str, risk: dict[str, Any]) -> None:
        path = _risk_path(self._base, dataset_id)
        _ensure_dir(path)
        path.write_text(json.dumps(risk, default=str), encoding="utf-8")

    def get_risk(self, dataset_id: str) -> dict[str, Any] | None:
        path = _risk_path(self._base, dataset_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ── Policy ───────────────────────────────────────────────────

    def save_policy(self, dataset_id: str, policy: dict[str, Any]) -> None:
        path = _policy_path(self._base, dataset_id)
        _ensure_dir(path)
        path.write_text(json.dumps(policy, default=str), encoding="utf-8")

    def get_policy(self, dataset_id: str) -> dict[str, Any] | None:
        path = _policy_path(self._base, dataset_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ── Readiness ────────────────────────────────────────────────

    def save_readiness(self, dataset_id: str, report: dict[str, Any]) -> None:
        path = _readiness_path(self._base, dataset_id)
        _ensure_dir(path)
        path.write_text(json.dumps(report, default=str), encoding="utf-8")

    def get_readiness(self, dataset_id: str) -> dict[str, Any] | None:
        path = _readiness_path(self._base, dataset_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ── Sanitization ─────────────────────────────────────────────

    def save_sanitization(self, dataset_id: str, result: dict[str, Any]) -> None:
        path = _sanitization_path(self._base, dataset_id)
        _ensure_dir(path)
        path.write_text(json.dumps(result, default=str), encoding="utf-8")

    def get_sanitization(self, dataset_id: str) -> dict[str, Any] | None:
        path = _sanitization_path(self._base, dataset_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ── Target Model Configuration ───────────────────────────────

    def save_model_config(self, config: dict[str, Any]) -> None:
        path = _model_config_path(self._base)
        _ensure_dir(path)
        config["updated_at"] = utcnow().isoformat()
        path.write_text(json.dumps(config, default=str), encoding="utf-8")

    def get_model_config(self) -> dict[str, Any] | None:
        path = _model_config_path(self._base)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

