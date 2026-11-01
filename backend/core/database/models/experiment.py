from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base


class ExperimentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (
        Index("ix_experiments_dataset_id", "dataset_id"),
        Index("ix_experiments_status", "status"),
        Index("ix_experiments_created_at", "created_at"),
        {"extend_existing": True},
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    dataset_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    task: Mapped[str] = mapped_column(String(50), nullable=False)  # classification, regression, clustering
    algorithm_id: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default=ExperimentStatus.PENDING.value, nullable=False)
    
    # Configuration and hyperparameters
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    hyperparameters: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    
    # Results and evaluation
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    artifact_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    model_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Execution logs and errors
    logs: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "dataset_id": self.dataset_id,
            "dataset_version_id": self.dataset_version_id,
            "task": self.task,
            "algorithm_id": self.algorithm_id,
            "status": self.status,
            "config": self.config,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "artifact_path": self.artifact_path,
            "model_size_bytes": self.model_size_bytes,
            "logs": self.logs,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
