from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base

if TYPE_CHECKING:
    from core.database.models.dataset import Dataset, DatasetProfile, DatasetValidationResult, Scan, Finding


class DatasetVersionStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    INVALID = "INVALID"
    PROFILING = "PROFILING"
    PROFILED = "PROFILED"
    FAILED = "FAILED"


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    format: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[DatasetVersionStatus] = mapped_column(Enum(DatasetVersionStatus), default=DatasetVersionStatus.UPLOADED, nullable=False, index=True)
    validation_status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    profiling_status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    record_count: Mapped["int | None"] = mapped_column(Integer, nullable=True)
    column_count: Mapped["int | None"] = mapped_column(Integer, nullable=True)
    schema_metadata: Mapped["dict | None"] = mapped_column(JSON, nullable=True)
    profile_metadata: Mapped["dict | None"] = mapped_column(JSON, nullable=True)
    error_summary: Mapped["str | None"] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="versions")
    profiles: Mapped[list["DatasetProfile"]] = relationship(
        "DatasetProfile",
        backref="version",
        cascade="all, delete-orphan",
    )
    validation_results: Mapped[list["DatasetValidationResult"]] = relationship(
        "DatasetValidationResult",
        backref="version",
        cascade="all, delete-orphan",
    )
    scans: Mapped[list["Scan"]] = relationship(
        "Scan",
        backref="version",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("dataset_id", "version_number", name="uq_dataset_version_number"),
        Index("ix_dataset_versions_dataset_id_created_at", "dataset_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<DatasetVersion(id={self.id}, dataset_id={self.dataset_id}, version={self.version_number}, status={self.status})>"