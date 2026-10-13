"""Shared API envelope schemas for consistent error and success responses."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorResponse(BaseModel):
    """Standardised API error body."""
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class SuccessResponse(BaseModel):
    """Generic wrapper for successful non-collection responses."""
    status: str = "ok"
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    service: str
    storage_path: str
