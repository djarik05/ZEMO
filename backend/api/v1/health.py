from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from core.config import Settings, get_settings
from core.database.config import get_database


class HealthResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    app_name: str
    status: str
    api_version: str
    environment: str


class LivenessResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: str


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: str
    database: str
    database_connected: bool


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness check - returns basic application health."""
    return HealthResponse(
        app_name=settings.app_name,
        status="healthy",
        api_version="v1",
        environment=settings.environment,
    )


@router.get("/health/live", response_model=LivenessResponse)
async def liveness_check() -> LivenessResponse:
    """Liveness probe - returns 200 if the application process is running."""
    return LivenessResponse(status="alive")


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(settings: Settings = Depends(get_settings)) -> ReadinessResponse:
    """Readiness probe - returns 200 if the application and its dependencies are ready."""
    db = get_database()
    db_connected = await db.check_connectivity()
    
    if not db_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed"
        )
    
    return ReadinessResponse(
        status="ready",
        database="connected" if db_connected else "disconnected",
        database_connected=db_connected,
    )