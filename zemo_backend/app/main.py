"""
Zemo FastAPI application entrypoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_primary_lan_ip, get_settings
from app.core.exceptions import ZemoError
from app.core.logging import setup_logging, get_logger
from app.schemas.common import ErrorResponse, HealthResponse

logger = get_logger("main")
FRONTEND_DIR = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    setup_logging()
    settings = get_settings()
    lan_ip = get_primary_lan_ip()
    logger.info(
        "Zemo %s starting - host=%s port=%s debug=%s storage=%s",
        settings.app_version,
        settings.host,
        settings.port,
        settings.debug,
        settings.storage_path,
    )
    logger.info(
        "Access URLs - Local: http://localhost:%s | LAN: http://%s:%s",
        settings.port,
        lan_ip,
        settings.port,
    )
    yield
    logger.info("Zemo shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI Data Privacy, Security & Training Safety Platform",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.allowed_origin_regex or None,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ─────────────────────────────────
    @app.exception_handler(ZemoError)
    async def zemo_error_handler(request: Request, exc: ZemoError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                code=exc.code,
                message=exc.message,
                detail=exc.detail,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred.",
            ).model_dump(),
        )

    # ── Health endpoint ──────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            service=settings.app_name,
            storage_path=str(settings.storage_path),
        )

    # ── API routes ───────────────────────────────────────────────
    from app.api.routes import datasets, models, readiness, sanitization, scans, workflow

    app.include_router(datasets.router, prefix="/api/v1", tags=["datasets"])
    app.include_router(scans.router, prefix="/api/v1", tags=["scans"])
    app.include_router(sanitization.router, prefix="/api/v1", tags=["sanitization"])
    app.include_router(readiness.router, prefix="/api/v1", tags=["readiness"])
    app.include_router(models.router, prefix="/api/v1", tags=["models"])
    app.include_router(workflow.router, prefix="/api/v1", tags=["workflow"])

    # ── Frontend Static Files & SPA Entrypoint ───────────────────
    css_dir = FRONTEND_DIR / "css"
    js_dir = FRONTEND_DIR / "js"
    assets_dir = FRONTEND_DIR / "assets"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/inter.woff2", include_in_schema=False)
    async def serve_inter_font() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "inter.woff2", media_type="font/woff2")

    @app.get("/instrument-serif-italic.woff2", include_in_schema=False)
    async def serve_serif_font() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "instrument-serif-italic.woff2", media_type="font/woff2")

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=_settings.host,
        port=_settings.port,
        reload=_settings.debug,
    )


