from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import Settings, get_settings
from core.database.config import get_database
from core.exceptions import register_exception_handlers
from core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_logging()
    app.state.settings = settings

    db = get_database()
    await db.create_tables()

    yield

    await db.close()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Zemo - AI/ML Model Training & Experimentation Platform API",
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    from api.v1.health import router as health_router
    from api.v1.datasets.router import router as datasets_router
    from api.v1.integrations.router import router as integrations_router
    from api.v1.algorithms.router import router as algorithms_router
    from api.v1.experiments.router import router as experiments_router
    from api.v1.chatbot.router import router as chatbot_router

    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(datasets_router, prefix=settings.api_prefix)
    app.include_router(integrations_router, prefix=settings.api_prefix)
    app.include_router(algorithms_router, prefix=settings.api_prefix)
    app.include_router(experiments_router, prefix=f"{settings.api_prefix}/experiments")
    app.include_router(chatbot_router, prefix=f"{settings.api_prefix}/chatbot")

    # Mount frontend static files and root web app
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    workspace_root = Path(__file__).resolve().parent.parent
    if (workspace_root / "index.html").exists():
        if (workspace_root / "js").exists():
            app.mount("/js", StaticFiles(directory=str(workspace_root / "js")), name="frontend_js")
        if (workspace_root / "css").exists():
            app.mount("/css", StaticFiles(directory=str(workspace_root / "css")), name="frontend_css")
        if (workspace_root / "assets").exists():
            app.mount("/assets", StaticFiles(directory=str(workspace_root / "assets")), name="frontend_assets")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(str(workspace_root / "index.html"))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )