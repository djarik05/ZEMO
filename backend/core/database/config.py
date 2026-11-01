import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import get_settings
from core.database.base import Base

from core.database.models.dataset import Dataset  # noqa: F401
from core.database.models.dataset_version import DatasetVersion  # noqa: F401


class Database:
    def __init__(self, database_url: str, echo: bool = False):
        self.engine: AsyncEngine = create_async_engine(database_url, echo=echo)
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def create_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def check_connectivity(self) -> bool:
        """Check if database is reachable."""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


def create_database(database_url: str, echo: bool = False) -> Database:
    return Database(database_url, echo=echo)


_db: Database | None = None


def get_database() -> Database:
    global _db
    if _db is None:
        settings = get_settings()
        # Use in-memory SQLite for test environment only if no explicit DATABASE_URL is set
        if settings.environment == "test" and not os.environ.get("DATABASE_URL"):
            database_url = "sqlite+aiosqlite:///:memory:"
            echo = False
        else:
            database_url = settings.database_url
            echo = settings.debug if settings.environment != "test" else False
        _db = Database(database_url, echo=echo)
    return _db


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    db = get_database()
    async with db.session() as session:
        yield session