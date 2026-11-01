from contextlib import asynccontextmanager
from typing import AsyncGenerator

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


def create_database(database_url: str, echo: bool = False) -> Database:
    return Database(database_url, echo=echo)


_test_db: Database | None = None
_test_database_url: str | None = None


def set_test_database_url(database_url: str) -> None:
    """Set the test database URL. Must be called before get_database()."""
    global _test_database_url, _test_db
    _test_database_url = database_url
    _test_db = None  # Reset to force recreation with new URL


def get_database() -> Database:
    global _test_db
    if _test_db is None:
        if _test_database_url is None:
            settings = get_settings()
            _test_database_url = settings.database_url
        _test_db = Database(_test_database_url, echo=False)
    return _test_db


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    db = get_database()
    async with db.session() as session:
        yield session