import tempfile
import os
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from core.database.base import Base
from core.database.models.integration import Provider, ModelCapabilityInfo, ModelIntegration, AgentIntegration, DatasetAccessGrant
from core.database.models.dataset import Dataset, DatasetProfile
from core.database.models.dataset_version import DatasetVersion

async def test():
    test_db_fd, test_db_path = tempfile.mkstemp(suffix='.db')
    os.close(test_db_fd)
    test_db_url = f"sqlite+aiosqlite:///{test_db_path}"
    
    engine = create_async_engine(test_db_url)
    
    # Check database before create_all
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = result.fetchall()
        print(f"Tables before create_all: {tables}")
        
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='index'"))
        indexes = result.fetchall()
        print(f"Indexes before create_all: {indexes}")
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully")
    
    # Check if tables exist
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = result.fetchall()
        print(f"Tables after create_all: {tables}")
        
        # Check indexes
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='index'"))
        indexes = result.fetchall()
        print(f"Indexes after create_all: {indexes}")
    
    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    print("Tables dropped successfully")
    
    # Check if tables exist after drop
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = result.fetchall()
        print(f"Tables after drop: {tables}")
        
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='index'"))
        indexes = result.fetchall()
        print(f"Indexes after drop: {indexes}")
    
    # Create tables again
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created again successfully")
    
    # Check if tables exist
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = result.fetchall()
        print(f"Tables after second create_all: {tables}")
    
    await engine.dispose()
    os.unlink(test_db_path)

asyncio.run(test())