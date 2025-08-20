"""
PostgreSQL database connection and session management.
Handles database initialization, connection pooling, and session lifecycle.
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from app.utils.config import get_database_url
from app.db.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections, sessions, and initialization.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            database_url: Database connection URL. If None, uses config.
        """
        self.database_url = database_url or get_database_url()
        self.engine: Optional[AsyncEngine] = None
        self.async_session: Optional[async_sessionmaker[AsyncSession]] = None
        
    async def initialize(self) -> None:
        """
        Initialize the database engine and session factory.
        """
        try:
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                self.database_url,
                echo=False,  # Set to True for SQL query logging
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=300,  # Recycle connections every 5 minutes
                poolclass=NullPool if "sqlite" in self.database_url else None,
            )
            
            # Create session factory
            self.async_session = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=True,
            )
            
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def create_tables(self) -> None:
        """
        Create all database tables.
        """
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def drop_tables(self) -> None:
        """
        Drop all database tables. Use with caution!
        """
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
            
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    async def health_check(self) -> bool:
        """
        Check if the database connection is healthy.
        
        Returns:
            True if database is accessible, False otherwise.
        """
        if not self.engine:
            return False
        
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with automatic cleanup.
        
        Yields:
            AsyncSession: Database session
        """
        if not self.async_session:
            raise RuntimeError("Database session factory not initialized")
        
        session = self.async_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def close(self) -> None:
        """
        Close the database engine and all connections.
        """
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


async def init_database() -> None:
    """
    Initialize the database connection and create tables.
    """
    await db_manager.initialize()
    await db_manager.create_tables()


async def close_database() -> None:
    """
    Close database connections.
    """
    await db_manager.close()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function to get database session for FastAPI.
    
    Yields:
        AsyncSession: Database session
    """
    async with db_manager.get_session() as session:
        yield session


async def get_db_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.
    
    Yields:
        AsyncSession: Database session
    """
    async with db_manager.get_session() as session:
        yield session


async def health_check_db() -> bool:
    """
    Check database health.
    
    Returns:
        True if database is healthy, False otherwise.
    """
    return await db_manager.health_check()


class DatabaseRepository:
    """
    Base repository class with common database operations.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Database session
        """
        self.session = session
    
    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()
    
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.session.rollback()
    
    async def refresh(self, instance) -> None:
        """Refresh an instance from the database."""
        await self.session.refresh(instance)
    
    async def flush(self) -> None:
        """Flush pending changes to the database."""
        await self.session.flush()


# Utility functions for common database operations

async def execute_raw_query(query: str, params: Optional[dict] = None) -> list:
    """
    Execute a raw SQL query.
    
    Args:
        query: SQL query string
        params: Query parameters
        
    Returns:
        Query results as a list
    """
    async with db_manager.get_session() as session:
        result = await session.execute(text(query), params or {})
        return result.fetchall()


async def get_table_count(table_name: str) -> int:
    """
    Get the number of rows in a table.
    
    Args:
        table_name: Name of the table
        
    Returns:
        Number of rows in the table
    """
    query = f"SELECT COUNT(*) FROM {table_name}"
    result = await execute_raw_query(query)
    return result[0][0] if result else 0