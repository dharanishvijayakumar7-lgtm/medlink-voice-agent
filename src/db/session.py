"""Async engine and session management.

Persistence is optional. If ``MEDLINK_ENABLE_DB`` is off or ``DATABASE_URL`` is
unset, ``is_enabled()`` returns False and the repository degrades to no-ops - a
call must never fail because the database is unavailable.

The URL is normalised to an async driver so operators can paste an ordinary
``postgresql://`` connection string.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine as _create_async_engine

from config import settings
from db.models import Base

logger = logging.getLogger("medlink.db")


def normalise_url(url: str) -> str:
    """Accept plain sync URLs and upgrade them to the async driver."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("sqlite://") and "+aiosqlite" not in url:
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


def is_enabled() -> bool:
    return bool(settings.enable_db and settings.database_url)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    url = normalise_url(settings.database_url)
    logger.info("connecting to database (%s)", url.split("://", 1)[0])
    # Fail fast rather than leaving a live call waiting on a dead database.
    connect_args: dict = {}
    if url.startswith("postgresql+asyncpg"):
        connect_args["timeout"] = settings.db_connect_timeout
    return _create_async_engine(
        url, pool_pre_ping=True, future=True, connect_args=connect_args
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def create_all(engine: AsyncEngine | None = None) -> None:
    """Create tables directly. Used by tests and local dev; prod uses Alembic."""
    engine = engine or get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope. Commits on success, rolls back on error."""
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def reset_for_tests() -> None:
    """Drop cached engine/sessionmaker so a test can point at a new URL."""
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
