"""数据库 session 管理。

提供：
    - create_engine(settings) → AsyncEngine
    - create_session_factory(engine) → async_sessionmaker
    - get_session() FastAPI 依赖注入
    - init_db(engine) 建表（开发用，生产走 alembic migration）

SQLite 兼容：测试默认 SQLite in-memory，生产 PG。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from vllmonline.config import Settings, get_settings
from vllmonline.db.models import Base


def create_engine(settings: Settings | None = None) -> AsyncEngine:
    """根据 settings.database.url 创建 async engine。

    SQLite vs PG 行为差异：
        - SQLite：无连接池概念，echo 关掉
        - PG：用连接池（pool_size + max_overflow + pre_ping）
    """
    settings = settings or get_settings()
    db = settings.database

    kwargs: dict[str, Any] = {"echo": db.echo, "future": True}

    if db.is_sqlite:
        # SQLite in-memory 必须用 StaticPool 共享单连接，
        # 否则每个连接看到不同的 in-memory 数据库（默认是 per-connection）。
        from sqlalchemy.pool import StaticPool

        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
        return create_async_engine(db.url, **kwargs)

    # PostgreSQL：连接池
    kwargs["pool_size"] = db.pool_size
    kwargs["max_overflow"] = db.max_overflow
    kwargs["pool_pre_ping"] = db.pool_pre_ping
    return create_async_engine(db.url, **kwargs)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """构造 session 工厂。

    expire_on_commit=False：commit 后对象仍可访问属性（async 习惯）。
    """
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """开发/测试用：在 engine 上建所有表（Base.metadata.create_all）。

    生产环境不要调用本函数——用 alembic upgrade head。
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db(engine: AsyncEngine) -> None:
    """测试用：drop 所有表。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 依赖注入
# ─────────────────────────────────────────────────────────────────────────────

# 全局 engine + session factory（lifespan 启动时初始化）
# 测试时通过 monkeypatch 替换为独立 engine。
_global_engine: AsyncEngine | None = None
_global_session_factory: async_sessionmaker[AsyncSession] | None = None


def set_global_engine(engine: AsyncEngine) -> None:
    """lifespan 启动时调用：注册全局 engine + session factory。"""
    global _global_engine, _global_session_factory
    _global_engine = engine
    _global_session_factory = create_session_factory(engine)


def clear_global_engine() -> None:
    """lifespan 关闭时调用。"""
    global _global_engine, _global_session_factory
    _global_engine = None
    _global_session_factory = None


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：提供一个 AsyncSession，请求结束自动关闭。

    Usage in route:
        async def route(session: AsyncSession = Depends(get_session)):
            ...
    """
    if _global_session_factory is None:
        msg = "数据库未初始化（lifespan 未调用 set_global_engine）"
        raise RuntimeError(msg)
    async with _global_session_factory() as session:
        yield session
