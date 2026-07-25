"""Alembic env.py —— async 配置。

从 vllmonline.config 读 DATABASE_URL，用 async engine 跑 migration。
target_metadata 指向 vllmonline.db.models.Base.metadata，支持 autogenerate。
"""

from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# 把项目根加入 sys.path（alembic 在子目录里跑）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllmonline.config import get_settings  # noqa: E402
from vllmonline.db.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：生成 SQL 脚本不连库。"""
    settings = get_settings()
    url = settings.database.url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """在线模式：用 async engine。"""
    settings = get_settings()
    connectable = async_engine_from_config(
        {"sqlalchemy.url": settings.database.url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
