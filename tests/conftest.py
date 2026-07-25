"""pytest 全局 fixtures。

提供：
    - fake vLLM app + ASGITransport httpx client（零网络、零端口）
    - 测试用 vllmonline app（指向 fake vLLM）
    - in-memory SQLite engine + session（单测 DB）
    - testcontainers PostgreSQL（标记 integration 用）

测试约定：
    - 默认测试不依赖 Docker / GPU / 网络
    - 集成测试加 `@pytest.mark.integration`，需要 Docker 起 testcontainers
    - 用 asyncio_mode=auto（pyproject.toml 配置），无需 @pytest.mark.asyncio
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

# 在导入 vllmonline 之前设置环境，避免首次 get_settings() 读到真实 env
os.environ.setdefault("VLLMONLINE_ENVIRONMENT", "test")
os.environ.setdefault("VLLMONLINE_DATABASE__URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("VLLMONLINE_REDIS__ENABLED", "false")
os.environ.setdefault("VLLMONLINE_LOGGING__LEVEL", "WARNING")

from tests.fake_vllm import make_fake_vllm_app


# autouse：每个测试前清理全局单例，避免跨文件污染。
# P2 起引入了多个模块级全局（routing_manager / registry / engine），
# 不清理会让前一个测试的状态影响后一个。
@pytest.fixture(autouse=True)
def _reset_global_singletons() -> Any:
    """重置所有模块级全局单例（autouse，每个测试自动跑）。"""
    import vllmonline.api.routes as api_routes
    import vllmonline.db.session as db_session
    import vllmonline.router.proxy as proxy_mod

    api_routes._global_registry = None  # type: ignore[attr-defined]
    db_session._global_engine = None  # type: ignore[attr-defined]
    db_session._global_session_factory = None  # type: ignore[attr-defined]
    proxy_mod._global_routing_manager = None  # type: ignore[attr-defined]
    yield


# ─────────────────────────────────────────────────────────────────────────────
# Fake vLLM backend
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def fake_vllm_app() -> Any:
    """默认配置的 fake vLLM FastAPI app。"""
    return make_fake_vllm_app()


@pytest.fixture
async def fake_vllm_client(fake_vllm_app: Any) -> AsyncIterator[AsyncClient]:
    """指向 fake vLLM 的 httpx client（ASGITransport，零网络）。"""
    transport = ASGITransport(app=fake_vllm_app)
    async with AsyncClient(transport=transport, base_url="http://fake-vllm") as ac:
        yield ac


# ─────────────────────────────────────────────────────────────────────────────
# vllmonline app + client（指向 fake vLLM）
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def vllmonline_app(fake_vllm_app: Any) -> Any:
    """构造一个 vllmonline FastAPI app，其 backend 指向 fake vLLM。

    通过 server.set_test_backend() 注入一个指向 fake vLLM 的 httpx client 工厂。
    lifespan 启动时会读这个 hook，构建零网络的 ASGITransport client。
    """
    from vllmonline import server
    from vllmonline.config import get_settings

    # 重置 settings 缓存，确保读到测试环境变量
    get_settings.cache_clear()

    # 注入测试 backend：每次 lifespan 调工厂时构造一个新的指向 fake 的 client
    def _factory() -> AsyncClient:
        return AsyncClient(
            transport=ASGITransport(app=fake_vllm_app),
            base_url="http://fake-vllm",
        )

    server.set_test_backend(client_factory=_factory, backend_url="http://fake-vllm")
    try:
        yield server.create_app()
    finally:
        server.clear_test_backend()
        # 全局单例的清理由 autouse fixture _reset_global_singletons 统一负责


@pytest.fixture
async def client(vllmonline_app: Any) -> AsyncIterator[AsyncClient]:
    """vllmonline 自身的测试 client（通过 ASGITransport，零网络）。

    用 asgi-lifespan 的 LifespanManager 显式驱动 FastAPI 的 lifespan 启动/关闭，
    因为 httpx.ASGITransport 默认不发 lifespan 事件（app.state.vllmonline 不会被设置）。
    """
    from asgi_lifespan import LifespanManager

    async with LifespanManager(vllmonline_app):
        transport = ASGITransport(app=vllmonline_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ─────────────────────────────────────────────────────────────────────────────
# 数据库 fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def sqlite_engine():
    """in-memory SQLite async engine（单测默认 DB）。"""
    from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

    engine: AsyncEngine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    yield engine
    await engine.dispose()


@pytest.fixture
async def sqlite_db(sqlite_engine):
    """建表 + 提供 AsyncSession 工厂的 fixture。"""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    # 导入所有 model 让 Base.metadata 注册（P2 实现后启用）
    # from vllmonline.db.models import Base
    # async with sqlite_engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(sqlite_engine, class_=AsyncSession, expire_on_commit=False)
    yield Session


# ─────────────────────────────────────────────────────────────────────────────
# testcontainers PostgreSQL（仅 integration 测试用）
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def postgres_url() -> str | None:
    """session 级 fixture：启动一个 testcontainers PostgreSQL 容器。

    仅在 `pytest -m integration` 时实际加载；普通测试不触发。
    通过 lazy + skip 实现：测试内部用 `request.getfixturevalue("postgres_url")`。
    """
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers 未安装；用 `uv sync --group test` 安装")

    # pgvector image（SPEC §11.1 用 pgvector/pgvector:pg16）
    with PostgresContainer("pgvector/pgvector:pg16", driver="asyncpg") as pg:
        yield pg.get_connection_url()
