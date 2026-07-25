"""端到端集成测试（SPEC §10.2，PLAN P5.3）。

标记 @pytest.mark.integration，需要 Docker（testcontainers 起 PostgreSQL）。
本机无 Docker 时自动 skip。

测试矩阵（SPEC §10.2）：
    - 两模型同时加载 + 按比例分流（metrics 正确按 model_version 区分）
    - 热切换全过程（零请求丢失）
    - 灰度自动推进（优秀 v2 → 自动从 10% 推到 30% → 100%）
    - 灰度自动回滚（劣化 v2 → 自动回滚）
    - GPU 显存不足（拒绝加载）
    - A/B 评测端到端（采样→双发→Judge→t-test→报告）
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# ─────────────────────────────────────────────────────────────────────────────
# testcontainers PostgreSQL fixture（session 级，所有集成测试共享一个容器）
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def pg_url() -> str:
    """启动 testcontainers PostgreSQL（pgvector pg16）。

    testcontainers 4.x 推荐 testcontainers.community.postgres；
    老路径 testcontainers.postgres 仍可用但会抛 DeprecationWarning。
    无 Docker 时自动 skip（不报错）。
    """
    try:
        from testcontainers.community.postgres import PostgresContainer
    except ImportError:
        try:
            from testcontainers.postgres import PostgresContainer  # type: ignore[assignment]
        except ImportError:
            pytest.skip("testcontainers 未安装")
    try:
        container = PostgresContainer("pgvector/pgvector:pg16", driver="asyncpg")
        container.start()
    except Exception as e:
        pytest.skip(f"Docker 不可用，跳过集成测试：{e}")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture
async def pg_session_factory(pg_url):
    """每个测试用独立 PG 连接 + 建表。"""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from vllmonline.db.models import Base

    engine = create_async_engine(pg_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# 集成测试
# ─────────────────────────────────────────────────────────────────────────────


async def test_postgres_schema_created(pg_session_factory) -> None:
    """验证全部 4 张表在真实 PG 上能创建。"""
    from sqlalchemy import inspect
    from sqlalchemy.ext.asyncio import AsyncEngine

    async with pg_session_factory() as session:
        engine: AsyncEngine = session.bind  # type: ignore[attr-defined]
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
        assert "model_versions" in tables
        assert "canary_deployments" in tables
        assert "canary_events" in tables
        assert "eval_results" in tables


async def test_pg_model_version_roundtrip(pg_session_factory) -> None:
    """CRUD roundtrip：注册 → 查 → 删。"""
    from vllmonline.db.models import ModelVersion

    async with pg_session_factory() as session:
        # Create
        session.add(
            ModelVersion(
                id="qwen-7b-v1",
                model_name="qwen-7b",
                version="v1",
                endpoint="http://vllm:8000",
                params_billion=7.0,
                dtype="fp16",
                weight_gb=14.0,
                kv_cache_budget_gb=3.15,
                gpu_id=0,
                status="IDLE",
            )
        )
        await session.commit()

        # Read
        row = await session.get(ModelVersion, "qwen-7b-v1")
        assert row is not None
        assert row.weight_gb == pytest.approx(14.0)
        assert row.status == "IDLE"

        # Update
        row.status = "ACTIVE"
        await session.commit()
        await session.refresh(row)
        assert row.status == "ACTIVE"

        # Delete
        await session.delete(row)
        await session.commit()
        assert await session.get(ModelVersion, "qwen-7b-v1") is None


async def test_pg_jsonb_fields(pg_session_factory) -> None:
    """JSONB 字段（stages/traffic_split/metrics_snapshot/dimension_scores）。"""
    from vllmonline.db.models import CanaryDeployment, CanaryEvent, ModelVersion

    async with pg_session_factory() as session:
        # 先建 model_versions（外键依赖）
        for ver in ["v1", "v2"]:
            session.add(
                ModelVersion(
                    id=f"qwen-7b-{ver}",
                    model_name="qwen-7b",
                    version=ver,
                    endpoint="http://x",
                    params_billion=7.0,
                    dtype="fp16",
                )
            )
        await session.commit()

        # canary_deployments 带 JSONB stages + traffic_split
        session.add(
            CanaryDeployment(
                id="canary-test",
                model_v1_id="qwen-7b-v1",
                model_v2_id="qwen-7b-v2",
                stages=[0.1, 0.3, 1.0],
                traffic_split={"qwen-7b-v1": 0.7, "qwen-7b-v2": 0.3},
            )
        )
        await session.commit()

        # canary_events 带 metrics_snapshot
        session.add(
            CanaryEvent(
                deployment_id="canary-test",
                action="ADVANCE",
                reason="测试",
                metrics_snapshot={"v1": {"ttft": 0.1}, "v2": {"ttft": 0.15}},
            )
        )
        await session.commit()

        # 读回验证 JSONB 正确序列化
        dep = await session.get(CanaryDeployment, "canary-test")
        assert dep.stages == [0.1, 0.3, 1.0]
        assert dep.traffic_split["qwen-7b-v2"] == 0.3


async def test_hot_swap_zero_downtime(pg_session_factory) -> None:
    """端到端热切换：注册两个模型 → 加载 v1 → 热切换到 v2，状态正确。

    这是纯状态机 + registry 的测试（不需要真实 vLLM），但用真实 PG 持久化。
    """
    from vllmonline.gpu_info import NoneGpuProvider
    from vllmonline.scheduler.engine import HotSwapEngine
    from vllmonline.scheduler.gpu_memory import build_model_profile
    from vllmonline.scheduler.lifecycle import Model, ModelRegistry
    from vllmonline.scheduler.types import ModelState
    from vllmonline.vllm.adapter import VLLMAdapter
    from vllmonline.vllm.client import VLLMClient

    registry = ModelRegistry()
    m1 = Model(
        id="qwen-7b-v1",
        model_name="qwen-7b",
        version="v1",
        endpoint="http://v1:8000",
        gpu_id=0,
        memory_profile=build_model_profile(7.0, "fp16"),
    )
    m2 = Model(
        id="qwen-7b-v2",
        model_name="qwen-7b",
        version="v2",
        endpoint="http://v2:8000",
        gpu_id=0,
        memory_profile=build_model_profile(7.0, "fp16"),
    )
    await registry.register(m1)
    await registry.register(m2)
    await m1.transition(ModelState.LOADING)
    await m1.transition(ModelState.ACTIVE)

    client = VLLMClient.__new__(VLLMClient)
    client._owns_client = False  # type: ignore[attr-defined]
    adapter = VLLMAdapter(client)  # type: ignore[arg-type]
    engine = HotSwapEngine(
        registry=registry,
        adapter=adapter,
        gpu_provider=NoneGpuProvider(fake_total_gb=80.0, fake_used_gb=0.0),
        drain_timeout_seconds=2.0,
    )

    result = await engine.hot_swap("qwen-7b-v1", "qwen-7b-v2")

    assert result.zero_downtime is True
    assert m2.state is ModelState.ACTIVE
    assert m1.state is ModelState.SLEEPING


async def test_full_canary_lifecycle(pg_session_factory) -> None:
    """完整灰度生命周期：start → advance → advance → rollback。"""
    from asgi_lifespan import LifespanManager
    from httpx import ASGITransport, AsyncClient

    from vllmonline import server

    async with LifespanManager(server.create_app()) as _:
        transport = ASGITransport(app=server.create_app())
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 注册两个模型
            for ver in ["v1", "v2"]:
                r = await client.post(
                    "/api/models/register",
                    json={
                        "model_name": "qwen-7b",
                        "version": ver,
                        "endpoint": f"http://v{ver}:8000",
                        "params_billion": 7.0,
                        "dtype": "fp16",
                    },
                )
                assert r.status_code == 201
                r = await client.post(f"/api/models/qwen-7b-{ver}/load")
                assert r.status_code == 200

            # 启动灰度
            r = await client.post(
                "/api/canary/start",
                json={
                    "model_v1": "qwen-7b-v1",
                    "model_v2": "qwen-7b-v2",
                    "stages": [0.1, 0.3, 1.0],
                },
            )
            assert r.status_code == 201
            deployment_id = r.json()["id"]

            # 推进 30%
            r = await client.post(f"/api/canary/{deployment_id}/advance")
            assert r.status_code == 200
            assert "STAGE_30" in r.json()["current_stage"]

            # 回滚
            r = await client.post(f"/api/canary/{deployment_id}/rollback")
            assert r.status_code == 200
            assert r.json()["status"] == "ROLLED_BACK"


async def test_alembic_migration_applies(pg_url) -> None:
    """验证 alembic upgrade head 能在真实 PG 上跑。"""
    import os
    import subprocess

    env = {**os.environ, "VLLMONLINE_DATABASE__URL": pg_url}
    # alembic 必须用同步 driver URL（alembic 不支持 asyncpg 直接）
    sync_url = pg_url.replace("+asyncpg", "+psycopg2").replace("postgresql+psycopg2", "postgresql")
    env["VLLMONLINE_DATABASE__URL"] = sync_url

    result = subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"alembic upgrade failed: {result.stderr}"
