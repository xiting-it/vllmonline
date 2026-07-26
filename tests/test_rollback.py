"""canary/rollback.py 测试（SPEC §5.6）。"""

from __future__ import annotations

import asyncio

import pytest

from vllmonline.canary.rollback import (
    RollbackChecker,
    RollbackDecision,
    RollbackExecutor,
    RollbackSupervisor,
    check_and_rollback,
)
from vllmonline.canary.strategy import VersionMetrics
from vllmonline.router.proxy import RoutingTableManager
from vllmonline.scheduler.gpu_memory import build_model_profile
from vllmonline.scheduler.lifecycle import Model, ModelRegistry
from vllmonline.scheduler.types import ModelState


def _metrics(
    ttft_p50: float = 0.1,
    ttft_p99: float = 0.4,
    tpot_mean: float = 0.02,
    throughput: float = 100.0,
    error_rate: float = 0.001,
) -> VersionMetrics:
    return VersionMetrics(
        ttft_p50=ttft_p50,
        ttft_p99=ttft_p99,
        tpot_mean=tpot_mean,
        throughput=throughput,
        error_rate=error_rate,
    )


def _model(model_id: str) -> Model:
    name, ver = model_id.rsplit("-", 1)
    return Model(
        id=model_id,
        model_name=name,
        version=ver,
        endpoint=f"http://{model_id}:8000",
        gpu_id=0,
        memory_profile=build_model_profile(7.0, "fp16"),
    )


async def _activate(m: Model) -> None:
    await m.transition(ModelState.LOADING)
    await m.transition(ModelState.ACTIVE)


# ─────────────────────────────────────────────────────────────────────────────
# RollbackChecker
# ─────────────────────────────────────────────────────────────────────────────


class TestRollbackChecker:
    async def test_no_degradation_no_rollback(self) -> None:
        checker = RollbackChecker()
        decision = checker.check(_metrics(), _metrics())
        assert decision.should_rollback is False

    async def test_ttft_degradation_triggers_rollback(self) -> None:
        checker = RollbackChecker()
        v1 = _metrics(ttft_p99=0.40)
        v2 = _metrics(ttft_p99=0.80)  # +100%
        decision = checker.check(v1, v2)
        assert decision.should_rollback is True
        assert any("TTFT P99" in m for m in decision.degraded_metrics)

    async def test_error_rate_spike_triggers(self) -> None:
        checker = RollbackChecker()
        v1 = _metrics(error_rate=0.005)
        v2 = _metrics(error_rate=0.05)  # +4.5%
        decision = checker.check(v1, v2)
        assert decision.should_rollback is True
        assert any("错误率" in m for m in decision.degraded_metrics)

    async def test_multiple_degradations_listed(self) -> None:
        checker = RollbackChecker()
        v1 = _metrics(ttft_p50=0.1, ttft_p99=0.4, throughput=100.0)
        v2 = _metrics(ttft_p50=0.5, ttft_p99=0.8, throughput=50.0)
        decision = checker.check(v1, v2)
        assert decision.should_rollback is True
        assert len(decision.degraded_metrics) >= 3

    async def test_missing_metrics_no_rollback(self) -> None:
        """None 的指标不触发回滚（数据不足）。"""
        checker = RollbackChecker()
        v1 = VersionMetrics()
        v2 = VersionMetrics()
        decision = checker.check(v1, v2)
        assert decision.should_rollback is False

    async def test_decision_frozen(self) -> None:
        d = RollbackDecision(should_rollback=False, reason="ok", degraded_metrics=[])
        with pytest.raises(AttributeError):
            d.should_rollback = True  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# RollbackExecutor
# ─────────────────────────────────────────────────────────────────────────────


class TestRollbackExecutor:
    async def test_execute_rollback_drains_v2_and_routes_to_v1(self) -> None:
        registry = ModelRegistry()
        m1, m2 = _model("qwen-7b-v1"), _model("qwen-7b-v2")
        await registry.register(m1)
        await registry.register(m2)
        await _activate(m1)
        await _activate(m2)

        routing = RoutingTableManager(registry)
        await routing.set_split({"qwen-7b-v1": 0.7, "qwen-7b-v2": 0.3})

        executor = RollbackExecutor(registry, routing)

        # 构造一个简单的 deployment-like 对象
        class _FakeDeployment:
            id = "canary-test"
            status = "IN_PROGRESS"

        await executor.execute_rollback(
            deployment=_FakeDeployment(),  # type: ignore[arg-type]
            v1_id="qwen-7b-v1",
            v2_id="qwen-7b-v2",
            reason="测试回滚",
        )

        # v2 应该被 sleep（不再 ACTIVE）
        assert m2.state is ModelState.SLEEPING
        # 流量 100% 到 v1
        split = routing.get_traffic_split("qwen-7b")
        assert split.get("qwen-7b-v1", 0) == pytest.approx(1.0)

    async def test_execute_rollback_when_v2_not_active(self) -> None:
        """v2 已经不是 ACTIVE 时回滚不抛错。"""
        registry = ModelRegistry()
        m1, m2 = _model("qwen-7b-v1"), _model("qwen-7b-v2")
        await registry.register(m1)
        await registry.register(m2)
        await _activate(m1)
        # m2 保持 IDLE
        routing = RoutingTableManager(registry)
        executor = RollbackExecutor(registry, routing)

        class _FakeDeployment:
            id = "canary-test"
            status = "IN_PROGRESS"

        await executor.execute_rollback(
            deployment=_FakeDeployment(),  # type: ignore[arg-type]
            v1_id="qwen-7b-v1",
            v2_id="qwen-7b-v2",
            reason="v2 未激活",
        )
        # m2 仍 IDLE
        assert m2.state is ModelState.IDLE

    async def test_execute_rollback_v2_not_in_registry_force_db_sleeping(self) -> None:
        """v2 不在内存 registry（进程重启场景）时，DB 状态必须强制改成 SLEEPING。

        这是 2026-07-26 MI300X 实测发现的 bug：进程重启后内存 registry 空，
        rollback 跳过 drain 分支，导致 /api/models 仍显示 ACTIVE，客户端误以为 v2 在服务。
        修复：无论内存有没有 v2，都强制把 DB 状态标成 SLEEPING。
        """
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from vllmonline.db.models import Base, ModelVersion

        # 建 in-memory SQLite + 表
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=__import__("sqlalchemy.pool", fromlist=["StaticPool"]).StaticPool,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        # DB 里写一个 ACTIVE 的 v2（模拟进程重启前的状态）
        async with Session() as session:
            session.add(
                ModelVersion(
                    id="qwen-7b-v2",
                    model_name="qwen-7b",
                    version="v2",
                    endpoint="http://x",
                    params_billion=1.5,
                    dtype="fp16",
                    status="ACTIVE",
                )
            )
            await session.commit()

        # registry 是空的（不注册 v2）
        registry = ModelRegistry()
        routing = RoutingTableManager(registry)
        executor = RollbackExecutor(registry, routing)

        class _FakeDeployment:
            id = "canary-test"
            status = "IN_PROGRESS"

        # 执行回滚（传 session）
        async with Session() as session:
            await executor.execute_rollback(
                deployment=_FakeDeployment(),  # type: ignore[arg-type]
                v1_id="qwen-7b-v1",
                v2_id="qwen-7b-v2",
                reason="v2 不在 registry",
                session=session,
            )

        # 验证：DB 里 v2 状态必须被强制改成 SLEEPING
        async with Session() as session:
            row = await session.get(ModelVersion, "qwen-7b-v2")
            assert row is not None
            assert row.status == "SLEEPING", f"期望 SLEEPING，实际 {row.status}"

        await engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# check_and_rollback 函数式 API
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckAndRollback:
    async def test_triggers_when_degraded(self) -> None:
        registry = ModelRegistry()
        m1, m2 = _model("qwen-7b-v1"), _model("qwen-7b-v2")
        await registry.register(m1)
        await registry.register(m2)
        await _activate(m1)
        await _activate(m2)
        routing = RoutingTableManager(registry)
        executor = RollbackExecutor(registry, routing)

        class _FakeDeployment:
            id = "canary-x"
            status = "IN_PROGRESS"

        v1 = _metrics()
        v2 = _metrics(ttft_p99=1.0)  # 严重劣化
        triggered = await check_and_rollback(
            _FakeDeployment(),  # type: ignore[arg-type]
            v1,
            v2,
            executor,
            v1_id="qwen-7b-v1",
            v2_id="qwen-7b-v2",
        )
        assert triggered is True
        assert m2.state is ModelState.SLEEPING

    async def test_no_rollback_when_healthy(self) -> None:
        registry = ModelRegistry()
        m1, m2 = _model("qwen-7b-v1"), _model("qwen-7b-v2")
        await registry.register(m1)
        await registry.register(m2)
        await _activate(m1)
        await _activate(m2)
        routing = RoutingTableManager(registry)
        executor = RollbackExecutor(registry, routing)

        class _FakeDeployment:
            id = "canary-y"
            status = "IN_PROGRESS"

        triggered = await check_and_rollback(
            _FakeDeployment(),  # type: ignore[arg-type]
            _metrics(),
            _metrics(),
            executor,
            v1_id="qwen-7b-v1",
            v2_id="qwen-7b-v2",
        )
        assert triggered is False
        assert m2.state is ModelState.ACTIVE  # 没被动


# ─────────────────────────────────────────────────────────────────────────────
# RollbackSupervisor（后台巡检）
# ─────────────────────────────────────────────────────────────────────────────


class TestRollbackSupervisor:
    async def test_supervisor_rolls_back_on_degradation(self) -> None:
        """后台巡检检测到劣化 → 自动回滚。"""
        registry = ModelRegistry()
        m1, m2 = _model("qwen-7b-v1"), _model("qwen-7b-v2")
        await registry.register(m1)
        await registry.register(m2)
        await _activate(m1)
        await _activate(m2)
        routing = RoutingTableManager(registry)
        executor = RollbackExecutor(registry, routing)

        # metrics provider 总是返回劣化数据
        async def bad_metrics():
            return _metrics(), _metrics(ttft_p99=1.0)

        supervisor = RollbackSupervisor(
            RollbackChecker(),
            executor,
            bad_metrics,
            check_interval_seconds=10,
        )

        class _FakeDeployment:
            id = "canary-sup"
            status = "IN_PROGRESS"

        supervisor.watch(_FakeDeployment(), "qwen-7b-v1", "qwen-7b-v2")  # type: ignore[arg-type]

        # 跑一轮就应回滚（interval 10s 但第一轮立即检查）
        await asyncio.wait_for(supervisor.run(), timeout=5.0)
        assert m2.state is ModelState.SLEEPING

    async def test_supervisor_stops_cleanly(self) -> None:
        """stop() 让 run() 退出。"""
        registry = ModelRegistry()
        routing = RoutingTableManager(registry)
        executor = RollbackExecutor(registry, routing)

        async def good_metrics():
            return _metrics(), _metrics()

        supervisor = RollbackSupervisor(
            RollbackChecker(),
            executor,
            good_metrics,
            check_interval_seconds=10,
        )

        class _FakeDeployment:
            id = "canary-stop"
            status = "IN_PROGRESS"

        supervisor.watch(_FakeDeployment(), "v1", "v2")  # type: ignore[arg-type]

        # 启动后立即 stop
        task = asyncio.create_task(supervisor.run())
        await asyncio.sleep(0.1)
        supervisor.stop()
        await asyncio.wait_for(task, timeout=2.0)

    async def test_supervisor_without_watch_warns(self) -> None:
        supervisor = RollbackSupervisor(
            RollbackChecker(),
            RollbackExecutor(ModelRegistry(), RoutingTableManager(ModelRegistry())),
            lambda: None,  # type: ignore[arg-type]
        )
        # 没 watch → run 立即返回
        await supervisor.run()
