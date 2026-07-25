"""自动回滚（SPEC §5.6）。

每 10s（可配）检查当前灰度阶段的 metrics，任一指标劣化超阈值 → 自动回滚。

回滚流程（SPEC §5.6）：
    1. drain 新模型（停止接新请求）
    2. 流量 100% 切回旧模型
    3. 记录事件到 canary_events（含 metrics_snapshot）

判定逻辑复用 GrayStrategy 的性能阈值检查（不重复实现）。
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from vllmonline.canary.strategy import (
    VersionMetrics,
    _check_performance_thresholds,
)
from vllmonline.config import CanaryConfig
from vllmonline.db.models import CanaryDeployment, CanaryEvent
from vllmonline.scheduler.lifecycle import ModelRegistry
from vllmonline.scheduler.types import ModelState

logger = structlog.get_logger("vllmonline.canary.rollback")


@dataclass(frozen=True, slots=True)
class RollbackDecision:
    """回滚判定结果。"""

    should_rollback: bool
    reason: str
    degraded_metrics: list[str]  # 劣化的指标名


class RollbackChecker:
    """检查当前 metrics 是否触发回滚。

    无状态——每次 check 接收当前 metrics 快照。
    判定逻辑与 GrayStrategy 的性能阈值一致（SPEC §5.5 表），
    但回滚阈值可以更严格（默认与推进阈值相同，可独立配置）。
    """

    def __init__(self, config: CanaryConfig | None = None) -> None:
        self._config = config or CanaryConfig()

    def check(
        self,
        metrics_v1: VersionMetrics,
        metrics_v2: VersionMetrics,
    ) -> RollbackDecision:
        """检查 v2 是否劣化到需要回滚。

        复用 _check_performance_thresholds——任何性能劣化都触发回滚。
        """
        blocks = _check_performance_thresholds(metrics_v1, metrics_v2, self._config)
        if not blocks:
            return RollbackDecision(
                should_rollback=False,
                reason="无劣化",
                degraded_metrics=[],
            )
        return RollbackDecision(
            should_rollback=True,
            reason=f"检测到 {len(blocks)} 项劣化：{'；'.join(blocks)}",
            degraded_metrics=blocks,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 回滚执行器
# ─────────────────────────────────────────────────────────────────────────────


class RollbackExecutor:
    """执行回滚动作（drain v2 + 100% v1 + 记录事件）。

    依赖：
        - ModelRegistry：操作模型状态
        - RoutingTableManager：调整流量配比
        - DB session：记录事件
    """

    def __init__(
        self,
        registry: ModelRegistry,
        routing_manager: Any,  # RoutingTableManager，避免循环导入
    ) -> None:
        self._registry = registry
        self._routing = routing_manager

    async def execute_rollback(
        self,
        deployment: CanaryDeployment,
        v1_id: str,
        v2_id: str,
        reason: str,
        metrics_snapshot: dict[str, Any] | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """执行完整回滚流程。

        1. v2 drain → SLEEPING（停止服务）
        2. 路由表 100% → v1
        3. 部署状态 → ROLLED_BACK
        4. 记录事件（如有 session）
        """
        log = logger.bind(deployment=deployment.id, v1=v1_id, v2=v2_id)
        log.warning("rollback executing", reason=reason)

        # 1. drain + sleep v2
        if v2_id in self._registry:
            v2 = self._registry.get(v2_id)
            if v2.state is ModelState.ACTIVE:
                from vllmonline.router.drain import drain_and_sleep

                await drain_and_sleep(v2, timeout=30.0)
                log.info("v2 drained and slept", model=v2_id)
                # 同步 v2 状态到 DB
                if session is not None:
                    from vllmonline.db.models import ModelVersion

                    v2_row = await session.get(ModelVersion, v2_id)
                    if v2_row is not None:
                        v2_row.status = v2.state.value
                        v2_row.state_changed_at = datetime.now(UTC)
                        await session.commit()

        # 2. 流量 100% 切回 v1
        await self._routing.set_split({v1_id: 1.0, v2_id: 0.0})
        log.info("traffic 100% back to v1")

        # 3. 更新部署状态
        deployment.status = "ROLLED_BACK"
        deployment.updated_at = datetime.now(UTC)
        if session is not None:
            await session.commit()

        # 4. 记录事件
        if session is not None:
            event = CanaryEvent(
                deployment_id=deployment.id,
                action="ROLLBACK",
                reason=reason,
                metrics_snapshot=metrics_snapshot,
            )
            session.add(event)
            await session.commit()

        log.info("rollback completed")


# ─────────────────────────────────────────────────────────────────────────────
# 后台巡检循环（SPEC §5.6："每 10s 执行一次"）
# ─────────────────────────────────────────────────────────────────────────────


class RollbackSupervisor:
    """后台巡检：定期检查灰度部署，劣化则自动回滚。

    用法：
        supervisor = RollbackSupervisor(...)
        task = asyncio.create_task(supervisor.run())
        # ...
        supervisor.stop()
        await task
    """

    def __init__(
        self,
        checker: RollbackChecker,
        executor: RollbackExecutor,
        metrics_provider: Any,  # Callable[[], Awaitable[tuple[VersionMetrics, VersionMetrics]]]
        *,
        check_interval_seconds: int = 10,
    ) -> None:
        self._checker = checker
        self._executor = executor
        self._metrics_provider = metrics_provider
        self._interval = check_interval_seconds
        self._stop_event = asyncio.Event()
        self._deployment: Any = None  # 当前监控的部署
        self._v1_id: str | None = None
        self._v2_id: str | None = None

    def watch(
        self,
        deployment: Any,
        v1_id: str,
        v2_id: str,
    ) -> None:
        """设置当前监控的部署。"""
        self._deployment = deployment
        self._v1_id = v1_id
        self._v2_id = v2_id

    def stop(self) -> None:
        """停止巡检。"""
        self._stop_event.set()

    async def run(self) -> None:
        """巡检主循环。每 _interval 秒检查一次。

        检测到劣化 → 调 executor.execute_rollback → 停止循环。
        """
        if self._deployment is None:
            logger.warning("supervisor started without deployment to watch")
            return

        log = logger.bind(deployment=self._deployment.id if self._deployment else None)
        log.info("rollback supervisor started", interval_s=self._interval)

        while not self._stop_event.is_set():
            try:
                metrics_v1, metrics_v2 = await self._metrics_provider()
                decision = self._checker.check(metrics_v1, metrics_v2)
                if decision.should_rollback:
                    log.warning(
                        "degradation detected, rolling back",
                        reason=decision.reason,
                        degraded=decision.degraded_metrics,
                    )
                    snapshot = {
                        "v1": _metrics_to_dict(metrics_v1),
                        "v2": _metrics_to_dict(metrics_v2),
                        "degraded": decision.degraded_metrics,
                        "checked_at": datetime.now(UTC).isoformat(),
                    }
                    await self._executor.execute_rollback(
                        deployment=self._deployment,
                        v1_id=self._v1_id,  # type: ignore[arg-type]
                        v2_id=self._v2_id,  # type: ignore[arg-type]
                        reason=decision.reason,
                        metrics_snapshot=snapshot,
                    )
                    return  # 回滚后退出
            except Exception as e:
                log.error("supervisor check failed", error=str(e))

            # 等待下次检查（可被 stop 中断）。TimeoutError 是预期行为。
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)

        log.info("rollback supervisor stopped")


def _metrics_to_dict(m: VersionMetrics) -> dict[str, float | None]:
    """VersionMetrics → dict（用于 metrics_snapshot JSON）。"""
    return {
        "ttft_p50": m.ttft_p50,
        "ttft_p99": m.ttft_p99,
        "tpot_mean": m.tpot_mean,
        "throughput": m.throughput,
        "error_rate": m.error_rate,
    }


# 兼容：SPEC §5.6 的 check_and_rollback 函数式 API
async def check_and_rollback(
    deployment: CanaryDeployment,
    metrics_v1: VersionMetrics,
    metrics_v2: VersionMetrics,
    executor: RollbackExecutor,
    config: CanaryConfig | None = None,
    *,
    v1_id: str,
    v2_id: str,
) -> bool:
    """SPEC §5.6 的函数式 API：检查 + 必要时回滚。

    Returns:
        True 表示触发了回滚，False 表示无需回滚。
    """
    checker = RollbackChecker(config)
    decision = checker.check(metrics_v1, metrics_v2)
    if not decision.should_rollback:
        return False
    await executor.execute_rollback(
        deployment=deployment,
        v1_id=v1_id,
        v2_id=v2_id,
        reason=decision.reason,
    )
    return True
