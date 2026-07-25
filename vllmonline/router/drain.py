"""优雅排空（SPEC §4 状态机 DRAINING 态的核心逻辑）。

目标：模型切换前，先停止向旧模型发新请求，等在飞请求自然完成，
避免中断用户请求。超时后强制 cancel 作为兜底。

流程（SPEC §4.1）：
    ACTIVE → DRAINING（start_drain：路由表不再指向它）
    → 等待 pending_requests 降到 0
    → SLEEPING（drain 完成，可安全 sleep/unload）

用法：
    drainer = Drainer(model)
    await drainer.start()           # ACTIVE → DRAINING
    completed = await drainer.wait(timeout=30)
    if not completed:
        await drainer.force()       # 强制 cancel
"""

from __future__ import annotations

import asyncio

import structlog

from vllmonline.scheduler.lifecycle import Model
from vllmonline.scheduler.types import ModelState

logger = structlog.get_logger("vllmonline.router.drain")


class DrainTimeoutError(Exception):
    """drain 超时：在指定时间内 pending_requests 没降到 0。"""


class Drainer:
    """对单个 Model 执行优雅排空。

    注意：本类不直接操作路由表（那是 proxy 的职责）。
    它只负责：
        1. 把 model 从 ACTIVE 转到 DRAINING（状态机层标记"不再接新请求"）
        2. 轮询 pending_requests 等其降到 0
        3. 超时后调用 force_drain 兜底

    路由表的实际更新（让 proxy 不再选这个 model）由 engine.py 在调用
    start_drain 前后协调。
    """

    def __init__(
        self,
        model: Model,
        *,
        poll_interval: float = 0.5,
        force_cancel_timeout: float = 5.0,
    ) -> None:
        self._model = model
        self._poll_interval = poll_interval
        self._force_cancel_timeout = force_cancel_timeout

    @property
    def model(self) -> Model:
        return self._model

    async def start(self) -> None:
        """启动 drain：ACTIVE → DRAINING。

        幂等：如果已经在 DRAINING，直接返回；其他非 ACTIVE 态抛错。
        """
        if self._model.state is ModelState.DRAINING:
            logger.info("drain already in progress", model=self._model.id)
            return
        if self._model.state is not ModelState.ACTIVE:
            msg = (
                f"无法 drain：模型 {self._model.id} 处于 {self._model.state.value} 态"
                f"（必须 ACTIVE）"
            )
            raise ValueError(msg)
        await self._model.transition(ModelState.DRAINING)
        logger.info(
            "drain started",
            model=self._model.id,
            pending=self._model.pending_requests,
        )

    async def wait(self, timeout: float = 30.0) -> bool:
        """等待在飞请求降到 0。

        Returns:
            True 表示 drain 完成（pending=0）
            False 表示超时未完成（调用方应决定是否 force）
        """
        if self._model.state is not ModelState.DRAINING:
            msg = f"模型 {self._model.id} 不在 DRAINING 态，无法 wait"
            raise ValueError(msg)

        try:
            await self._poll_until_zero(timeout)
        except DrainTimeoutError:
            logger.warning(
                "drain timed out",
                model=self._model.id,
                pending=self._model.pending_requests,
                timeout=timeout,
            )
            return False

        logger.info("drain completed", model=self._model.id)
        return True

    async def force(self) -> None:
        """强制 cancel：忽略在飞请求，把 pending 归零。

        ⚠️ 这会中断用户的请求，只在 wait 超时后作为兜底用。
        真实实现需要 cancel 正在跑的 httpx 请求；这里简化为归零计数器。
        P3 proxy 实现后，应通过共享的 cancel event 通知在途请求。
        """
        logger.warning(
            "force drain invoked (in-flight requests may be dropped)",
            model=self._model.id,
            pending=self._model.pending_requests,
        )
        # 归零计数器（保守：实际请求可能仍在 vLLM 侧跑，但我们标记本侧已放弃）
        async with self._model._lock:
            self._model.pending_requests = 0

    async def run(self, timeout: float = 30.0) -> bool:
        """便捷方法：start + wait（+ 超时则 force）。

        Returns:
            True 自然完成，False 超时后强制。
        """
        await self.start()
        completed = await self.wait(timeout)
        if not completed:
            await self.force()
        return completed

    # ── 内部 ──

    async def _poll_until_zero(self, timeout: float) -> None:
        """轮询 pending_requests 直到 0 或超时。"""
        deadline = asyncio.get_event_loop().time() + timeout
        while self._model.pending_requests > 0:
            now = asyncio.get_event_loop().time()
            if now >= deadline:
                msg = (
                    f"drain 超时 ({timeout}s)："
                    f"{self._model.id} 仍有 {self._model.pending_requests} 个在飞请求"
                )
                raise DrainTimeoutError(msg)
            await asyncio.sleep(min(self._poll_interval, max(0.01, deadline - now)))


# ─────────────────────────────────────────────────────────────────────────────
# 便捷函数（engine.py 用）
# ─────────────────────────────────────────────────────────────────────────────


async def drain_and_sleep(
    model: Model,
    *,
    timeout: float = 30.0,
) -> bool:
    """排空 + sleep：drain 完成后立即转 SLEEPING。

    Returns:
        True 表示自然 drain 完成（已 SLEEPING）。
        False 表示超时强制（也已 SLEEPING，但部分请求被中断）。
    """
    drainer = Drainer(model)
    completed = await drainer.run(timeout)
    # 无论是否 force，都进入 SLEEPING（状态机要求）
    await model.transition(ModelState.SLEEPING)
    return completed
