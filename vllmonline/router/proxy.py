"""路由表管理 + 加权路由选择（SPEC §5.1-5.2）。

RoutingTableManager 维护内存路由表，提供：
    - 按 model_name 查询所有 ACTIVE 版本
    - 加权随机选择目标版本
    - 更新流量配比（灰度推进时调用）
    - 从 ModelRegistry 同步（ACTIVE 版本自动加入）

SPEC §5.1 流程：
    1. 根据请求的 model 字段查路由表
    2. 获取所有 ACTIVE 版本的 endpoint + weight
    3. 加权随机选目标
    4. 注入 x-model-version header
    5. 转发到 vLLM
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import structlog

from vllmonline.scheduler.lifecycle import ModelRegistry
from vllmonline.scheduler.types import ModelState, RouteEntry

logger = structlog.get_logger("vllmonline.router.proxy")


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """一次路由决策的结果。"""

    selected_model_id: str
    endpoint: str
    model_version: str  # 注入到 x-model-version header 的值


class RoutingError(Exception):
    """路由失败的基类。"""


class NoActiveVersionError(RoutingError):
    """没有 ACTIVE 版本可路由。"""


class RoutingTableManager:
    """路由表管理器（线程/协程安全）。

    路由来源（优先级高到低）：
        1. 显式配置的 weight（灰度发布时由 canary strategy 设置）
        2. ModelRegistry 中 ACTIVE 的版本（默认等权重）

    设计：
        - _weights: {model_id: weight} 显式配比（灰度用）
        - 未在 _weights 里的 ACTIVE 版本按等权处理
        - 同一 model_name 下所有 ACTIVE 版本的 weight 归一化后参与加权随机
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._weights: dict[str, float] = {}  # model_id -> weight
        self._disabled: set[str] = set()  # 显式禁用的版本（weight=0）
        self._lock = asyncio.Lock()

    async def set_weight(self, model_id: str, weight: float) -> None:
        """设置某版本的流量权重（灰度推进时调用）。

        weight=0 表示显式禁用该版本（不参与路由）。
        """
        if weight < 0:
            msg = f"weight 不能为负：{weight}"
            raise ValueError(msg)
        async with self._lock:
            if weight == 0:
                self._weights.pop(model_id, None)
                self._disabled.add(model_id)
            else:
                self._weights[model_id] = weight
                self._disabled.discard(model_id)

    async def set_split(self, split: dict[str, float]) -> None:
        """批量设置流量配比（{model_id: weight}）。

        会清空之前的显式配比。SPEC §5.2 的 traffic_split 格式。
        weight=0 的版本被显式禁用。
        """
        async with self._lock:
            self._weights = {k: v for k, v in split.items() if v > 0}
            self._disabled = {k for k, v in split.items() if v <= 0}

    async def clear_weights(self) -> None:
        """清空所有显式配比（回到等权模式）。"""
        async with self._lock:
            self._weights.clear()
            self._disabled.clear()

    def get_active_entries(self, model_name: str) -> list[tuple[RouteEntry, float]]:
        """获取某 model_name 的所有可路由版本 + 有效权重。

        返回 [(RouteEntry, effective_weight), ...]，按权重降序。
        只有 ACTIVE 态的版本参与。
        """
        versions = self._registry.list_versions(model_name)
        active = [m for m in versions if m.state is ModelState.ACTIVE]
        # 过滤掉显式禁用的版本（weight=0）
        active = [m for m in active if m.id not in self._disabled]
        if not active:
            return []

        # 计算每个的有效权重
        result: list[tuple[RouteEntry, float]] = []
        configured = [m for m in active if m.id in self._weights]
        unconfigured = [m for m in active if m.id not in self._weights]

        if configured and not unconfigured:
            # 全部有显式配比，按配比
            for m in configured:
                result.append(
                    (
                        RouteEntry(model_id=m.id, endpoint=m.endpoint, weight=self._weights[m.id]),
                        self._weights[m.id],
                    )
                )
        elif not configured and unconfigured:
            # 全部无配比，等权
            equal = 1.0 / len(unconfigured)
            for m in unconfigured:
                result.append(
                    (
                        RouteEntry(model_id=m.id, endpoint=m.endpoint, weight=equal),
                        equal,
                    )
                )
        else:
            # 混合：有配比的按配比，无配比的给一个默认小权重
            # （避免无配比版本被完全饿死）
            for m in configured:
                result.append(
                    (
                        RouteEntry(model_id=m.id, endpoint=m.endpoint, weight=self._weights[m.id]),
                        self._weights[m.id],
                    )
                )
            for m in unconfigured:
                # 默认权重 = 所有显式配比的最小值（保守）
                default_w = min(self._weights[mid] for mid in self._weights) / len(unconfigured)
                result.append(
                    (
                        RouteEntry(model_id=m.id, endpoint=m.endpoint, weight=default_w),
                        default_w,
                    )
                )

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def select(self, model_name: str, *, rng: random.Random | None = None) -> RoutingDecision:
        """加权随机选一个版本。

        Args:
            model_name: 请求里的 model 字段
            rng: 可选的随机数生成器（测试用，注入固定 seed）

        Raises:
            NoActiveVersionError: 没有 ACTIVE 版本可路由
        """
        entries = self.get_active_entries(model_name)
        if not entries:
            msg = f"model {model_name!r} 没有 ACTIVE 版本可路由"
            raise NoActiveVersionError(msg)

        rng = rng or random.Random()
        total = sum(w for _, w in entries)
        if total <= 0:
            msg = f"model {model_name!r} 所有版本权重为 0"
            raise NoActiveVersionError(msg)

        # 加权随机：生成 [0, total) 的随机数，累加权重落点
        r = rng.random() * total
        cumulative = 0.0
        for route, weight in entries:
            cumulative += weight
            if r <= cumulative:
                version = route.model_id.rsplit("-", 1)[-1]  # qwen-7b-v1 -> v1
                return RoutingDecision(
                    selected_model_id=route.model_id,
                    endpoint=route.endpoint,
                    model_version=version,
                )
        # 浮点边界兜底
        route, _ = entries[-1]
        version = route.model_id.rsplit("-", 1)[-1]
        return RoutingDecision(
            selected_model_id=route.model_id,
            endpoint=route.endpoint,
            model_version=version,
        )

    def get_traffic_split(self, model_name: str) -> dict[str, float]:
        """获取归一化的流量配比（用于 API 响应）。

        返回 {model_id: normalized_weight}，所有值之和为 1.0（或 0 如果无版本）。
        """
        entries = self.get_active_entries(model_name)
        if not entries:
            return {}
        total = sum(w for _, w in entries)
        if total <= 0:
            return {}
        return {route.model_id: round(w / total, 4) for route, w in entries}


# ─────────────────────────────────────────────────────────────────────────────
# 全局单例（lifespan 注册）
# ─────────────────────────────────────────────────────────────────────────────

_global_routing_manager: RoutingTableManager | None = None


def set_global_routing_manager(manager: RoutingTableManager) -> None:
    global _global_routing_manager
    _global_routing_manager = manager


def get_routing_manager() -> RoutingTableManager:
    if _global_routing_manager is None:
        msg = "RoutingTableManager 未初始化"
        raise RuntimeError(msg)
    return _global_routing_manager
