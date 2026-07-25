"""模型状态机 + ModelRegistry（SPEC §4）。

⚠️ 状态机是保证系统行为可预测的基础。必须完备，不允许未定义行为。
   任何状态变更都要走 transition()，先校验合法性，再持久化。

并发安全：每个 Model 实例持有一个 asyncio.Lock，状态变更必须加锁。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from vllmonline.scheduler.types import ModelMemoryProfile, ModelState

# ─────────────────────────────────────────────────────────────────────────────
# 异常
# ─────────────────────────────────────────────────────────────────────────────


class IllegalTransitionError(Exception):
    """非法状态转移。

    SPEC §4.3：transition(from, to) 先检查 TRANSITIONS[from] 是否包含 to，
    不包含则抛本异常。
    """

    def __init__(self, from_state: ModelState, to_state: ModelState) -> None:
        self.from_state = from_state
        self.to_state = to_state
        allowed: frozenset[ModelState] = TRANSITIONS.get(from_state, frozenset())
        super().__init__(
            f"非法状态转移：{from_state.value} → {to_state.value}。"
            f"{from_state.value} 允许的目标状态："
            f"{sorted(s.value for s in allowed) or '<无>'}。"
        )


class ModelNotFoundError(KeyError):
    """ModelRegistry 中找不到指定 model_id。"""


# ─────────────────────────────────────────────────────────────────────────────
# 状态转移表（SPEC §4.2，硬编码）
# ─────────────────────────────────────────────────────────────────────────────

TRANSITIONS: dict[ModelState, frozenset[ModelState]] = {
    ModelState.IDLE: frozenset({ModelState.LOADING}),
    ModelState.LOADING: frozenset({ModelState.ACTIVE, ModelState.ERROR}),
    ModelState.ACTIVE: frozenset({ModelState.DRAINING, ModelState.SLEEPING, ModelState.ERROR}),
    ModelState.DRAINING: frozenset({ModelState.SLEEPING, ModelState.ERROR}),
    ModelState.SLEEPING: frozenset({ModelState.LOADING, ModelState.UNLOADING, ModelState.ERROR}),
    ModelState.UNLOADING: frozenset({ModelState.IDLE, ModelState.ERROR}),
    ModelState.ERROR: frozenset({ModelState.IDLE}),
}


def is_valid_transition(from_state: ModelState, to_state: ModelState) -> bool:
    """检查状态转移是否合法（纯函数，供测试直接调用）。"""
    return to_state in TRANSITIONS.get(from_state, frozenset())


# ─────────────────────────────────────────────────────────────────────────────
# 持久化回调类型
# ─────────────────────────────────────────────────────────────────────────────

# transition 后调用的持久化回调（写 DB）。
# P2 起注入真实 DB session；P1 单测里用 noop 或 mock。
PersistCallback = Callable[[ModelState, ModelState], Awaitable[None]]


async def _noop_persist(_from: ModelState, _to: ModelState) -> None:
    """默认持久化回调：什么都不做（P1 单测用）。"""


# ─────────────────────────────────────────────────────────────────────────────
# Model 数据类
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Model:
    """一个模型实例（注册到 registry 的某个版本）。

    每个 Model 持有独立的 asyncio.Lock，保证状态变更串行化。
    SPEC §4.3 要求：每次状态变更必须持久化到 DB（通过 persist_callback）。
    """

    # 不可变标识（创建后不变）
    id: str  # 如 "qwen-7b-v2"
    model_name: str  # 如 "qwen-7b"（同 model_name 可有多个 version）
    version: str  # 如 "v2"
    endpoint: str  # vLLM 服务 URL
    gpu_id: int  # 绑定的 GPU
    memory_profile: ModelMemoryProfile

    # 可变状态
    state: ModelState = ModelState.IDLE
    pending_requests: int = 0
    total_requests_served: int = 0
    total_errors: int = 0
    state_changed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # 内部
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _persist: PersistCallback = field(default=_noop_persist, repr=False)

    # ── 状态查询（SPEC §4.4）──
    def can_serve(self) -> bool:
        """能否接受新请求（仅 ACTIVE）。"""
        return self.state.can_serve()

    def is_on_gpu(self) -> bool:
        """是否占用 GPU 显存。"""
        return self.state.is_on_gpu()

    def is_terminal(self) -> bool:
        """是否终态。"""
        return self.state.is_terminal()

    @property
    def weight_gb(self) -> float:
        """权重显存（GB）。"""
        return self.memory_profile.weight_gb

    @property
    def kv_cache_gb(self) -> float:
        """KV cache 预算（GB）。"""
        return self.memory_profile.kv_cache_budget_gb

    @property
    def total_memory_gb(self) -> float:
        """满载总显存（权重 + KV cache）。"""
        return self.memory_profile.total_gb

    @property
    def current_memory_gb(self) -> float:
        """当前实际占用显存（按 state 折算）。

        SLEEPING 态只占权重；其他 on_gpu 态占满；非 on_gpu 态占 0。
        """
        if self.state is ModelState.SLEEPING:
            return self.memory_profile.sleeping_gb
        if self.is_on_gpu():
            return self.total_memory_gb
        return 0.0

    # ── 状态转移（SPEC §4.3）──
    async def transition(self, to: ModelState) -> None:
        """状态转移（加锁 + 校验 + 持久化）。

        流程：
            1. 加锁（asyncio.Lock）
            2. 校验 TRANSITIONS[from] 包含 to，否则 IllegalTransitionError
            3. 更新 state + state_changed_at
            4. 调用 persist_callback 写 DB

        Raises:
            IllegalTransitionError: 转移不在合法表里。
        """
        async with self._lock:
            from_state = self.state
            if not is_valid_transition(from_state, to):
                raise IllegalTransitionError(from_state, to)
            self.state = to
            self.state_changed_at = datetime.now(UTC)
            await self._persist(from_state, to)

    def set_persist_callback(self, callback: PersistCallback) -> None:
        """注入持久化回调（P2 由 DB session 注入）。"""
        self._persist = callback

    # ── 请求计数（drain 用）──
    async def increment_pending(self) -> None:
        """在飞请求数 +1（代理转发前调用）。"""
        async with self._lock:
            self.pending_requests += 1

    async def decrement_pending(self) -> None:
        """在飞请求数 -1（请求完成后调用，不会降到 0 以下）。"""
        async with self._lock:
            self.pending_requests = max(0, self.pending_requests - 1)

    async def record_request_served(self) -> None:
        """记录一次成功请求。"""
        async with self._lock:
            self.total_requests_served += 1

    async def record_error(self) -> None:
        """记录一次错误。"""
        async with self._lock:
            self.total_errors += 1


# ─────────────────────────────────────────────────────────────────────────────
# ModelRegistry（线程/协程安全的模型注册表）
# ─────────────────────────────────────────────────────────────────────────────


class ModelRegistry:
    """所有已注册模型的内存注册表。

    SPEC §4.3 要求线程安全——本实现用一个 asyncio.Lock 保护 dict 操作。
    注意：lock 在异步上下文里协程串行，但跨 event loop 不安全。
    单进程内多协程使用是 OK 的；多进程部署需另用 Redis 共享状态。
    """

    def __init__(self) -> None:
        self._models: dict[str, Model] = {}
        self._lock = asyncio.Lock()

    async def register(self, model: Model) -> Model:
        """注册一个模型。

        Raises:
            ValueError: model_id 已存在。
        """
        async with self._lock:
            if model.id in self._models:
                msg = f"模型 {model.id} 已注册"
                raise ValueError(msg)
            self._models[model.id] = model
            return model

    async def unregister(self, model_id: str) -> Model:
        """注销一个模型（必须为终态）。

        Raises:
            ModelNotFoundError: 不存在。
            ValueError: 模型非终态（不能直接删）。
        """
        async with self._lock:
            if model_id not in self._models:
                raise ModelNotFoundError(model_id)
            model = self._models[model_id]
            if not model.is_terminal():
                msg = (
                    f"模型 {model_id} 处于 {model.state.value} 态，"
                    f"不能注销（必须是 IDLE 或 ERROR 终态）"
                )
                raise ValueError(msg)
            return self._models.pop(model_id)

    def get(self, model_id: str) -> Model:
        """同步获取（不加锁，读 dict 是原子的）。

        Raises:
            ModelNotFoundError: 不存在。
        """
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        return self._models[model_id]

    def list_all(self) -> list[Model]:
        """列出所有模型（任意状态）。"""
        return list(self._models.values())

    def list_serving(self) -> list[Model]:
        """列出所有 ACTIVE 模型（正在服务请求的）。"""
        return [m for m in self._models.values() if m.can_serve()]

    def list_on_gpu(self, gpu_id: int | None = None) -> list[Model]:
        """列出所有占用 GPU 显存的模型。

        Args:
            gpu_id: 指定 GPU id 过滤；None 表示所有 GPU。
        """
        return [
            m
            for m in self._models.values()
            if m.is_on_gpu() and (gpu_id is None or m.gpu_id == gpu_id)
        ]

    def get_active_version(self, model_name: str) -> Model | None:
        """获取某 model_name 当前 ACTIVE 的版本（用于默认路由）。

        若有多个 ACTIVE 版本（灰度中），返回第一个。
        无 ACTIVE 版本返回 None。
        """
        for m in self._models.values():
            if m.model_name == model_name and m.can_serve():
                return m
        return None

    def list_versions(self, model_name: str) -> list[Model]:
        """列出某 model_name 的所有版本。"""
        return [m for m in self._models.values() if m.model_name == model_name]

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._models
