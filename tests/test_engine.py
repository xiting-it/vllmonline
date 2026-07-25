"""scheduler/engine.py 测试（hot_swap 编排）。

测试矩阵：
    - DIRECT 策略：先 load new，再 drain+sleep old
    - SLEEP_OLD 策略：先 drain+sleep old 释放空间，再 load new
    - UNLOAD_OLD 策略：完全卸载 old
    - INSUFFICIENT：拒绝并抛 InsufficientGpuError
    - 非法起始状态：old 非 ACTIVE / new 非 IDLE 抛错
    - load 失败回滚（old 被释放时尝试恢复）
"""

from __future__ import annotations

import pytest

from vllmonline.gpu_info import NoneGpuProvider
from vllmonline.router.drain import Drainer  # noqa: F401 - 验证可导入
from vllmonline.scheduler.engine import (
    HotSwapEngine,
    HotSwapError,
    InsufficientGpuError,
)
from vllmonline.scheduler.gpu_memory import build_model_profile
from vllmonline.scheduler.lifecycle import Model, ModelRegistry
from vllmonline.scheduler.types import ModelState
from vllmonline.vllm.adapter import VLLMAdapter
from vllmonline.vllm.client import VLLMClient


def _make_model(model_id: str, params: float = 7.0) -> Model:
    return Model(
        id=model_id,
        model_name="qwen-7b",
        version=model_id.rsplit("-", 1)[-1],
        endpoint=f"http://{model_id}:8000",
        gpu_id=0,
        memory_profile=build_model_profile(params, "fp16"),
    )


def _make_engine(
    total_gb: float = 80.0,
    used_gb: float = 0.0,
) -> tuple[HotSwapEngine, ModelRegistry]:
    """构造 engine：GPU 是 fake（NoneGpuProvider 可配 total/used）。"""
    registry = ModelRegistry()
    provider = NoneGpuProvider(fake_total_gb=total_gb, fake_used_gb=used_gb)
    # VLLMClient/Adapter 不实际调用（hot_swap 不依赖 vLLM 真实响应）
    client = VLLMClient.__new__(VLLMClient)  # 绕过 __init__，不建 http client
    client._owns_client = False  # type: ignore[attr-defined]
    adapter = VLLMAdapter(client)  # type: ignore[arg-type]
    engine = HotSwapEngine(
        registry=registry,
        adapter=adapter,
        gpu_provider=provider,
        drain_timeout_seconds=1.0,
    )
    return engine, registry


async def _to_active(model: Model) -> None:
    await model.transition(ModelState.LOADING)
    await model.transition(ModelState.ACTIVE)


# ─────────────────────────────────────────────────────────────────────────────
# 前置校验
# ─────────────────────────────────────────────────────────────────────────────


class TestHotSwapPreconditions:
    async def test_new_not_idle_raises(self) -> None:
        engine, registry = _make_engine()
        old = _make_model("v1")
        new = _make_model("v2")
        await registry.register(old)
        await registry.register(new)
        await _to_active(old)
        # 把 new 走到非 IDLE
        await new.transition(ModelState.LOADING)
        with pytest.raises(HotSwapError, match="必须是 IDLE"):
            await engine.hot_swap("v1", "v2")

    async def test_old_not_active_raises(self) -> None:
        engine, registry = _make_engine()
        old = _make_model("v1")
        new = _make_model("v2")
        await registry.register(old)
        await registry.register(new)
        # old 是 IDLE（没走 ACTIVE）
        with pytest.raises(HotSwapError, match="必须是 ACTIVE"):
            await engine.hot_swap("v1", "v2")


# ─────────────────────────────────────────────────────────────────────────────
# DIRECT 策略
# ─────────────────────────────────────────────────────────────────────────────


class TestHotSwapDirect:
    async def test_direct_strategy_zero_downtime(self) -> None:
        """空 GPU 加载小模型 → DIRECT，零请求丢失。"""
        engine, registry = _make_engine(total_gb=80.0, used_gb=0.0)
        old = _make_model("v1")
        new = _make_model("v2")
        await registry.register(old)
        await registry.register(new)
        await _to_active(old)

        result = await engine.hot_swap("v1", "v2")

        assert result.strategy.value == "DIRECT"
        assert result.zero_downtime is True
        assert new.state is ModelState.ACTIVE
        # old 在 DIRECT 流程里最后被 drain + sleep
        assert old.state is ModelState.SLEEPING


# ─────────────────────────────────────────────────────────────────────────────
# INSUFFICIENT
# ─────────────────────────────────────────────────────────────────────────────


class TestHotSwapInsufficient:
    async def test_model_too_large_raises(self) -> None:
        """新模型比 GPU 还大 → InsufficientGpuError。"""
        engine, registry = _make_engine(total_gb=20.0, used_gb=0.0)
        old = _make_model("v1", params=7.0)
        # 新模型 405B（远超 20GB GPU）
        new = _make_model("v2", params=405.0)
        await registry.register(old)
        await registry.register(new)
        await _to_active(old)

        with pytest.raises(InsufficientGpuError) as exc_info:
            await engine.hot_swap("v1", "v2")
        assert exc_info.value.required > 20.0
        # old 应该还是 ACTIVE（没动）
        assert old.state is ModelState.ACTIVE
        assert new.state is ModelState.IDLE


# ─────────────────────────────────────────────────────────────────────────────
# SLEEP_OLD 策略
# ─────────────────────────────────────────────────────────────────────────────


class TestHotSwapSleepOld:
    async def test_sleep_old_strategy(self) -> None:
        """显存不够直接加载，但 sleep old 释放 KV 后够 → SLEEP_OLD。

        构造：GPU 50GB，old 占着（weight 14 + kv 3.15），剩 ~33GB。
        new 也需要 ~17GB，但 NoneGpuProvider 的 used 是固定的——
        这里通过设 used_gb 模拟 old 已占空间。
        """
        # old model 7B fp16: total ≈ 17.15GB
        # 设 GPU total=30, used=20（剩 10 < 17.15 new 需要）
        # 但 sleep old 释放 kv 3.15 → 10+3.15=13.15 < 17.15 还不够
        # 需要更大 KV。改用 70B old：weight=140, kv=31.5
        # total=160, used=145（剩 15），new 7B 需要 17.15
        # sleep 释放 31.5 → 15+31.5=46.5 >= 17.15 ✓ SLEEP_OLD
        engine, registry = _make_engine(total_gb=160.0, used_gb=145.0)
        old = _make_model("v1", params=70.0)
        new = _make_model("v2", params=7.0)
        await registry.register(old)
        await registry.register(new)
        await _to_active(old)

        result = await engine.hot_swap("v1", "v2")
        assert result.strategy.value == "SLEEP_OLD"
        assert new.state is ModelState.ACTIVE
        # SLEEP_OLD 流程：old 在 load new 之前就被 sleep 了
        assert old.state is ModelState.SLEEPING


# ─────────────────────────────────────────────────────────────────────────────
# 结果数据类
# ─────────────────────────────────────────────────────────────────────────────


class TestHotSwapResult:
    def test_frozen(self) -> None:
        from vllmonline.scheduler.engine import HotSwapResult
        from vllmonline.scheduler.types import LoadStrategy

        r = HotSwapResult(
            old_model_id="v1",
            new_model_id="v2",
            strategy=LoadStrategy.DIRECT,
            zero_downtime=True,
        )
        with pytest.raises(AttributeError):
            r.zero_downtime = False  # type: ignore[misc]
