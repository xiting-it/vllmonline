"""router/proxy.py 路由表测试（SPEC §5.1-5.2）。

测试矩阵：
    - 加权随机选择的分布正确性
    - 无 ACTIVE 版本时抛 NoActiveVersionError
    - set_weight / set_split 改变配比
    - get_traffic_split 归一化
    - 混合配比（部分显式 + 部分等权）
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from vllmonline.router.proxy import (
    NoActiveVersionError,
    RoutingDecision,
    RoutingTableManager,
)
from vllmonline.scheduler.gpu_memory import build_model_profile
from vllmonline.scheduler.lifecycle import Model, ModelRegistry
from vllmonline.scheduler.types import ModelState


def _model(model_id: str, gpu: int = 0) -> Model:
    name, ver = model_id.rsplit("-", 1)
    return Model(
        id=model_id,
        model_name=name,
        version=ver,
        endpoint=f"http://{model_id}:8000",
        gpu_id=gpu,
        memory_profile=build_model_profile(7.0, "fp16"),
    )


async def _activate(model: Model) -> None:
    await model.transition(ModelState.LOADING)
    await model.transition(ModelState.ACTIVE)


class TestRoutingSelection:
    async def test_no_active_version_raises(self) -> None:
        registry = ModelRegistry()
        await registry.register(_model("qwen-7b-v1"))  # IDLE
        manager = RoutingTableManager(registry)
        with pytest.raises(NoActiveVersionError):
            manager.select("qwen-7b")

    async def test_single_active_version_always_selected(self) -> None:
        registry = ModelRegistry()
        m = _model("qwen-7b-v1")
        await registry.register(m)
        await _activate(m)
        manager = RoutingTableManager(registry)
        for _ in range(10):
            decision = manager.select("qwen-7b")
            assert decision.selected_model_id == "qwen-7b-v1"
            assert decision.model_version == "v1"

    async def test_equal_weight_distribution(self) -> None:
        """两版本等权，大量请求后分布约 50/50。"""
        registry = ModelRegistry()
        m1 = _model("qwen-7b-v1")
        m2 = _model("qwen-7b-v2")
        await registry.register(m1)
        await registry.register(m2)
        await _activate(m1)
        await _activate(m2)
        manager = RoutingTableManager(registry)

        rng = random.Random(42)
        counts: Counter[str] = Counter()
        for _ in range(1000):
            d = manager.select("qwen-7b", rng=rng)
            counts[d.selected_model_id] += 1

        # 误差 < 10%
        assert abs(counts["qwen-7b-v1"] - 500) < 100
        assert abs(counts["qwen-7b-v2"] - 500) < 100

    async def test_weighted_distribution(self) -> None:
        """显式配比 70/30。"""
        registry = ModelRegistry()
        m1 = _model("qwen-7b-v1")
        m2 = _model("qwen-7b-v2")
        await registry.register(m1)
        await registry.register(m2)
        await _activate(m1)
        await _activate(m2)
        manager = RoutingTableManager(registry)
        await manager.set_split({"qwen-7b-v1": 0.7, "qwen-7b-v2": 0.3})

        rng = random.Random(42)
        counts: Counter[str] = Counter()
        for _ in range(1000):
            d = manager.select("qwen-7b", rng=rng)
            counts[d.selected_model_id] += 1

        # 误差 < 8%
        assert abs(counts["qwen-7b-v1"] - 700) < 80
        assert abs(counts["qwen-7b-v2"] - 300) < 80


class TestRoutingSplit:
    async def test_get_traffic_split_equal(self) -> None:
        registry = ModelRegistry()
        m1 = _model("qwen-7b-v1")
        m2 = _model("qwen-7b-v2")
        await registry.register(m1)
        await registry.register(m2)
        await _activate(m1)
        await _activate(m2)
        manager = RoutingTableManager(registry)
        split = manager.get_traffic_split("qwen-7b")
        assert split == pytest.approx({"qwen-7b-v1": 0.5, "qwen-7b-v2": 0.5}, abs=0.01)

    async def test_get_traffic_split_weighted(self) -> None:
        registry = ModelRegistry()
        m1 = _model("qwen-7b-v1")
        m2 = _model("qwen-7b-v2")
        await registry.register(m1)
        await registry.register(m2)
        await _activate(m1)
        await _activate(m2)
        manager = RoutingTableManager(registry)
        await manager.set_split({"qwen-7b-v1": 7.0, "qwen-7b-v2": 3.0})
        split = manager.get_traffic_split("qwen-7b")
        assert split["qwen-7b-v1"] == pytest.approx(0.7, abs=0.01)
        assert split["qwen-7b-v2"] == pytest.approx(0.3, abs=0.01)

    async def test_get_traffic_split_empty_when_no_active(self) -> None:
        registry = ModelRegistry()
        manager = RoutingTableManager(registry)
        assert manager.get_traffic_split("nope") == {}

    async def test_set_weight_zero_removes(self) -> None:
        """weight=0 删除该版本的配比。"""
        registry = ModelRegistry()
        m1 = _model("qwen-7b-v1")
        m2 = _model("qwen-7b-v2")
        await registry.register(m1)
        await registry.register(m2)
        await _activate(m1)
        await _activate(m2)
        manager = RoutingTableManager(registry)
        await manager.set_split({"qwen-7b-v1": 1.0, "qwen-7b-v2": 0.0})
        # v2 weight=0 → 不参与路由
        rng = random.Random(0)
        for _ in range(20):
            d = manager.select("qwen-7b", rng=rng)
            assert d.selected_model_id == "qwen-7b-v1"


class TestRoutingDecision:
    def test_frozen(self) -> None:
        d = RoutingDecision(selected_model_id="x", endpoint="http://x", model_version="v1")
        with pytest.raises(AttributeError):
            d.endpoint = "http://y"  # type: ignore[misc]
