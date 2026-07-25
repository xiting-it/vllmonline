"""lifecycle.py 单测（SPEC §4，覆盖率目标 ≥90%）。

测试矩阵：
    - 所有合法转移通过
    - 所有非法转移被 IllegalTransitionError 拦截
    - 并发安全：两协程同时 transition 不出现竞态
    - ModelRegistry CRUD + 查询方法
    - 辅助判定 can_serve / is_on_gpu / is_terminal
    - 请求计数
"""

from __future__ import annotations

import asyncio

import pytest

from vllmonline.scheduler.gpu_memory import build_model_profile
from vllmonline.scheduler.lifecycle import (
    TRANSITIONS,
    IllegalTransitionError,
    Model,
    ModelNotFoundError,
    ModelRegistry,
    is_valid_transition,
)
from vllmonline.scheduler.types import ModelState

# ─────────────────────────────────────────────────────────────────────────────
# 测试辅助
# ─────────────────────────────────────────────────────────────────────────────


def _make_model(model_id: str = "qwen-7b-v1") -> Model:
    """构造一个 IDLE 态的 Model。"""
    return Model(
        id=model_id,
        model_name="qwen-7b",
        version="v1",
        endpoint="http://vllm:8000/v1",
        gpu_id=0,
        memory_profile=build_model_profile(7.0, "fp16"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 状态转移表（SPEC §4.2）
# ─────────────────────────────────────────────────────────────────────────────


class TestTransitionTable:
    """验证 TRANSITIONS 表与 SPEC §4.2 完全一致。"""

    def test_idle_only_allows_loading(self) -> None:
        assert set(TRANSITIONS[ModelState.IDLE]) == {ModelState.LOADING}

    def test_loading_allows_active_and_error(self) -> None:
        assert set(TRANSITIONS[ModelState.LOADING]) == {ModelState.ACTIVE, ModelState.ERROR}

    def test_active_allows_draining_sleeping_error(self) -> None:
        assert set(TRANSITIONS[ModelState.ACTIVE]) == {
            ModelState.DRAINING,
            ModelState.SLEEPING,
            ModelState.ERROR,
        }

    def test_draining_allows_sleeping_error(self) -> None:
        assert set(TRANSITIONS[ModelState.DRAINING]) == {ModelState.SLEEPING, ModelState.ERROR}

    def test_sleeping_allows_loading_unloading_error(self) -> None:
        assert set(TRANSITIONS[ModelState.SLEEPING]) == {
            ModelState.LOADING,  # wake
            ModelState.UNLOADING,
            ModelState.ERROR,
        }

    def test_unloading_allows_idle_error(self) -> None:
        assert set(TRANSITIONS[ModelState.UNLOADING]) == {ModelState.IDLE, ModelState.ERROR}

    def test_error_only_allows_idle(self) -> None:
        """SPEC §4.2：ERROR 只能 reset 到 IDLE。"""
        assert set(TRANSITIONS[ModelState.ERROR]) == {ModelState.IDLE}

    def test_all_states_covered(self) -> None:
        """TRANSITIONS 覆盖全部 7 个状态。"""
        assert set(TRANSITIONS.keys()) == set(ModelState)

    def test_is_valid_transition_pure_function(self) -> None:
        """纯函数版校验。"""
        assert is_valid_transition(ModelState.IDLE, ModelState.LOADING)
        assert not is_valid_transition(ModelState.IDLE, ModelState.ACTIVE)


# ─────────────────────────────────────────────────────────────────────────────
# Model.transition 合法转移
# ─────────────────────────────────────────────────────────────────────────────


class TestLegalTransitions:
    async def test_full_load_cycle(self) -> None:
        """完整加载周期：IDLE → LOADING → ACTIVE。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        assert m.state is ModelState.LOADING
        await m.transition(ModelState.ACTIVE)
        assert m.state is ModelState.ACTIVE

    async def test_active_to_sleeping(self) -> None:
        """ACTIVE → SLEEPING（休眠）。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ACTIVE)
        await m.transition(ModelState.SLEEPING)
        assert m.state is ModelState.SLEEPING

    async def test_sleeping_to_loading_wake(self) -> None:
        """SLEEPING → LOADING（wake 唤醒）。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ACTIVE)
        await m.transition(ModelState.SLEEPING)
        await m.transition(ModelState.LOADING)  # wake
        assert m.state is ModelState.LOADING

    async def test_active_to_draining(self) -> None:
        """ACTIVE → DRAINING。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ACTIVE)
        await m.transition(ModelState.DRAINING)
        assert m.state is ModelState.DRAINING

    async def test_draining_to_sleeping(self) -> None:
        """DRAINING → SLEEPING（drain 完成后）。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ACTIVE)
        await m.transition(ModelState.DRAINING)
        await m.transition(ModelState.SLEEPING)
        assert m.state is ModelState.SLEEPING

    async def test_full_unload_cycle(self) -> None:
        """完整卸载：ACTIVE → DRAINING → SLEEPING → UNLOADING → IDLE。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ACTIVE)
        await m.transition(ModelState.DRAINING)
        await m.transition(ModelState.SLEEPING)
        await m.transition(ModelState.UNLOADING)
        assert m.state is ModelState.UNLOADING
        await m.transition(ModelState.IDLE)
        assert m.state is ModelState.IDLE

    async def test_error_from_any_state(self) -> None:
        """任何状态都能转 ERROR（除 ERROR 自身）。"""
        for target in [
            ModelState.LOADING,
            ModelState.ACTIVE,
            ModelState.DRAINING,
            ModelState.SLEEPING,
            ModelState.UNLOADING,
        ]:
            m = _make_model()
            # 走到 target 态
            await _force_state(m, target)
            await m.transition(ModelState.ERROR)
            assert m.state is ModelState.ERROR

    async def test_error_to_idle_reset(self) -> None:
        """ERROR → IDLE（reset）。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ERROR)
        await m.transition(ModelState.IDLE)
        assert m.state is ModelState.IDLE

    async def test_state_changed_at_updates(self) -> None:
        """每次转移更新 state_changed_at。"""
        import time

        m = _make_model()
        t0 = m.state_changed_at
        time.sleep(0.001)  # 确保时间戳不同
        await m.transition(ModelState.LOADING)
        assert m.state_changed_at > t0


async def _force_state(model: Model, target: ModelState) -> None:
    """测试辅助：把 model 走到指定状态（假设路径存在）。"""
    path_map: dict[ModelState, list[ModelState]] = {
        ModelState.IDLE: [],  # 初始就是 IDLE
        ModelState.LOADING: [ModelState.LOADING],
        ModelState.ACTIVE: [ModelState.LOADING, ModelState.ACTIVE],
        ModelState.DRAINING: [ModelState.LOADING, ModelState.ACTIVE, ModelState.DRAINING],
        ModelState.SLEEPING: [
            ModelState.LOADING,
            ModelState.ACTIVE,
            ModelState.DRAINING,
            ModelState.SLEEPING,
        ],
        ModelState.UNLOADING: [
            ModelState.LOADING,
            ModelState.ACTIVE,
            ModelState.DRAINING,
            ModelState.SLEEPING,
            ModelState.UNLOADING,
        ],
        ModelState.ERROR: [ModelState.LOADING, ModelState.ERROR],
    }
    for s in path_map[target]:
        await model.transition(s)


# ─────────────────────────────────────────────────────────────────────────────
# Model.transition 非法转移
# ─────────────────────────────────────────────────────────────────────────────


class TestIllegalTransitions:
    """穷举所有非法转移，确保都被拦截。"""

    async def test_active_to_idle_illegal(self) -> None:
        """SPEC §4 验收：ACTIVE → IDLE 非法。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ACTIVE)
        with pytest.raises(IllegalTransitionError):
            await m.transition(ModelState.IDLE)

    async def test_idle_to_active_illegal(self) -> None:
        """IDLE 不能直接到 ACTIVE（必须先 LOADING）。"""
        m = _make_model()
        with pytest.raises(IllegalTransitionError):
            await m.transition(ModelState.ACTIVE)

    async def test_error_to_active_illegal(self) -> None:
        """ERROR 只能到 IDLE，不能到 ACTIVE。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ERROR)
        with pytest.raises(IllegalTransitionError):
            await m.transition(ModelState.ACTIVE)

    async def test_unloading_to_active_illegal(self) -> None:
        m = _make_model()
        await _force_state(m, ModelState.UNLOADING)
        with pytest.raises(IllegalTransitionError):
            await m.transition(ModelState.ACTIVE)

    async def test_draining_to_active_illegal(self) -> None:
        """DRAINING 不能回到 ACTIVE（必须重新 load）。"""
        m = _make_model()
        await _force_state(m, ModelState.DRAINING)
        with pytest.raises(IllegalTransitionError):
            await m.transition(ModelState.ACTIVE)

    async def test_error_message_includes_states(self) -> None:
        """异常信息含 from/to 状态和合法目标。"""
        m = _make_model()
        try:
            await m.transition(ModelState.ACTIVE)  # IDLE → ACTIVE 非法
        except IllegalTransitionError as e:
            assert "IDLE" in str(e)
            assert "ACTIVE" in str(e)
            assert "LOADING" in str(e)  # 合法目标
        else:
            pytest.fail("应抛 IllegalTransitionError")

    @pytest.mark.parametrize(
        ("from_state", "to_state"),
        [
            (s, t)
            for s in ModelState
            for t in ModelState
            if s != t and not is_valid_transition(s, t)
        ],
    )
    async def test_all_illegal_transitions_blocked(
        self, from_state: ModelState, to_state: ModelState
    ) -> None:
        """穷举：所有非法转移都被拦截。"""
        m = _make_model()
        await _force_state(m, from_state)
        assert m.state is from_state
        with pytest.raises(IllegalTransitionError):
            await m.transition(to_state)


# ─────────────────────────────────────────────────────────────────────────────
# 并发安全（SPEC §4.3）
# ─────────────────────────────────────────────────────────────────────────────


class TestConcurrency:
    async def test_concurrent_transitions_no_race(self) -> None:
        """两个协程同时 transition，只有一个成功，另一个抛 IllegalTransitionError。

        场景：model 在 IDLE。协程 A 想 IDLE→LOADING，协程 B 也想 IDLE→LOADING。
        A 先拿到锁，转移成功（state=LOADING）。B 拿到锁时 state 已是 LOADING，
        LOADING→LOADING 不在 TRANSITIONS 表里 → 抛错。
        """
        m = _make_model()
        results: list[Exception | None] = [None, None]

        async def try_transition(idx: int) -> None:
            try:
                await m.transition(ModelState.LOADING)
            except IllegalTransitionError as e:
                results[idx] = e

        await asyncio.gather(try_transition(0), try_transition(1))

        # 恰好一个成功，一个失败
        successes = sum(1 for r in results if r is None)
        failures = sum(1 for r in results if r is not None)
        assert successes == 1, f"应有 1 个成功，实际 {successes}"
        assert failures == 1, f"应有 1 个失败，实际 {failures}"
        assert m.state is ModelState.LOADING

    async def test_concurrent_increment_pending_safe(self) -> None:
        """并发 +1 不丢更新。"""
        m = _make_model()
        await asyncio.gather(*[m.increment_pending() for _ in range(100)])
        assert m.pending_requests == 100


# ─────────────────────────────────────────────────────────────────────────────
# 辅助判定（SPEC §4.4）
# ─────────────────────────────────────────────────────────────────────────────


class TestStateHelpers:
    @pytest.mark.parametrize(
        ("state", "can_serve"),
        [
            (ModelState.IDLE, False),
            (ModelState.LOADING, False),
            (ModelState.ACTIVE, True),
            (ModelState.DRAINING, False),
            (ModelState.SLEEPING, False),
            (ModelState.UNLOADING, False),
            (ModelState.ERROR, False),
        ],
    )
    def test_can_serve(self, state: ModelState, can_serve: bool) -> None:
        assert state.can_serve() is can_serve

    @pytest.mark.parametrize(
        ("state", "on_gpu"),
        [
            (ModelState.IDLE, False),
            (ModelState.LOADING, True),
            (ModelState.ACTIVE, True),
            (ModelState.DRAINING, True),
            (ModelState.SLEEPING, True),
            (ModelState.UNLOADING, False),
            (ModelState.ERROR, False),
        ],
    )
    def test_is_on_gpu(self, state: ModelState, on_gpu: bool) -> None:
        assert state.is_on_gpu() is on_gpu

    @pytest.mark.parametrize(
        ("state", "terminal"),
        [
            (ModelState.IDLE, True),
            (ModelState.LOADING, False),
            (ModelState.ACTIVE, False),
            (ModelState.DRAINING, False),
            (ModelState.SLEEPING, False),
            (ModelState.UNLOADING, False),
            (ModelState.ERROR, True),
        ],
    )
    def test_is_terminal(self, state: ModelState, terminal: bool) -> None:
        assert state.is_terminal() is terminal


# ─────────────────────────────────────────────────────────────────────────────
# Model 显存属性
# ─────────────────────────────────────────────────────────────────────────────


class TestModelMemoryAttrs:
    async def test_current_memory_active(self) -> None:
        """ACTIVE 态占满（权重 + KV）。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ACTIVE)
        assert m.current_memory_gb == pytest.approx(m.total_memory_gb)

    async def test_current_memory_sleeping(self) -> None:
        """SLEEPING 态只占权重。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ACTIVE)
        await m.transition(ModelState.SLEEPING)
        assert m.current_memory_gb == pytest.approx(m.weight_gb)

    async def test_current_memory_idle_zero(self) -> None:
        """IDLE 态不占显存。"""
        m = _make_model()
        assert m.current_memory_gb == 0.0

    async def test_current_memory_loading(self) -> None:
        """LOADING 态按满载算（保守估计）。"""
        m = _make_model()
        await m.transition(ModelState.LOADING)
        assert m.current_memory_gb == pytest.approx(m.total_memory_gb)


# ─────────────────────────────────────────────────────────────────────────────
# 请求计数
# ─────────────────────────────────────────────────────────────────────────────


class TestRequestCounters:
    async def test_increment_decrement(self) -> None:
        m = _make_model()
        await m.increment_pending()
        await m.increment_pending()
        assert m.pending_requests == 2
        await m.decrement_pending()
        assert m.pending_requests == 1

    async def test_decrement_clamped_at_zero(self) -> None:
        """降到 0 不会变负。"""
        m = _make_model()
        await m.decrement_pending()
        assert m.pending_requests == 0
        await m.decrement_pending()
        assert m.pending_requests == 0

    async def test_record_served(self) -> None:
        m = _make_model()
        for _ in range(5):
            await m.record_request_served()
        assert m.total_requests_served == 5

    async def test_record_error(self) -> None:
        m = _make_model()
        for _ in range(3):
            await m.record_error()
        assert m.total_errors == 3


# ─────────────────────────────────────────────────────────────────────────────
# 持久化回调
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistCallback:
    async def test_persist_called_on_transition(self) -> None:
        """transition 后调用 persist_callback。"""
        calls: list[tuple[ModelState, ModelState]] = []

        async def persist(_from: ModelState, _to: ModelState) -> None:
            calls.append((_from, _to))

        m = _make_model()
        m.set_persist_callback(persist)
        await m.transition(ModelState.LOADING)
        assert calls == [(ModelState.IDLE, ModelState.LOADING)]
        await m.transition(ModelState.ACTIVE)
        assert calls[-1] == (ModelState.LOADING, ModelState.ACTIVE)

    async def test_persist_called_even_on_error_transition(self) -> None:
        calls: list[tuple[ModelState, ModelState]] = []

        async def persist(_from: ModelState, _to: ModelState) -> None:
            calls.append((_from, _to))

        m = _make_model()
        m.set_persist_callback(persist)
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ERROR)
        assert (ModelState.LOADING, ModelState.ERROR) in calls


# ─────────────────────────────────────────────────────────────────────────────
# ModelRegistry
# ─────────────────────────────────────────────────────────────────────────────


class TestModelRegistry:
    async def test_register_and_get(self) -> None:
        reg = ModelRegistry()
        m = _make_model("qwen-7b-v1")
        await reg.register(m)
        assert reg.get("qwen-7b-v1") is m
        assert "qwen-7b-v1" in reg
        assert len(reg) == 1

    async def test_register_duplicate_raises(self) -> None:
        reg = ModelRegistry()
        await reg.register(_make_model("qwen-7b-v1"))
        with pytest.raises(ValueError, match="已注册"):
            await reg.register(_make_model("qwen-7b-v1"))

    async def test_get_missing_raises(self) -> None:
        reg = ModelRegistry()
        with pytest.raises(ModelNotFoundError):
            reg.get("nonexistent")

    async def test_unregister(self) -> None:
        reg = ModelRegistry()
        m = _make_model()
        await reg.register(m)
        removed = await reg.unregister(m.id)
        assert removed is m
        assert m.id not in reg

    async def test_unregister_non_terminal_raises(self) -> None:
        """非终态模型不能直接注销。"""
        reg = ModelRegistry()
        m = _make_model()
        await reg.register(m)
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ACTIVE)
        with pytest.raises(ValueError, match="终态"):
            await reg.unregister(m.id)

    async def test_unregister_missing_raises(self) -> None:
        reg = ModelRegistry()
        with pytest.raises(ModelNotFoundError):
            await reg.unregister("nope")

    async def test_list_all(self) -> None:
        reg = ModelRegistry()
        await reg.register(_make_model("v1"))
        await reg.register(_make_model("v2"))
        assert len(reg.list_all()) == 2

    async def test_list_serving_only_active(self) -> None:
        reg = ModelRegistry()
        m1 = _make_model("v1")
        m2 = _make_model("v2")
        await reg.register(m1)
        await reg.register(m2)
        # 都 IDLE，没有 serving
        assert reg.list_serving() == []
        # v1 ACTIVE
        await m1.transition(ModelState.LOADING)
        await m1.transition(ModelState.ACTIVE)
        serving = reg.list_serving()
        assert len(serving) == 1
        assert serving[0].id == "v1"

    async def test_list_on_gpu_filters_by_state(self) -> None:
        reg = ModelRegistry()
        m_idle = _make_model("idle")
        m_active = _make_model("active")
        m_sleeping = _make_model("sleeping")
        await reg.register(m_idle)
        await reg.register(m_active)
        await reg.register(m_sleeping)
        await m_active.transition(ModelState.LOADING)
        await m_active.transition(ModelState.ACTIVE)
        await m_sleeping.transition(ModelState.LOADING)
        await m_sleeping.transition(ModelState.ACTIVE)
        await m_sleeping.transition(ModelState.SLEEPING)
        on_gpu = reg.list_on_gpu()
        assert {m.id for m in on_gpu} == {"active", "sleeping"}

    async def test_list_on_gpu_filters_by_gpu_id(self) -> None:
        reg = ModelRegistry()
        m_gpu0 = Model(
            id="g0",
            model_name="x",
            version="v",
            endpoint="http://x",
            gpu_id=0,
            memory_profile=build_model_profile(7.0, "fp16"),
        )
        m_gpu1 = Model(
            id="g1",
            model_name="x",
            version="v",
            endpoint="http://x",
            gpu_id=1,
            memory_profile=build_model_profile(7.0, "fp16"),
        )
        await reg.register(m_gpu0)
        await reg.register(m_gpu1)
        await m_gpu0.transition(ModelState.LOADING)
        await m_gpu0.transition(ModelState.ACTIVE)
        await m_gpu1.transition(ModelState.LOADING)
        await m_gpu1.transition(ModelState.ACTIVE)
        assert {m.id for m in reg.list_on_gpu(0)} == {"g0"}
        assert {m.id for m in reg.list_on_gpu(1)} == {"g1"}
        assert len(reg.list_on_gpu()) == 2  # 所有 GPU

    async def test_get_active_version(self) -> None:
        reg = ModelRegistry()
        m1 = _make_model("v1")
        await reg.register(m1)
        await m1.transition(ModelState.LOADING)
        await m1.transition(ModelState.ACTIVE)
        active = reg.get_active_version("qwen-7b")
        assert active is not None
        assert active.id == "v1"

    async def test_get_active_version_none_when_no_active(self) -> None:
        reg = ModelRegistry()
        await reg.register(_make_model("v1"))  # IDLE
        assert reg.get_active_version("qwen-7b") is None

    async def test_get_active_version_none_when_wrong_name(self) -> None:
        reg = ModelRegistry()
        m = _make_model()
        await reg.register(m)
        await m.transition(ModelState.LOADING)
        await m.transition(ModelState.ACTIVE)
        assert reg.get_active_version("nonexistent-model") is None

    async def test_list_versions(self) -> None:
        reg = ModelRegistry()
        await reg.register(_make_model("v1"))
        await reg.register(_make_model("v2"))
        # _make_model 用 model_name="qwen-7b"
        versions = reg.list_versions("qwen-7b")
        assert {m.id for m in versions} == {"v1", "v2"}
        assert reg.list_versions("other-model") == []
