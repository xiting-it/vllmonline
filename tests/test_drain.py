"""drain.py 测试（SPEC §4 DRAINING 态）。"""

from __future__ import annotations

import asyncio

import pytest

from vllmonline.router.drain import Drainer, drain_and_sleep
from vllmonline.scheduler.gpu_memory import build_model_profile
from vllmonline.scheduler.lifecycle import Model
from vllmonline.scheduler.types import ModelState


def _active_model(model_id: str = "m1") -> Model:
    """构造一个已经 ACTIVE 的 model（手动设 state）。"""
    m = Model(
        id=model_id,
        model_name="qwen-7b",
        version="v1",
        endpoint="http://x",
        gpu_id=0,
        memory_profile=build_model_profile(7.0, "fp16"),
    )
    # 直接走状态机到 ACTIVE
    return m


async def _to_active(m: Model) -> Model:
    await m.transition(ModelState.LOADING)
    await m.transition(ModelState.ACTIVE)
    return m


class TestDrainerStart:
    async def test_start_transitions_to_draining(self) -> None:
        m = await _to_active(_active_model())
        drainer = Drainer(m, poll_interval=0.01)
        await drainer.start()
        assert m.state is ModelState.DRAINING

    async def test_start_idempotent_when_already_draining(self) -> None:
        m = await _to_active(_active_model())
        drainer = Drainer(m, poll_interval=0.01)
        await drainer.start()
        await drainer.start()  # 不抛
        assert m.state is ModelState.DRAINING

    async def test_start_from_non_active_raises(self) -> None:
        m = _active_model()  # IDLE
        drainer = Drainer(m)
        with pytest.raises(ValueError, match="必须 ACTIVE"):
            await drainer.start()


class TestDrainerWait:
    async def test_wait_completes_when_pending_zero(self) -> None:
        m = await _to_active(_active_model())
        drainer = Drainer(m, poll_interval=0.01)
        await drainer.start()
        # pending 已经是 0
        completed = await drainer.wait(timeout=1.0)
        assert completed is True

    async def test_wait_times_out_with_pending(self) -> None:
        m = await _to_active(_active_model())
        await m.increment_pending()  # 模拟在飞请求
        drainer = Drainer(m, poll_interval=0.01)
        await drainer.start()
        completed = await drainer.wait(timeout=0.1)
        assert completed is False

    async def test_wait_from_non_draining_raises(self) -> None:
        m = _active_model()  # IDLE，不是 DRAINING
        drainer = Drainer(m)
        with pytest.raises(ValueError, match="不在 DRAINING"):
            await drainer.wait(timeout=0.1)

    async def test_wait_unblocks_when_pending_drops(self) -> None:
        """wait 期间 pending 降到 0 → 立即完成。"""
        m = await _to_active(_active_model())
        await m.increment_pending()
        drainer = Drainer(m, poll_interval=0.01)
        await drainer.start()

        async def drop_pending_after_delay() -> None:
            await asyncio.sleep(0.05)
            await m.decrement_pending()

        results = await asyncio.gather(
            drainer.wait(timeout=2.0),
            drop_pending_after_delay(),
        )
        assert results[0] is True


class TestDrainerForce:
    async def test_force_zeros_pending(self) -> None:
        m = await _to_active(_active_model())
        for _ in range(5):
            await m.increment_pending()
        assert m.pending_requests == 5
        drainer = Drainer(m, poll_interval=0.01)
        await drainer.start()
        await drainer.force()
        assert m.pending_requests == 0


class TestDrainerRun:
    async def test_run_natural_completion(self) -> None:
        m = await _to_active(_active_model())
        drainer = Drainer(m, poll_interval=0.01)
        completed = await drainer.run(timeout=1.0)
        assert completed is True
        assert m.state is ModelState.DRAINING

    async def test_run_force_on_timeout(self) -> None:
        m = await _to_active(_active_model())
        await m.increment_pending()
        drainer = Drainer(m, poll_interval=0.01)
        completed = await drainer.run(timeout=0.1)
        assert completed is False
        # force 后 pending 归零
        assert m.pending_requests == 0


class TestDrainAndSleep:
    async def test_drain_and_sleep_transitions_to_sleeping(self) -> None:
        m = await _to_active(_active_model())
        completed = await drain_and_sleep(m, timeout=1.0)
        assert completed is True
        assert m.state is ModelState.SLEEPING

    async def test_drain_and_sleep_force_still_sleeps(self) -> None:
        """超时 force 后仍进入 SLEEPING（状态机要求）。"""
        m = await _to_active(_active_model())
        await m.increment_pending()
        completed = await drain_and_sleep(m, timeout=0.1)
        assert completed is False
        assert m.state is ModelState.SLEEPING
