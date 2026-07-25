"""router/middleware.py per-request metrics 采集测试（SPEC §5.4）。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from vllmonline.router.middleware import RequestMetrics, instrument_stream


class TestRequestMetrics:
    async def test_start_increments_in_flight(self) -> None:
        from vllmonline.metrics import REQUESTS_IN_FLIGHT

        m = RequestMetrics(model_version="v1")
        before = REQUESTS_IN_FLIGHT.labels(model_version="v1")._value.get()
        m.start()
        after = REQUESTS_IN_FLIGHT.labels(model_version="v1")._value.get()
        assert after == before + 1
        m.on_success()  # 清理
        final = REQUESTS_IN_FLIGHT.labels(model_version="v1")._value.get()
        assert final == before

    async def test_on_token_records_ttft(self) -> None:
        m = RequestMetrics(model_version="v2")
        m.start()
        await asyncio.sleep(0.01)
        m.on_token()  # 第一个 token → TTFT
        assert m.first_token_time is not None
        assert m.ttft_seconds is not None
        assert m.ttft_seconds > 0
        assert m.token_count == 1

    async def test_on_token_records_tpot(self) -> None:
        """第二个 token 起，每次记录 TPOT。"""
        m = RequestMetrics(model_version="v2")
        m.start()
        m.on_token()
        await asyncio.sleep(0.01)
        m.on_token()  # 第二个 → TPOT
        assert m.token_count == 2
        assert m.last_token_time is not None

    async def test_on_success(self) -> None:
        m = RequestMetrics(model_version="v1")
        m.start()
        m.on_success()
        assert m.success is True

    async def test_on_error(self) -> None:
        m = RequestMetrics(model_version="v1")
        m.start()
        m.on_error("timeout")
        assert m.success is False
        assert m.error_type == "timeout"

    async def test_throughput_calculation(self) -> None:
        m = RequestMetrics(model_version="v1")
        m.start()
        for _ in range(5):
            m.on_token()
        # throughput = 5 / duration
        assert m.throughput_tokens_per_second is not None
        assert m.throughput_tokens_per_second > 0

    async def test_throughput_none_when_no_tokens(self) -> None:
        m = RequestMetrics(model_version="v1")
        m.start()
        assert m.throughput_tokens_per_second is None


class TestInstrumentStream:
    async def test_instruments_sse_stream(self) -> None:
        """包装 SSE 流，解析 content chunk 调 on_token。"""
        chunks = [
            b'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"hello "},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":null}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        async def upstream() -> AsyncIterator[bytes]:
            for c in chunks:
                yield c

        metrics = RequestMetrics(model_version="v1")
        metrics.start()

        received = []
        async for chunk in instrument_stream(upstream(), metrics):
            received.append(chunk)

        # 透传所有原始 chunk
        assert b"".join(received) == b"".join(chunks)
        # 2 个 content chunk → 2 tokens（role chunk 不算）
        assert metrics.token_count == 2
        assert metrics.success is True

    async def test_instrument_handles_partial_chunks(self) -> None:
        """跨 chunk 的 SSE 行也能解析。"""
        # 一个完整的 SSE 行被拆成两半
        part1 = b'data: {"choices":[{"delta":{"content":"x"'
        part2 = b'},"finish_reason":null}]}\n\n'

        async def upstream() -> AsyncIterator[bytes]:
            yield part1
            yield part2

        metrics = RequestMetrics(model_version="v1")
        metrics.start()
        async for _ in instrument_stream(upstream(), metrics):
            pass
        assert metrics.token_count == 1

    async def test_instrument_marks_error_on_exception(self) -> None:
        async def upstream() -> AsyncIterator[bytes]:
            yield b"data: valid\n\n"
            raise RuntimeError("upstream broke")

        metrics = RequestMetrics(model_version="v1")
        metrics.start()
        with pytest.raises(RuntimeError, match="upstream broke"):
            async for _ in instrument_stream(upstream(), metrics):
                pass
        assert metrics.success is False
        assert metrics.error_type == "stream_error"
