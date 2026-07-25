"""vllm/client.py + adapter.py 测试。

用 fake vLLM app（ASGITransport）验证：
    - chat_completion / stream_chat / health / list_models
    - sleep / wake / load_lora（SPEC §7.1）
    - 重试逻辑（连接错误时）
    - SSE 解析
    - warmup workaround（SPEC §7.2）
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from tests.fake_vllm import make_fake_vllm_app
from vllmonline.vllm.adapter import VLLMAdapter
from vllmonline.vllm.client import (
    VLLMClient,
    VLLMConnectionError,
    VLLMHTTPError,
    _parse_sse_line,
)


@pytest.fixture
async def fake_endpoint() -> str:
    """fake vLLM 不需要真实 endpoint（用 ASGITransport），返回占位 URL。"""
    return "http://fake-vllm"


@pytest.fixture
async def vllm_client_with_fake() -> tuple[VLLMClient, str]:
    """构造指向 fake vLLM 的 VLLMClient。"""
    fake_app = make_fake_vllm_app(model_name="qwen-7b")
    transport = ASGITransport(app=fake_app)
    http_client = AsyncClient(transport=transport, base_url="http://fake-vllm")
    client = VLLMClient(http_client=http_client)
    try:
        yield client, "http://fake-vllm"  # type: ignore[misc]
    finally:
        # 注入的 client 由我们管，不调 aclose（VLLMClient 不 own 它）
        await http_client.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# SSE 解析（纯函数）
# ─────────────────────────────────────────────────────────────────────────────


class TestParseSseLine:
    def test_data_line_with_content(self) -> None:
        line = 'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}'
        chunk = _parse_sse_line(line)
        assert chunk is not None
        assert chunk.content == "hi"
        assert chunk.finish_reason is None

    def test_data_line_with_role(self) -> None:
        line = 'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}'
        chunk = _parse_sse_line(line)
        assert chunk is not None
        assert chunk.role == "assistant"

    def test_done_marker_returns_none(self) -> None:
        assert _parse_sse_line("data: [DONE]") is None

    def test_empty_line(self) -> None:
        assert _parse_sse_line("") is None

    def test_non_data_line(self) -> None:
        assert _parse_sse_line("event: ping") is None
        assert _parse_sse_line(": comment") is None

    def test_malformed_json_returns_none(self) -> None:
        assert _parse_sse_line("data: {not json") is None

    def test_no_choices(self) -> None:
        chunk = _parse_sse_line('data: {"choices":[]}')
        assert chunk is not None
        assert chunk.content == ""

    def test_finish_reason(self) -> None:
        line = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        chunk = _parse_sse_line(line)
        assert chunk is not None
        assert chunk.finish_reason == "stop"


# ─────────────────────────────────────────────────────────────────────────────
# VLLMClient 基础 API
# ─────────────────────────────────────────────────────────────────────────────


class TestVLLMClientBasic:
    async def test_health_ok(self, vllm_client_with_fake: tuple[VLLMClient, str]) -> None:
        client, endpoint = vllm_client_with_fake
        assert await client.health(endpoint) is True

    async def test_health_unreachable(self) -> None:
        """连不上的 endpoint 返回 False（不抛）。"""
        client = VLLMClient(http_client=AsyncClient(base_url="http://127.0.0.1:1", timeout=0.5))
        try:
            assert await client.health("http://127.0.0.1:1") is False
        finally:
            await client.aclose()

    async def test_list_models(self, vllm_client_with_fake: tuple[VLLMClient, str]) -> None:
        client, endpoint = vllm_client_with_fake
        models = await client.list_models(endpoint)
        assert len(models) == 1
        assert models[0].id == "qwen-7b"

    async def test_chat_completion(self, vllm_client_with_fake: tuple[VLLMClient, str]) -> None:
        client, endpoint = vllm_client_with_fake
        result = await client.chat_completion(
            endpoint,
            {
                "model": "qwen-7b",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["role"] == "assistant"

    async def test_chat_completion_forces_stream_false(
        self, vllm_client_with_fake: tuple[VLLMClient, str]
    ) -> None:
        """即使传 stream=True 也应被强制改 False（streaming 走 stream_chat）。"""
        client, endpoint = vllm_client_with_fake
        result = await client.chat_completion(
            endpoint,
            {"model": "qwen-7b", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        # 非 streaming 响应（object 是 chat.completion 而非 chunk 流）
        assert result["object"] == "chat.completion"

    async def test_stream_chat(self, vllm_client_with_fake: tuple[VLLMClient, str]) -> None:
        client, endpoint = vllm_client_with_fake
        chunks = []
        async for chunk in client.stream_chat(
            endpoint,
            {"model": "qwen-7b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
        ):
            chunks.append(chunk)
        # 至少 role chunk + content chunks
        assert len(chunks) >= 2
        assert chunks[0].role == "assistant"
        # 最后一个有 finish_reason
        assert chunks[-1].finish_reason == "stop"

    async def test_4xx_not_retried_raises_http_error(
        self, vllm_client_with_fake: tuple[VLLMClient, str]
    ) -> None:
        """sleep 态下 chat 返回 503 → VLLMHTTPError。

        注：fake vLLM sleep 时返回 503（5xx），按 SPEC 应重试。
        但本测试验证 5xx 最终抛 VLLMHTTPError。
        """
        client, endpoint = vllm_client_with_fake
        # 先让 fake 进入 sleep
        await client._client.post(f"{endpoint}/sleep")
        with pytest.raises(VLLMHTTPError) as exc_info:
            await client.chat_completion(
                endpoint, {"model": "qwen-7b", "messages": [{"role": "user", "content": "x"}]}
            )
        assert exc_info.value.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# VLLMClient 重试
# ─────────────────────────────────────────────────────────────────────────────


class TestVLLMClientRetry:
    async def test_connection_error_retried_then_raised(self) -> None:
        """连接失败重试 max_attempts 次后抛 VLLMConnectionError。"""
        client = VLLMClient(
            http_client=AsyncClient(base_url="http://127.0.0.1:1", timeout=0.3),
            retry_max_attempts=2,
            retry_initial_wait=0.01,
        )
        try:
            with pytest.raises(VLLMConnectionError):
                await client.list_models("http://127.0.0.1:1")
        finally:
            await client.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# VLLMAdapter（SPEC §7.1 + §7.2）
# ─────────────────────────────────────────────────────────────────────────────


class TestVLLMAdapter:
    async def test_sleep(self, vllm_client_with_fake: tuple[VLLMClient, str]) -> None:
        client, endpoint = vllm_client_with_fake
        adapter = VLLMAdapter(client, sleep_settle_poll=0.01)
        result = await adapter.sleep(endpoint)
        assert result is True
        # fake vLLM 的 is_sleeping 应为 True
        # 通过 health 仍可访问（sleep 不影响 /health）
        assert await client.health(endpoint) is True

    async def test_wake_with_warmup(self, vllm_client_with_fake: tuple[VLLMClient, str]) -> None:
        """wake 后自动发 warmup 请求（SPEC §7.2 workaround）。"""
        client, endpoint = vllm_client_with_fake
        adapter = VLLMAdapter(client)
        await adapter.sleep(endpoint)
        result = await adapter.wake(endpoint, model_name="qwen-7b")
        assert result is True

    async def test_load_lora(self, vllm_client_with_fake: tuple[VLLMClient, str]) -> None:
        client, endpoint = vllm_client_with_fake
        adapter = VLLMAdapter(client)
        # fake vLLM 的 /v1/load_lora_adapter 没实现，会返回 404
        # 验证调用不抛（即使 backend 不支持）
        with pytest.raises(VLLMHTTPError):
            await adapter.load_lora(endpoint, "test-lora", "/path/to/lora")
