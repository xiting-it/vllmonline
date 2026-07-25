"""Phase 0 基础测试：服务存活、健康检查、代理透传、streaming。

验收对应 PLAN.md P0：
    - POST /v1/chat/completions 能正常返回模型回答 ✓
    - streaming 请求逐 chunk 透传 ✓
    - /healthz 返回 ok ✓
    - /readyz 检查 vLLM backend 连通性 ✓
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from tests.fake_vllm import make_fake_vllm_app


@asynccontextmanager
async def app_client(
    app: Any,
    *,
    base_url: str = "http://test",
) -> AsyncIterator[AsyncClient]:
    """驱动 FastAPI lifespan 并返回 ASGITransport client 的辅助上下文。"""
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url=base_url) as ac:
            yield ac


# ─────────────────────────────────────────────────────────────────────────────
# Meta 路由
# ─────────────────────────────────────────────────────────────────────────────


async def test_root(client: AsyncClient) -> None:
    r = await client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "vllmonline"
    assert "version" in body


async def test_healthz(client: AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_readyz_with_fake_backend(client: AsyncClient) -> None:
    """readyz 通过——fake vLLM 的 /health 返回 ok。"""
    r = await client.get("/readyz")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["checks"]["vllm"] == "ok"


async def test_readyz_when_backend_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """readyz 失败——backend 不可达。

    直接 mock http_client.get 抛 ConnectError，避免依赖真实网络行为
    （不同环境下"不可达地址"的表现不一致：可能超时、可能 ConnectionRefused、
    也可能被本地代理拦截返回 502）。
    """
    from vllmonline import server
    from vllmonline.config import get_settings

    get_settings.cache_clear()

    # 构造一个 fake vLLM，但 monkeypatch 它的 /health 路由让 client.get 抛异常
    fake_app = make_fake_vllm_app()

    server.set_test_backend(
        client_factory=lambda: AsyncClient(
            transport=ASGITransport(app=fake_app),
            base_url="http://fake-vllm",
        ),
        backend_url="http://fake-vllm",
    )

    # 让 httpx 在 GET /health 时抛 ConnectError
    original_get = httpx.AsyncClient.get

    async def failing_get(self, url, **kwargs):
        if "/health" in str(url):
            raise httpx.ConnectError("mocked unreachable")
        return await original_get(self, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "get", failing_get)

    try:
        app = server.create_app()
        async with app_client(app) as ac:
            r = await ac.get("/readyz")
    finally:
        server.clear_test_backend()

    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert "unreachable" in body["checks"]["vllm"]


# ─────────────────────────────────────────────────────────────────────────────
# 代理透传
# ─────────────────────────────────────────────────────────────────────────────


async def test_proxy_nonstream_chat(client: AsyncClient) -> None:
    """非 streaming chat completion 透传。"""
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]  # 有内容


async def test_proxy_stream_chat(client: AsyncClient) -> None:
    """streaming chat completion 透传，SSE 不变形。"""
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "max_tokens": 4,
        },
    ) as r:
        assert r.status_code == 200
        chunks: list[dict] = []
        terminal_seen = False
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                terminal_seen = True
                break
            chunks.append(json.loads(payload))

    assert terminal_seen, "应收到 data: [DONE]"
    assert len(chunks) >= 2, "至少应含 role chunk + 1 个 content chunk"
    # 第一个 chunk 应带 role
    first = chunks[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"].get("role") == "assistant"
    # 最后一个非 [DONE] chunk 应带 finish_reason=stop
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "stop"


async def test_proxy_preserves_model_field(client: AsyncClient) -> None:
    """代理不应改写请求体的 model 字段。"""
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "my-custom-model-name",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    # fake vllm 把 model 回显到响应
    assert r.json()["model"] == "my-custom-model-name"


async def test_proxy_returns_502_on_upstream_error() -> None:
    """upstream 返回 500 时，proxy 把状态透传（不吞掉）。"""
    # 用一个会主动失败的 fake
    failing_fake = make_fake_vllm_app(error_rate=1.0)  # 100% 失败

    from vllmonline import server
    from vllmonline.config import get_settings

    get_settings.cache_clear()
    server.set_test_backend(
        client_factory=lambda: AsyncClient(
            transport=ASGITransport(app=failing_fake),
            base_url="http://fake-vllm",
        ),
        backend_url="http://fake-vllm",
    )
    try:
        app = server.create_app()
        async with app_client(app) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            )
    finally:
        server.clear_test_backend()
    assert r.status_code == 500


# ─────────────────────────────────────────────────────────────────────────────
# Metrics 端点
# ─────────────────────────────────────────────────────────────────────────────


async def test_metrics_endpoint(client: AsyncClient) -> None:
    """/metrics 端点能被 Prometheus 抓取。"""
    # 先发一个请求，让 REQUESTS_TOTAL 有计数
    await client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
    )
    r = await client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "vllmonline_requests_total" in text
    # 应至少有一行带 status="success"
    assert 'status="success"' in text


async def test_openapi_available(client: AsyncClient) -> None:
    """OpenAPI schema 可访问（docs 用）。"""
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/v1/chat/completions" in paths
    assert "/healthz" in paths
    assert "/readyz" in paths
    assert "/metrics" in paths
