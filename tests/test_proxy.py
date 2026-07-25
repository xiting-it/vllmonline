"""Phase 0 基础测试：服务存活、健康检查、代理透传、streaming。

验收对应 PLAN.md P0：
    - POST /v1/chat/completions 能正常返回模型回答 ✓
    - streaming 请求逐 chunk 透传 ✓
    - /healthz 返回 ok ✓
    - /readyz 检查 vLLM backend 连通性 ✓
"""

from __future__ import annotations

import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from tests.fake_vllm import make_fake_vllm_app


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
    """readyz 失败——backend 不可达。"""
    from vllmonline import server
    from vllmonline.config import get_settings

    get_settings.cache_clear()

    app = server.create_app()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan_with_dead_backend(a):  # type: ignore[no-untyped-def]
        async with app.router.lifespan_context(a):
            state = server.AppState.from_app(a)
            await state.http_client.aclose()
            # 指向一个必然不可达的端口
            state.http_client = AsyncClient(
                transport=httpx.HTTPTransport(),
                base_url="http://127.0.0.1:1",  # 1 号端口：几乎肯定没服务
            )
            state.default_backend_url = "http://127.0.0.1:1"
            yield

    app.router.lifespan_context = _lifespan_with_dead_backend  # type: ignore[method-assign]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/readyz")
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


async def test_proxy_returns_502_on_upstream_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """upstream 返回 500 时，proxy 把状态透传（不吞掉）。"""
    # 用一个会主动失败的 fake
    failing_fake = make_fake_vllm_app(error_rate=1.0)  # 100% 失败

    from vllmonline import server
    from vllmonline.config import get_settings
    from contextlib import asynccontextmanager

    get_settings.cache_clear()
    app = server.create_app()

    @asynccontextmanager
    async def _lifespan(a):  # type: ignore[no-untyped-def]
        async with app.router.lifespan_context(a):
            state = server.AppState.from_app(a)
            await state.http_client.aclose()
            state.http_client = AsyncClient(
                transport=ASGITransport(app=failing_fake),
                base_url="http://fake-vllm",
            )
            state.default_backend_url = "http://fake-vllm"
            yield

    app.router.lifespan_context = _lifespan  # type: ignore[method-assign]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )
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
