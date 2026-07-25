"""vLLM API 客户端（SPEC §7.1）。

封装 OpenAI 兼容 API + 高级端点：
    chat_completion / stream_chat / health / list_models

每个外部调用都有超时 + 重试（tenacity）。重试只针对瞬时错误（连接/5xx），
4xx 不重试（请求格式问题，重试无用）。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger("vllmonline.vllm.client")


# ─────────────────────────────────────────────────────────────────────────────
# 数据类
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """vLLM /v1/models 返回的模型信息。"""

    id: str
    owned_by: str = "vllm"


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """streaming 的一个 chunk（OpenAI chat.completion.chunk 格式）。"""

    content: str  # 增量内容（可能为空，如首个 role chunk）
    role: str | None  # 仅首个 chunk 带 role
    finish_reason: str | None  # 仅最后 chunk 非 None


# ─────────────────────────────────────────────────────────────────────────────
# 异常
# ─────────────────────────────────────────────────────────────────────────────


class VLLMError(Exception):
    """vLLM 调用失败的基类。"""


class VLLMConnectionError(VLLMError):
    """无法连接到 vLLM backend。"""


class VLLMHTTPError(VLLMError):
    """vLLM 返回了非 2xx。"""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"vLLM HTTP {status_code}: {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# 客户端
# ─────────────────────────────────────────────────────────────────────────────


class VLLMClient:
    """vLLM backend API 客户端。

    所有方法都接收 endpoint 参数（如 http://vllm:8000），
    因为同一 vllmonline 实例可能管理多个 vLLM backend（v1/v2）。

    线程安全：httpx.AsyncClient 本身协程安全，可被多协程共享。
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 120.0,
        retry_max_attempts: int = 3,
        retry_initial_wait: float = 0.5,
    ) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
        )
        self._retry_max = retry_max_attempts
        self._retry_initial_wait = retry_initial_wait

    async def aclose(self) -> None:
        """如果是自建的 client，关闭它。注入的 client 由调用方管。"""
        if self._owns_client:
            await self._client.aclose()

    # ── 内部：重试装饰 ──
    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self._retry_max),
            wait=wait_exponential(
                multiplier=self._retry_initial_wait, min=self._retry_initial_wait
            ),
            retry=retry_if_exception_type((VLLMConnectionError, httpx.TransportError)),
            reraise=True,
        )

    # ── 基础 API ──

    async def health(self, endpoint: str) -> bool:
        """GET /health —— 存活检查。不重试（健康检查应快速失败）。"""
        try:
            resp = await self._client.get(f"{endpoint}/health", timeout=2.0)
            return resp.is_success
        except httpx.HTTPError:
            return False

    async def list_models(self, endpoint: str) -> list[ModelInfo]:
        """GET /v1/models —— 列出可用模型。"""
        resp = await self._request("GET", endpoint, "/v1/models")
        data = resp.json()
        return [
            ModelInfo(id=m["id"], owned_by=m.get("owned_by", "vllm")) for m in data.get("data", [])
        ]

    async def chat_completion(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST /v1/chat/completions（非 streaming）。

        Args:
            endpoint: vLLM 服务 URL（如 http://vllm:8000）
            payload: OpenAI chat completion 请求体

        Returns:
            完整响应 dict
        """
        # 强制 stream=False（streaming 走 stream_chat）
        payload = {**payload, "stream": False}
        resp = await self._request("POST", endpoint, "/v1/chat/completions", json=payload)
        result: dict[str, Any] = resp.json()
        return result

    async def stream_chat(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        """POST /v1/chat/completions（streaming）。

        逐 chunk yield StreamChunk，直到收到 [DONE]。
        """
        payload = {**payload, "stream": True}
        url = f"{endpoint}/v1/chat/completions"

        async with self._client.stream("POST", url, json=payload) as resp:
            if not resp.is_success:
                body = await resp.aread()
                raise VLLMHTTPError(resp.status_code, body.decode(errors="replace"))
            async for line in resp.aiter_lines():
                chunk = _parse_sse_line(line)
                if chunk is not None:
                    yield chunk

    # ── 内部：统一请求 + 错误映射 ──

    async def _request(
        self,
        method: str,
        endpoint: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """带重试的请求。把 httpx 错误映射成 VLLMError 子类。"""
        url = f"{endpoint}{path}"

        async for attempt in self._retrying():
            with attempt:
                try:
                    resp = await self._client.request(method, url, **kwargs)
                except httpx.ConnectError as e:
                    raise VLLMConnectionError(f"无法连接 {endpoint}: {e}") from e
                except httpx.TransportError as e:
                    raise VLLMConnectionError(f"传输错误 {endpoint}: {e}") from e

                if not resp.is_success:
                    # 4xx 不重试（请求格式问题）
                    if 400 <= resp.status_code < 500:
                        raise VLLMHTTPError(resp.status_code, resp.text)
                    # 5xx 抛 retryable，让 tenacity 决定
                    raise VLLMHTTPError(resp.status_code, resp.text)
                return resp

        # tenacity reraise=True 不会走到这里，但 mypy 需要
        msg = "unreachable"
        raise RuntimeError(msg)


def _parse_sse_line(line: str) -> StreamChunk | None:
    """解析一行 SSE，返回 StreamChunk。

    - "data: [DONE]" → None（流终止信号，调用方应停止迭代）
    - "data: {...}" → StreamChunk
    - 其他（空行、event:、注释）→ None
    """
    line = line.strip()
    if not line or not line.startswith("data: "):
        return None
    payload = line[6:]
    if payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("malformed SSE line", line=line)
        return None

    choices = obj.get("choices", [])
    if not choices:
        return StreamChunk(content="", role=None, finish_reason=None)
    delta = choices[0].get("delta", {})
    return StreamChunk(
        content=delta.get("content", "") or "",
        role=delta.get("role"),
        finish_reason=choices[0].get("finish_reason"),
    )
