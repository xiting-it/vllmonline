"""Per-request metrics 采集（SPEC §5.4）。

记录每个代理请求的：
    - TTFT（首 token 延迟，仅 streaming）
    - TPOT（每 token 生成时间）
    - 吞吐量
    - 错误率
    - 在飞请求数

按 model_version label 写入 Prometheus，方便 v1 vs v2 对比。

实现方式：在 streaming 响应的 chunk 流里测量时间点。
非 streaming 请求只能测总时长（无 TTFT/TPOT 细分）。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from vllmonline.metrics import (
    REQUESTS_IN_FLIGHT,
    REQUESTS_TOTAL,
    TPOT_SECONDS,
    TTFT_SECONDS,
    record_proxy_request,
)


@dataclass(slots=True)
class RequestMetrics:
    """单个请求的 metrics 采集器。

    用法：
        m = RequestMetrics(model_version="v2")
        m.start()
        try:
            # streaming 时每个 chunk 调 m.on_token()
            ...
            m.on_success()
        except Exception:
            m.on_error("exception")
    """

    model_version: str
    start_time: float = 0.0
    first_token_time: float | None = None
    last_token_time: float | None = None
    token_count: int = 0
    success: bool | None = None
    error_type: str | None = None
    # 自定义标签（未来扩展）
    extra_labels: dict[str, str] = field(default_factory=dict)

    def start(self) -> None:
        """请求开始时调用。"""
        self.start_time = time.monotonic()
        REQUESTS_IN_FLIGHT.labels(model_version=self.model_version).inc()

    def on_token(self) -> None:
        """收到一个 token 时调用（streaming）。"""
        now = time.monotonic()
        if self.first_token_time is None:
            self.first_token_time = now
            # TTFT = first_token - start
            ttft = now - self.start_time
            TTFT_SECONDS.labels(model_version=self.model_version).observe(ttft)
        if self.last_token_time is not None and self.token_count > 0:
            # TPOT = current - last
            tpot = now - self.last_token_time
            TPOT_SECONDS.labels(model_version=self.model_version).observe(tpot)
        self.last_token_time = now
        self.token_count += 1

    def on_success(self) -> None:
        """请求成功完成。"""
        self.success = True
        self._finalize(status="success")

    def on_error(self, error_type: str = "error") -> None:
        """请求失败。"""
        self.success = False
        self.error_type = error_type
        self._finalize(status="error", error_type=error_type)

    def _finalize(self, *, status: str, error_type: str | None = None) -> None:
        """收尾：写 Counter + 在飞 -1。"""
        REQUESTS_TOTAL.labels(
            model_version=self.model_version,
            status=status,
            error_type=error_type or "none",
        ).inc()
        REQUESTS_IN_FLIGHT.labels(model_version=self.model_version).dec()
        # 兼容 P0 的 record_proxy_request（避免重复计数逻辑分裂）
        # 注意：这里不再调 record_proxy_request，因为 REQUESTS_TOTAL 已在上面 inc
        _ = record_proxy_request  # 防止未使用导入

    # ── 计算派生指标（供 metrics_collector 拉取）──

    @property
    def duration_seconds(self) -> float:
        """总时长（start 到最后一个 token 或完成）。"""
        end = self.last_token_time or time.monotonic()
        return end - self.start_time if self.start_time > 0 else 0.0

    @property
    def ttft_seconds(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.start_time

    @property
    def throughput_tokens_per_second(self) -> float | None:
        """吞吐量 = token_count / duration。"""
        if self.token_count == 0 or self.duration_seconds <= 0:
            return None
        return self.token_count / self.duration_seconds


# ─────────────────────────────────────────────────────────────────────────────
# streaming 包装器：在 chunk 流里采集 metrics
# ─────────────────────────────────────────────────────────────────────────────


async def instrument_stream(
    upstream: AsyncIterator[bytes],
    metrics: RequestMetrics,
) -> AsyncIterator[bytes]:
    """包装上游 chunk 流，解析 SSE 并采集 TTFT/TPOT。

    对每个 chunk：
        - 解析 SSE data 行
        - 如果是 content chunk → metrics.on_token()
        - 透传原始字节（不修改）
    """
    import json

    buffer = b""
    try:
        async for chunk in upstream:
            if not chunk:
                continue
            # 尝试解析 SSE 行（可能跨 chunk，需要 buffer）
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line_str = line.decode(errors="replace").strip()
                if not line_str.startswith("data: "):
                    continue
                payload = line_str[6:]
                if payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    metrics.on_token()
            yield chunk
        metrics.on_success()
    except Exception:
        metrics.on_error("stream_error")
        raise
