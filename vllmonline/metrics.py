"""Prometheus metrics 定义与采集。

所有指标都按 SPEC §5.4 设计，带 `model_version` label，方便 v1 vs v2 对比。

Phase 0 仅暴露 vllmonline 自身的存活/代理指标；
P3 起补充 per-version 的 TTFT/TPOT/throughput/in_flight 指标。

线程安全：prometheus_client 的 metric 操作内部加锁，可在 asyncio 中直接调用
（注意：在 event loop 里调用同步 prometheus_client 操作是 OK 的，因为它们是
CPU-only 的快速字典更新，不会阻塞）。
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Registry, get_default_registry

# 用一个独立 Registry，方便测试时隔离（避免全局污染）。
# 生产里我们其实用默认 registry（这样 /metrics 端点能自动 expose）。
# 这里保留 REGISTRY 作为对外名字，测试时 monkeypatch 可换。
REGISTRY: Registry = get_default_registry()

# ─────────────────────────────────────────────────────────────────────────────
# Metric 定义（模块级单例）
# ─────────────────────────────────────────────────────────────────────────────

# 代理请求总数（Phase 0 已用）
REQUESTS_TOTAL = Counter(
    name="vllmonline_requests_total",
    documentation="代理请求总数",
    labelnames=["model_version", "status", "error_type"],
    registry=REGISTRY,
)

# 错误总数
ERRORS_TOTAL = Counter(
    name="vllmonline_errors_total",
    documentation="代理错误总数",
    labelnames=["model_version", "error_type"],
    registry=REGISTRY,
)

# 在飞请求数
REQUESTS_IN_FLIGHT = Gauge(
    name="vllmonline_requests_in_flight",
    documentation="当前在飞请求数",
    labelnames=["model_version"],
    registry=REGISTRY,
)

# TTFT（首 token 延迟）—— P3 middleware 填充
TTFT_SECONDS = Histogram(
    name="vllmonline_ttft_seconds",
    documentation="首 token 延迟（Time To First Token）",
    labelnames=["model_version"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=REGISTRY,
)

# TPOT（每 token 生成时间）—— P3 middleware 填充
TPOT_SECONDS = Histogram(
    name="vllmonline_tpot_seconds",
    documentation="每 token 生成时间（Time Per Output Token）",
    labelnames=["model_version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
    registry=REGISTRY,
)

# 吞吐量
THROUGHPUT = Gauge(
    name="vllmonline_throughput_tokens_per_second",
    documentation="吞吐量（tokens/s）",
    labelnames=["model_version"],
    registry=REGISTRY,
)


# ─────────────────────────────────────────────────────────────────────────────
# 采集辅助函数
# ─────────────────────────────────────────────────────────────────────────────

# 默认 model_version：Phase 0 代理还没接路由表，统一标 "unknown"。
# P3 起从 x-model-version header 读真实版本。
_DEFAULT_VERSION = "unknown"


def record_proxy_request(
    *,
    status: str,
    error_type: str | None = None,
    model_version: str = _DEFAULT_VERSION,
) -> None:
    """记录一次代理请求的结果。

    Args:
        status: "success" | "error"
        error_type: 失败时归类（upstream_unreachable / http_500 / stream_error / ...）
        model_version: 被路由到的模型版本
    """
    REQUESTS_TOTAL.labels(
        model_version=model_version,
        status=status,
        error_type=error_type or "none",
    ).inc()
    if status == "error" and error_type is not None:
        ERRORS_TOTAL.labels(model_version=model_version, error_type=error_type).inc()


def init_metrics(settings: object) -> None:
    """应用启动时调用：确保所有 metric collector 注册。

    prometheus_client 的全局/独立 Registry 在模块导入时已注册；
    这里只是占位，未来按 settings 配置 buckets 等。
    """
    # 当前 buckets 已在定义处硬编码（SPEC §5.4），无需额外动作。
    # 显式引用避免被 mypy/ruff 认为是未使用导入。
    assert REQUESTS_TOTAL is not None
    assert TTFT_SECONDS is not None
    _ = settings
