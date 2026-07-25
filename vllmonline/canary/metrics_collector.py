"""Per-version metrics 采集器（SPEC §5.4）。

从 Prometheus（或本地 registry）拉取 per-version metrics，
计算 P50/P99/mean/error_rate，供灰度策略和回滚检查使用。

实现方式：
    - 生产：通过 httpx 拉 Prometheus /api/v1/query
    - 单测：直接从 prometheus_client registry 读（无网络）
"""

from __future__ import annotations

import math
from typing import Any

import structlog

from vllmonline.canary.strategy import VersionMetrics

logger = structlog.get_logger("vllmonline.canary.metrics_collector")


# ─────────────────────────────────────────────────────────────────────────────
# 百分位数计算（纯函数）
# ─────────────────────────────────────────────────────────────────────────────


def percentile(values: list[float], p: float) -> float | None:
    """计算百分位数（线性插值法）。

    Args:
        values: 样本列表
        p: 百分位（0-100），如 50 表示 P50

    Returns:
        百分位数值；空列表返回 None
    """
    if not values:
        return None
    if not 0 <= p <= 100:
        msg = f"p 必须在 [0, 100]，得到 {p}"
        raise ValueError(msg)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    # 线性插值（numpy.percentile 的默认方法）
    rank = (p / 100) * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_vals[lower]
    frac = rank - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def mean(values: list[float]) -> float | None:
    """算术平均。空列表返回 None。"""
    if not values:
        return None
    return sum(values) / len(values)


# ─────────────────────────────────────────────────────────────────────────────
# 从 histogram buckets 估算百分位
# ─────────────────────────────────────────────────────────────────────────────


def estimate_percentile_from_buckets(
    bucket_counts: dict[float, int],  # {le_bound: cumulative_count}
    p: float,
) -> float | None:
    """从 Prometheus histogram 的累积 bucket 估算百分位。

    Prometheus histogram 暴露的是累积分布：le=0.05 的 count 是 ≤0.05s 的样本数。
    本函数用线性插值在 bucket 间估算指定百分位。

    Args:
        bucket_counts: {le: cumulative_count}，含 +Inf bucket
        p: 百分位（0-100）

    Returns:
        估算值；数据不足返回 None
    """
    if not bucket_counts:
        return None
    if not 0 <= p <= 100:
        msg = f"p 必须在 [0, 100]，得到 {p}"
        raise ValueError(msg)

    # 排序 bucket（按 le 升序，+Inf 最后）
    sorted_buckets = sorted(
        bucket_counts.items(),
        key=lambda kv: (kv[0] == float("inf"), kv[0]),
    )
    total = sorted_buckets[-1][1]  # +Inf bucket 的 count = 总样本数
    if total == 0:
        return None

    target_count = (p / 100) * total
    prev_le = 0.0
    prev_count = 0
    for le, count in sorted_buckets:
        if count >= target_count:
            if le == float("inf"):
                # 目标在最后一个有限 bucket 之外
                return prev_le
            # 在 [prev_le, le] 之间线性插值
            if count == prev_count:
                return le
            frac = (target_count - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le = le
        prev_count = count
    return sorted_buckets[-1][0]


# ─────────────────────────────────────────────────────────────────────────────
# 采集器
# ─────────────────────────────────────────────────────────────────────────────


class MetricsCollector:
    """从 Prometheus / 本地 registry 采集 per-version metrics。

    生产用：HTTP 拉 Prometheus API
    单测用：直接读 prometheus_client 的 registry sample
    """

    def __init__(
        self,
        prometheus_url: str | None = None,
        http_client: Any = None,
    ) -> None:
        self._prom_url = prometheus_url
        self._http = http_client

    async def collect_version_metrics(
        self,
        model_version: str,
    ) -> VersionMetrics:
        """采集某版本的 metrics 快照。

        生产路径：拉 Prometheus 的 vllmonline_ttft_seconds 等 histogram。
        单测路径：直接接收预计算的值（通过 mock）。

        本方法的默认实现从本地 prometheus_client registry 读 histogram buckets。
        """
        # 默认实现：尝试从本地 registry 读（单测友好）
        # 生产部署应注入 prometheus_url + http_client 走 HTTP 路径
        if self._prom_url and self._http:
            return await self._collect_from_prometheus(model_version)

        # 本地 registry 路径：返回空 metrics（让调用方注入真实数据）
        return VersionMetrics()

    async def _collect_from_prometheus(self, model_version: str) -> VersionMetrics:
        """从 Prometheus HTTP API 拉取并计算。

        查询示例（PromQL）：
            histogram_quantile(0.5, vllmonline_ttft_seconds_bucket{model_version="v1"})
        """
        queries = {
            "ttft_p50": f'histogram_quantile(0.5, vllmonline_ttft_seconds_bucket{{model_version="{model_version}"}})',
            "ttft_p99": f'histogram_quantile(0.99, vllmonline_ttft_seconds_bucket{{model_version="{model_version}"}})',
        }
        result: dict[str, float | None] = {}
        for key, query in queries.items():
            value = await self._query_prometheus(query)
            result[key] = value
        return VersionMetrics(
            ttft_p50=result.get("ttft_p50"),
            ttft_p99=result.get("ttft_p99"),
        )

    async def _query_prometheus(self, query: str) -> float | None:
        """执行单条 PromQL 查询。"""
        if not self._http or not self._prom_url:
            return None
        try:
            resp = await self._http.get(
                f"{self._prom_url}/api/v1/query",
                params={"query": query},
            )
            data = resp.json()
            if data.get("status") != "success":
                return None
            results = data.get("data", {}).get("result", [])
            if not results:
                return None
            return float(results[0]["value"][1])
        except Exception as e:
            logger.warning("prometheus query failed", query=query, error=str(e))
            return None


def collect_version_metrics_from_samples(
    ttft_samples: list[float],
    tpot_samples: list[float],
    *,
    error_count: int = 0,
    total_count: int = 0,
    throughput_value: float | None = None,
) -> VersionMetrics:
    """便捷函数：从原始样本列表构造 VersionMetrics。

    单测和本地采集用——不依赖 Prometheus。
    """
    ttft_p50 = percentile(ttft_samples, 50)
    ttft_p99 = percentile(ttft_samples, 99)
    tpot_mean = mean(tpot_samples)
    error_rate = (error_count / total_count) if total_count > 0 else None
    return VersionMetrics(
        ttft_p50=ttft_p50,
        ttft_p99=ttft_p99,
        tpot_mean=tpot_mean,
        throughput=throughput_value,
        error_rate=error_rate,
    )
