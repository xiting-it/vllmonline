"""canary/metrics_collector.py 测试。"""

from __future__ import annotations

import math

import pytest

from vllmonline.canary.metrics_collector import (
    MetricsCollector,
    collect_version_metrics_from_samples,
    estimate_percentile_from_buckets,
    mean,
    percentile,
)


class TestPercentile:
    def test_basic(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(data, 50) == pytest.approx(3.0)
        assert percentile(data, 0) == pytest.approx(1.0)
        assert percentile(data, 100) == pytest.approx(5.0)

    def test_p50_even_count(self) -> None:
        """偶数样本的 P50 是中间两个的平均。"""
        data = [1.0, 2.0, 3.0, 4.0]
        assert percentile(data, 50) == pytest.approx(2.5)

    def test_empty_returns_none(self) -> None:
        assert percentile([], 50) is None

    def test_single_value(self) -> None:
        assert percentile([42.0], 99) == 42.0

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError):
            percentile([1.0], -1)
        with pytest.raises(ValueError):
            percentile([1.0], 101)

    def test_matches_numpy(self) -> None:
        import numpy as np

        data = [0.3, 0.7, 0.1, 0.9, 0.5, 0.2, 0.8, 0.4, 0.6]
        for p in [10, 25, 50, 75, 90, 99]:
            assert percentile(data, p) == pytest.approx(float(np.percentile(data, p)))


class TestMean:
    def test_basic(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty(self) -> None:
        assert mean([]) is None


class TestEstimateFromBuckets:
    def test_basic(self) -> None:
        """4 个样本在 4 个 bucket 各 1 个，P50 应在中间。"""
        buckets = {0.05: 1, 0.1: 2, 0.25: 3, 0.5: 4, math.inf: 4}
        p50 = estimate_percentile_from_buckets(buckets, 50)
        assert p50 is not None
        # P50 落在第 2-3 个 bucket 之间
        assert 0.1 <= p50 <= 0.25

    def test_empty_returns_none(self) -> None:
        assert estimate_percentile_from_buckets({}, 50) is None

    def test_all_zero_returns_none(self) -> None:
        assert estimate_percentile_from_buckets({0.1: 0, math.inf: 0}, 50) is None

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError):
            estimate_percentile_from_buckets({0.1: 1}, -1)


class TestCollectFromSamples:
    def test_basic(self) -> None:
        ttft = [0.1, 0.15, 0.2, 0.25, 0.3]
        tpot = [0.01, 0.02, 0.03]
        m = collect_version_metrics_from_samples(ttft, tpot, error_count=1, total_count=100)
        assert m.ttft_p50 is not None
        assert m.tpot_mean == pytest.approx(0.02)
        assert m.error_rate == pytest.approx(0.01)

    def test_empty_samples(self) -> None:
        m = collect_version_metrics_from_samples([], [])
        assert m.ttft_p50 is None
        assert m.tpot_mean is None
        assert m.error_rate is None  # total_count=0

    def test_throughput_passed_through(self) -> None:
        m = collect_version_metrics_from_samples([0.1], [0.02], throughput_value=150.0)
        assert m.throughput == 150.0


class TestMetricsCollector:
    async def test_default_returns_empty_metrics(self) -> None:
        """无 prometheus_url 时返回空 VersionMetrics（本地路径）。"""
        c = MetricsCollector()
        m = await c.collect_version_metrics("v1")
        assert m.ttft_p50 is None
        assert m.error_rate is None
