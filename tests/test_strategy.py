"""canary/strategy.py 灰度策略测试（SPEC §5.3-5.5，覆盖率目标 ≥90%）。

测试矩阵：
    - 阶段状态机：next_stage / stage_from_index
    - 流量配比：get_v2_ratio / get_traffic_split
    - should_advance 推进条件（满足/不满足/部分满足）
    - 性能劣化检测（5 个阈值）
    - blocking reason 具体性（SPEC §5.5 要求）
"""

from __future__ import annotations

import pytest

from vllmonline.canary.strategy import (
    AdvanceDecision,
    CanaryStage,
    CanaryStatus,
    EvalSnapshot,
    GrayStrategy,
    VersionMetrics,
    _check_performance_thresholds,
)
from vllmonline.config import CanaryConfig

# ─────────────────────────────────────────────────────────────────────────────
# 阶段状态机
# ─────────────────────────────────────────────────────────────────────────────


class TestStageMachine:
    def test_stage_order(self) -> None:
        s = GrayStrategy()
        assert s.next_stage(CanaryStage.INIT) is CanaryStage.STAGE_10
        assert s.next_stage(CanaryStage.STAGE_10) is CanaryStage.STAGE_30
        assert s.next_stage(CanaryStage.STAGE_30) is CanaryStage.STAGE_100
        assert s.next_stage(CanaryStage.STAGE_100) is CanaryStage.COMPLETED
        assert s.next_stage(CanaryStage.COMPLETED) is None

    def test_next_stage_for_rolled_back(self) -> None:
        s = GrayStrategy()
        # ROLLED_BACK 不在 STAGE_ORDER 里 → None
        assert s.next_stage(CanaryStage.ROLLED_BACK) is None

    def test_stage_from_index(self) -> None:
        s = GrayStrategy()
        stages = [0.1, 0.3, 1.0]
        assert s.stage_from_index(stages, 0) is CanaryStage.STAGE_10
        assert s.stage_from_index(stages, 1) is CanaryStage.STAGE_30
        assert s.stage_from_index(stages, 2) is CanaryStage.STAGE_100
        assert s.stage_from_index(stages, 3) is CanaryStage.COMPLETED

    def test_v2_ratio_per_stage(self) -> None:
        s = GrayStrategy()
        assert s.get_v2_ratio(CanaryStage.INIT) == 0.0
        assert s.get_v2_ratio(CanaryStage.STAGE_10) == 0.10
        assert s.get_v2_ratio(CanaryStage.STAGE_30) == 0.30
        assert s.get_v2_ratio(CanaryStage.STAGE_100) == 1.0
        assert s.get_v2_ratio(CanaryStage.ROLLED_BACK) == 0.0

    def test_get_traffic_split(self) -> None:
        s = GrayStrategy()
        split = s.get_traffic_split(CanaryStage.STAGE_30, "v1", "v2")
        assert split["v1"] == pytest.approx(0.7)
        assert split["v2"] == pytest.approx(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# should_advance —— 完整推进条件
# ─────────────────────────────────────────────────────────────────────────────


def _good_eval(sample_count: int = 100) -> EvalSnapshot:
    """构造一个"v2 明显更好"的评测快照。"""
    return EvalSnapshot(
        sample_count=sample_count,
        score_v1_mean=0.80,
        score_v2_mean=0.85,
        p_value=0.01,
        significant=True,
        effect_size=0.4,
        recommendation="advance",
    )


def _good_metrics() -> VersionMetrics:
    """v1/v2 性能基本一致的快照（无劣化）。"""
    return VersionMetrics(
        ttft_p50=0.15,
        ttft_p99=0.45,
        tpot_mean=0.03,
        throughput=100.0,
        error_rate=0.001,
    )


class TestShouldAdvance:
    async def test_all_conditions_met_advances(self) -> None:
        """样本足 + 显著 + 性能未劣化 → 推进。"""
        s = GrayStrategy()
        decision = s.should_advance(
            CanaryStage.STAGE_10,
            metrics_v1=_good_metrics(),
            metrics_v2=_good_metrics(),
            eval_result=_good_eval(),
        )
        assert decision.can_advance is True
        assert "条件满足" in decision.reason

    async def test_no_eval_blocks(self) -> None:
        """无评测数据 → 阻塞。"""
        s = GrayStrategy()
        decision = s.should_advance(
            CanaryStage.STAGE_10,
            metrics_v1=_good_metrics(),
            metrics_v2=_good_metrics(),
            eval_result=None,
        )
        assert decision.can_advance is False
        assert "无 A/B 评测数据" in decision.reason

    async def test_insufficient_samples_blocks(self) -> None:
        """样本量不足 → 阻塞。"""
        s = GrayStrategy(CanaryConfig(min_sample_size=100))
        decision = s.should_advance(
            CanaryStage.STAGE_10,
            metrics_v1=_good_metrics(),
            metrics_v2=_good_metrics(),
            eval_result=_good_eval(sample_count=50),
        )
        assert decision.can_advance is False
        assert "样本量不足" in decision.reason
        assert "50" in decision.reason
        assert "100" in decision.reason

    async def test_not_significant_blocks(self) -> None:
        """p >= 0.05 → 阻塞。"""
        s = GrayStrategy()
        eval_result = EvalSnapshot(
            sample_count=200,
            score_v1_mean=0.80,
            score_v2_mean=0.81,
            p_value=0.20,  # 不显著
            significant=False,
            effect_size=0.05,
            recommendation="hold",
        )
        decision = s.should_advance(
            CanaryStage.STAGE_10,
            metrics_v1=_good_metrics(),
            metrics_v2=_good_metrics(),
            eval_result=eval_result,
        )
        assert decision.can_advance is False
        assert "统计不显著" in decision.reason

    async def test_v2_worse_quality_blocks(self) -> None:
        """v2 质量低于 v1 → 阻塞。"""
        s = GrayStrategy()
        eval_result = EvalSnapshot(
            sample_count=200,
            score_v1_mean=0.85,
            score_v2_mean=0.75,  # 更差
            p_value=0.01,
            significant=True,
            effect_size=0.5,
            recommendation="rollback",
        )
        decision = s.should_advance(
            CanaryStage.STAGE_10,
            metrics_v1=_good_metrics(),
            metrics_v2=_good_metrics(),
            eval_result=eval_result,
        )
        assert decision.can_advance is False
        assert "v2 质量低于 v1" in decision.reason

    async def test_terminal_state_blocks(self) -> None:
        """终态不再推进。"""
        s = GrayStrategy()
        for stage in [CanaryStage.COMPLETED, CanaryStage.ROLLED_BACK]:
            decision = s.should_advance(
                stage,
                metrics_v1=_good_metrics(),
                metrics_v2=_good_metrics(),
                eval_result=_good_eval(),
            )
            assert decision.can_advance is False
            assert "终态" in decision.reason


# ─────────────────────────────────────────────────────────────────────────────
# 性能劣化检测（SPEC §5.5 阈值表）
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformanceThresholds:
    def test_no_degradation_passes(self) -> None:
        v1 = _good_metrics()
        v2 = _good_metrics()
        config = CanaryConfig()
        assert _check_performance_thresholds(v1, v2, config) == []

    def test_ttft_p50_degradation_blocked(self) -> None:
        """SPEC §5.5：ttft_p50 不超过 v1 +20%。"""
        v1 = VersionMetrics(ttft_p50=0.10)
        v2 = VersionMetrics(ttft_p50=0.15)  # +50%，超阈值
        blocks = _check_performance_thresholds(v1, v2, CanaryConfig())
        assert any("TTFT P50" in b for b in blocks)
        assert any("50.0%" in b for b in blocks)

    def test_ttft_p99_degradation_blocked(self) -> None:
        """SPEC §5.5：ttft_p99 不超过 v1 +30%。"""
        v1 = VersionMetrics(ttft_p99=0.40)
        v2 = VersionMetrics(ttft_p99=0.60)  # +50%
        blocks = _check_performance_thresholds(v1, v2, CanaryConfig())
        assert any("TTFT P99" in b for b in blocks)

    def test_tpot_degradation_blocked(self) -> None:
        """SPEC §5.5：tpot_mean 不超过 v1 +15%。"""
        v1 = VersionMetrics(tpot_mean=0.020)
        v2 = VersionMetrics(tpot_mean=0.030)  # +50%
        blocks = _check_performance_thresholds(v1, v2, CanaryConfig())
        assert any("TPOT" in b for b in blocks)

    def test_throughput_drop_blocked(self) -> None:
        """SPEC §5.5：throughput 不低于 v1 -10%。"""
        v1 = VersionMetrics(throughput=100.0)
        v2 = VersionMetrics(throughput=80.0)  # -20%
        blocks = _check_performance_thresholds(v1, v2, CanaryConfig())
        assert any("吞吐" in b for b in blocks)

    def test_error_rate_increase_blocked(self) -> None:
        """SPEC §5.5：error_rate 不高于 v1 +2%（绝对值）。"""
        v1 = VersionMetrics(error_rate=0.005)
        v2 = VersionMetrics(error_rate=0.040)  # +3.5%
        blocks = _check_performance_thresholds(v1, v2, CanaryConfig())
        assert any("错误率" in b for b in blocks)

    def test_just_at_threshold_passes(self) -> None:
        """刚好在阈值内 → 通过（边界）。"""
        v1 = VersionMetrics(ttft_p50=0.10, ttft_p99=0.40, error_rate=0.01)
        v2 = VersionMetrics(
            ttft_p50=0.119,  # +19% < 20%
            ttft_p99=0.519,  # +29.75% < 30%
            error_rate=0.029,  # +1.9% < 2%
        )
        blocks = _check_performance_thresholds(v1, v2, CanaryConfig())
        assert blocks == []

    def test_missing_metrics_skipped(self) -> None:
        """None 的指标跳过（不阻塞）。"""
        v1 = VersionMetrics()  # 全 None
        v2 = VersionMetrics()
        assert _check_performance_thresholds(v1, v2, CanaryConfig()) == []

    def test_partial_metrics(self) -> None:
        """部分指标有数据，部分无——只检查有的。"""
        v1 = VersionMetrics(ttft_p50=0.1)
        v2 = VersionMetrics(ttft_p50=0.2)  # +100% 劣化
        blocks = _check_performance_thresholds(v1, v2, CanaryConfig())
        assert len(blocks) == 1
        assert "TTFT P50" in blocks[0]


# ─────────────────────────────────────────────────────────────────────────────
# reason 具体性（SPEC §5.5 要求）
# ─────────────────────────────────────────────────────────────────────────────


class TestReasonSpecificity:
    async def test_blocking_reason_includes_numbers(self) -> None:
        """SPEC §5.5：blocking_reason 必须具体，含数值。

        例："P99 latency degraded 45% (threshold 30%)" 而非 "性能不达标"
        """
        v1 = VersionMetrics(ttft_p99=0.40)
        v2 = VersionMetrics(ttft_p99=0.58)  # +45%
        s = GrayStrategy()
        decision = s.should_advance(
            CanaryStage.STAGE_10,
            metrics_v1=v1,
            metrics_v2=v2,
            eval_result=_good_eval(),
        )
        assert decision.can_advance is False
        # reason 里应含 "45%" 和 "30%"（阈值）
        assert "45.0%" in decision.reason or "45%" in decision.reason
        assert "30%" in decision.reason

    async def test_multiple_blocking_factors_listed(self) -> None:
        """多个劣化同时存在，全部列出。"""
        v1 = VersionMetrics(
            ttft_p50=0.1, ttft_p99=0.4, tpot_mean=0.02, throughput=100.0, error_rate=0.01
        )
        v2 = VersionMetrics(
            ttft_p50=0.5,  # +400%
            ttft_p99=0.8,  # +100%
            tpot_mean=0.05,  # +150%
            throughput=50.0,  # -50%
            error_rate=0.10,  # +9%
        )
        s = GrayStrategy()
        decision = s.should_advance(
            CanaryStage.STAGE_10,
            metrics_v1=v1,
            metrics_v2=v2,
            eval_result=_good_eval(),
        )
        assert decision.can_advance is False
        assert len(decision.blocking_factors) >= 5
        reason = decision.reason
        assert "TTFT P50" in reason
        assert "TTFT P99" in reason
        assert "TPOT" in reason
        assert "吞吐" in reason
        assert "错误率" in reason


# ─────────────────────────────────────────────────────────────────────────────
# AdvanceDecision dataclass
# ─────────────────────────────────────────────────────────────────────────────


class TestAdvanceDecision:
    def test_default_blocking_empty(self) -> None:
        d = AdvanceDecision(can_advance=True, reason="ok")
        assert d.blocking_factors == []

    def test_frozen(self) -> None:
        d = AdvanceDecision(can_advance=True, reason="ok")
        with pytest.raises(AttributeError):
            d.can_advance = False  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# CanaryStage / CanaryStatus 枚举
# ─────────────────────────────────────────────────────────────────────────────


class TestEnums:
    def test_canary_stage_values(self) -> None:
        assert CanaryStage.INIT.value == "INIT"
        assert CanaryStage.STAGE_10.value == "STAGE_10%"
        assert CanaryStage.COMPLETED.value == "COMPLETED"

    def test_canary_status_values(self) -> None:
        assert CanaryStatus.IN_PROGRESS.value == "IN_PROGRESS"
        assert CanaryStatus.ROLLED_BACK.value == "ROLLED_BACK"
