"""eval/statistics.py 测试（SPEC §6.3-6.4，覆盖率目标 ≥95%）。

测试矩阵：
    - welch_ttest 用已知数据集验证（与 scipy.stats.ttest_ind 交叉验证）
    - min_sample_size：d=0.5→~64, d=0.2→~394（SPEC 示例）
    - cohen_d 正确性
    - 边界：相同值、空列表、极端值、方差为 0
    - recommendation 判定（advance/rollback/hold）
"""

from __future__ import annotations

import math

import pytest
from scipy import stats

from vllmonline.eval.statistics import (
    Z_ALPHA_HALF_005,
    Z_BETA_080,
    TTestResult,
    cohen_d,
    min_sample_size,
    min_sample_size_hardcoded,
    welch_ttest,
)

# ─────────────────────────────────────────────────────────────────────────────
# welch_ttest 正确性（与 scipy 交叉验证）
# ─────────────────────────────────────────────────────────────────────────────


class TestWelchTtest:
    def test_matches_scipy_directly(self) -> None:
        """结果应与 scipy.stats.ttest_ind(equal_var=False) 完全一致。"""
        v1 = [0.8, 0.82, 0.79, 0.81, 0.83, 0.78, 0.80, 0.82, 0.81, 0.79]
        v2 = [0.85, 0.87, 0.84, 0.86, 0.88, 0.83, 0.85, 0.87, 0.86, 0.84]

        result = welch_ttest(v1, v2)
        scipy_result = stats.ttest_ind(v1, v2, equal_var=False)

        assert result.t_statistic == pytest.approx(float(scipy_result.statistic), abs=1e-10)
        assert result.p_value == pytest.approx(float(scipy_result.pvalue), abs=1e-10)

    def test_mean_and_std_correct(self) -> None:
        v1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        v2 = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = welch_ttest(v1, v2)
        assert result.mean_v1 == pytest.approx(3.0)
        assert result.mean_v2 == pytest.approx(4.0)
        assert result.std_v1 == pytest.approx(math.sqrt(2.5))  # 样本方差
        assert result.n_v1 == 5
        assert result.n_v2 == 5

    def test_v2_better_when_v2_higher(self) -> None:
        v1 = [0.7, 0.71, 0.69, 0.70, 0.72]
        v2 = [0.85, 0.86, 0.84, 0.85, 0.87]
        result = welch_ttest(v1, v2)
        assert result.v2_better is True
        assert result.mean_v2 > result.mean_v1

    def test_significant_when_large_difference(self) -> None:
        """差异巨大时 p < 0.05。"""
        v1 = [0.5] * 20
        v2 = [0.9] * 20
        # 注意：方差为 0 会抛错，加微小扰动
        v1 = [0.50 + i * 0.001 for i in range(20)]
        v2 = [0.90 + i * 0.001 for i in range(20)]
        result = welch_ttest(v1, v2)
        assert result.significant is True
        assert result.p_value < 0.05

    def test_not_significant_when_small_difference(self) -> None:
        """差异小且样本少时 p >= 0.05。"""
        v1 = [0.80, 0.82, 0.79]
        v2 = [0.81, 0.80, 0.82]
        result = welch_ttest(v1, v2)
        assert not result.significant
        assert result.p_value >= 0.05


# ─────────────────────────────────────────────────────────────────────────────
# recommendation（SPEC §6.3 判定）
# ─────────────────────────────────────────────────────────────────────────────


class TestRecommendation:
    def test_advance_when_v2_significantly_better(self) -> None:
        v1 = [0.70 + i * 0.01 for i in range(30)]
        v2 = [0.85 + i * 0.01 for i in range(30)]
        result = welch_ttest(v1, v2)
        assert result.recommendation == "advance"

    def test_rollback_when_v2_significantly_worse(self) -> None:
        v1 = [0.85 + i * 0.01 for i in range(30)]
        v2 = [0.70 + i * 0.01 for i in range(30)]
        result = welch_ttest(v1, v2)
        assert result.recommendation == "rollback"

    def test_hold_when_not_significant(self) -> None:
        v1 = [0.80, 0.81, 0.79, 0.82, 0.78]
        v2 = [0.81, 0.80, 0.82, 0.79, 0.81]
        result = welch_ttest(v1, v2)
        assert result.recommendation == "hold"


# ─────────────────────────────────────────────────────────────────────────────
# Cohen's d
# ─────────────────────────────────────────────────────────────────────────────


class TestCohenD:
    def test_zero_when_identical(self) -> None:
        v1 = [0.8, 0.82, 0.79, 0.81]
        result = cohen_d(v1, v1)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_positive_when_v2_higher(self) -> None:
        v1 = [0.7, 0.8, 0.75, 0.78]
        v2 = [0.85, 0.9, 0.87, 0.88]
        d = cohen_d(v1, v2)
        assert d > 0

    def test_large_effect_for_big_difference(self) -> None:
        """差异巨大 → 大效应量（d > 0.8 是大效应）。"""
        v1 = [0.5, 0.51, 0.49, 0.50, 0.51]
        v2 = [0.9, 0.91, 0.89, 0.90, 0.91]
        d = cohen_d(v1, v2)
        assert d > 0.8

    def test_matches_manual_calculation(self) -> None:
        """手算验证：v1=[1,2,3], v2=[4,5,6]。"""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        # mean1=2, mean2=5, std1=std2=1, s_pooled=1, d=(5-2)/1=3
        d = cohen_d(v1, v2)
        assert d == pytest.approx(3.0)

    def test_welch_ttest_includes_cohen_d(self) -> None:
        v1 = [0.7, 0.8, 0.75]
        v2 = [0.85, 0.9, 0.87]
        result = welch_ttest(v1, v2)
        assert result.cohen_d == pytest.approx(cohen_d(v1, v2))


# ─────────────────────────────────────────────────────────────────────────────
# min_sample_size（SPEC §6.4）
# ─────────────────────────────────────────────────────────────────────────────


class TestMinSampleSize:
    def test_medium_effect(self) -> None:
        """SPEC §6.4：d=0.5 → 约 64（精确 63）。"""
        n = min_sample_size(0.5)
        # SPEC 文档示例是 64（用 Z=1.96/0.84 近似），scipy 精确值是 63
        assert 60 <= n <= 65

    def test_small_effect(self) -> None:
        """SPEC §6.4：d=0.2 → 约 394（精确 393）。"""
        n = min_sample_size(0.2)
        assert 390 <= n <= 395

    def test_default_power_alpha(self) -> None:
        """默认 power=0.80, alpha=0.05。"""
        # d=0.3 → SPEC §6.4 隐含的默认目标效应量
        n = min_sample_size(0.3)
        assert 170 <= n <= 180

    def test_higher_power_needs_more_samples(self) -> None:
        """power 越高，需要的样本越多。"""
        n_low = min_sample_size(0.5, power=0.80)
        n_high = min_sample_size(0.5, power=0.95)
        assert n_high > n_low

    def test_smaller_effect_needs_more_samples(self) -> None:
        """效应量越小，需要的样本越多（反比平方关系）。"""
        assert min_sample_size(0.2) > min_sample_size(0.5)
        assert min_sample_size(0.5) > min_sample_size(0.8)

    def test_invalid_effect_size_raises(self) -> None:
        with pytest.raises(ValueError, match="effect_size"):
            min_sample_size(0)
        with pytest.raises(ValueError, match="effect_size"):
            min_sample_size(-0.1)

    def test_invalid_power_raises(self) -> None:
        with pytest.raises(ValueError, match="power"):
            min_sample_size(0.5, power=0)
        with pytest.raises(ValueError, match="power"):
            min_sample_size(0.5, power=1.5)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            min_sample_size(0.5, alpha=0)
        with pytest.raises(ValueError, match="alpha"):
            min_sample_size(0.5, alpha=1.5)

    def test_hardcoded_matches_scipy_within_tolerance(self) -> None:
        """硬编码 Z 值版本与 scipy 精确版本接近（±2 内）。"""
        for d in [0.2, 0.3, 0.5, 0.8]:
            n_scipy = min_sample_size(d)
            n_hard = min_sample_size_hardcoded(d)
            assert abs(n_scipy - n_hard) <= 2

    def test_hardcoded_uses_spec_constants(self) -> None:
        """验证硬编码常量与 SPEC §6.4 一致。"""
        assert Z_ALPHA_HALF_005 == 1.96
        assert Z_BETA_080 == 0.84


# ─────────────────────────────────────────────────────────────────────────────
# 边界条件
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_insufficient_v1_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="v1 样本量不足"):
            welch_ttest([0.5], [0.6, 0.7])

    def test_insufficient_v2_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="v2 样本量不足"):
            welch_ttest([0.5, 0.6], [0.7])

    def test_empty_lists_raise(self) -> None:
        with pytest.raises(ValueError):
            welch_ttest([], [0.5, 0.6])
        with pytest.raises(ValueError):
            welch_ttest([0.5, 0.6], [])

    def test_identical_constant_groups(self) -> None:
        """两组都是相同常数（方差 0，均值同）→ p=1.0, d=0。"""
        result = welch_ttest([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert result.p_value == pytest.approx(1.0)
        assert result.cohen_d == pytest.approx(0.0)
        assert result.recommendation == "hold"

    def test_different_constant_groups_raises(self) -> None:
        """两组都是常数但均值不同 → 抛错（方差 0 无法算 t-test）。"""
        with pytest.raises(ValueError, match="方差均为 0"):
            welch_ttest([0.5, 0.5, 0.5], [0.9, 0.9, 0.9])

    def test_extreme_values(self) -> None:
        """极端值（0 和 1）也能算。"""
        v1 = [0.0, 0.01, 0.02, 0.01]
        v2 = [1.0, 0.99, 0.98, 0.99]
        result = welch_ttest(v1, v2)
        assert result.significant
        assert result.cohen_d > 0

    def test_single_sample_in_each(self) -> None:
        """每组只有 2 个样本（最小合法量）。"""
        result = welch_ttest([0.7, 0.8], [0.85, 0.9])
        # 不抛，有结果
        assert result.n_v1 == 2
        assert result.n_v2 == 2

    def test_one_group_constant(self) -> None:
        """一组方差为 0（常数），另一组有方差 → 走 scipy 路径不抛。"""
        v1 = [0.5, 0.5, 0.5, 0.5]
        v2 = [0.6, 0.7, 0.8, 0.9]
        result = welch_ttest(v1, v2)
        assert math.isfinite(result.t_statistic)
        assert result.std_v1 == 0.0

    def test_satterthwaite_df_fallback(self) -> None:
        """denominator==0 的兜底分支。"""
        from vllmonline.eval.statistics import _satterthwaite_df

        df = _satterthwaite_df(0.0, 0.0, 5, 5)
        assert df == 8.0

    def test_cohen_d_zero_pooled(self) -> None:
        """s_pooled=0 时返回 0（兜底）。"""
        from vllmonline.eval.statistics import _cohen_d_from_stats

        assert _cohen_d_from_stats(0.5, 0.5, 0.0, 0.0) == 0.0

    def test_hardcoded_invalid_effect_size_raises(self) -> None:
        with pytest.raises(ValueError, match="effect_size"):
            min_sample_size_hardcoded(0)
        with pytest.raises(ValueError, match="effect_size"):
            min_sample_size_hardcoded(-1)


# ─────────────────────────────────────────────────────────────────────────────
# TTestResult dataclass
# ─────────────────────────────────────────────────────────────────────────────


class TestTTestResult:
    def test_frozen(self) -> None:
        r = TTestResult(
            t_statistic=1.0,
            p_value=0.5,
            degrees_of_freedom=10,
            mean_v1=0.5,
            mean_v2=0.6,
            std_v1=0.1,
            std_v2=0.1,
            n_v1=10,
            n_v2=10,
            cohen_d=0.5,
        )
        with pytest.raises(AttributeError):
            r.p_value = 0.01  # type: ignore[misc]

    def test_significant_property(self) -> None:
        r_significant = TTestResult(
            t_statistic=3.0,
            p_value=0.01,
            degrees_of_freedom=10,
            mean_v1=0.5,
            mean_v2=0.7,
            std_v1=0.1,
            std_v2=0.1,
            n_v1=10,
            n_v2=10,
            cohen_d=1.0,
        )
        assert r_significant.significant is True

        r_not = TTestResult(
            t_statistic=0.5,
            p_value=0.6,
            degrees_of_freedom=10,
            mean_v1=0.5,
            mean_v2=0.51,
            std_v1=0.1,
            std_v2=0.1,
            n_v1=10,
            n_v2=10,
            cohen_d=0.1,
        )
        assert r_not.significant is False
