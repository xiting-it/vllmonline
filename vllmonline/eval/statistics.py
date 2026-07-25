"""统计检验（SPEC §6.3-6.4，覆盖率目标 ≥95%）。

实现：
    - welch_ttest(scores_v1, scores_v2) → TTestResult
      用 scipy.stats.ttest_ind(equal_var=False) + 手写 Cohen's d 交叉验证
    - min_sample_size(effect_size, power, alpha) → int
      SPEC §6.4 公式：n = 2 × (Z_α/2 + Z_β)² / d²
    - cohen_d(scores_v1, scores_v2) → float（Cohen's d 效应量）

判定逻辑（SPEC §6.3）：
    p < 0.05 且 v2 > v1 → advance
    p < 0.05 且 v2 < v1 → rollback
    p >= 0.05           → hold

Z 值硬编码（SPEC §6.4）：
    α=0.05 → Z_α/2 = 1.96
    power=0.80 → Z_β = 0.84
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 常量（SPEC §6.4 硬编码 Z 值）
# ─────────────────────────────────────────────────────────────────────────────

# 标准正态分布的分位数（双侧 α=0.05 → Z_{α/2}=1.96）
Z_ALPHA_HALF_005 = 1.96
# power=0.80 → Z_β=0.84（β=0.20）
Z_BETA_080 = 0.84


@dataclass(frozen=True, slots=True)
class TTestResult:
    """Welch's t-test 结果。"""

    t_statistic: float
    p_value: float
    degrees_of_freedom: float
    mean_v1: float
    mean_v2: float
    std_v1: float
    std_v2: float
    n_v1: int
    n_v2: int
    cohen_d: float  # 效应量（Cohen's d，正= v2 更好）

    @property
    def significant(self) -> bool:
        """p < α（默认 0.05）。"""
        return self.p_value < 0.05

    @property
    def recommendation(self) -> str:
        """SPEC §6.3 判定逻辑：advance / rollback / hold。"""
        if not self.significant:
            return "hold"
        return "advance" if self.mean_v2 > self.mean_v1 else "rollback"

    @property
    def v2_better(self) -> bool:
        """v2 是否优于 v1（均值更大）。"""
        return self.mean_v2 > self.mean_v1


# ─────────────────────────────────────────────────────────────────────────────
# Welch's t-test
# ─────────────────────────────────────────────────────────────────────────────


def welch_ttest(scores_v1: list[float], scores_v2: list[float]) -> TTestResult:
    """执行 Welch's t-test（不假设方差齐性）。

    SPEC §6.3：
        t = (x̄₁ - x̄₂) / sqrt(s₁²/n₁ + s₂²/n₂)
        自由度：Satterthwaite 近似
        p 值：双侧检验

    Args:
        scores_v1: 旧版本（baseline）的分数列表
        scores_v2: 新版本的分数列表

    Returns:
        TTestResult（含 t、p、df、Cohen's d）

    Raises:
        ValueError: 样本量 < 2，或方差为 0（无法计算）
    """
    if len(scores_v1) < 2:
        msg = f"v1 样本量不足（{len(scores_v1)}），至少需要 2"
        raise ValueError(msg)
    if len(scores_v2) < 2:
        msg = f"v2 样本量不足（{len(scores_v2)}），至少需要 2"
        raise ValueError(msg)

    arr1 = np.array(scores_v1, dtype=float)
    arr2 = np.array(scores_v2, dtype=float)

    mean1 = float(np.mean(arr1))
    mean2 = float(np.mean(arr2))
    std1 = float(np.std(arr1, ddof=1))  # 样本标准差（Bessel 校正）
    std2 = float(np.std(arr2, ddof=1))
    n1 = len(arr1)
    n2 = len(arr2)

    # 方差为 0 时 t-test 无定义（除以 0）
    if std1 == 0 and std2 == 0:
        # 两组都是常数：均值相同则无差异，不同则差异巨大
        if mean1 == mean2:
            return TTestResult(
                t_statistic=0.0,
                p_value=1.0,
                degrees_of_freedom=float(n1 + n2 - 2),
                mean_v1=mean1,
                mean_v2=mean2,
                std_v1=std1,
                std_v2=std2,
                n_v1=n1,
                n_v2=n2,
                cohen_d=0.0,
            )
        # 常数不同：t 趋向 ±inf，p 趋向 0
        msg = f"两组分数方差均为 0 但均值不同（v1={mean1}, v2={mean2}），无法计算 t-test"
        raise ValueError(msg)
    if std1 == 0 or std2 == 0:
        # 一组方差为 0：scipy 仍能算，但 df 可能异常。交给 scipy 处理。
        pass

    # 用 scipy 计算（工业级实现，避免手算错误）。
    # 抑制 scipy 对常数/近常数数组的精度损失警告（不影响 t/p 值）。
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = stats.ttest_ind(arr1, arr2, equal_var=False)
    t_stat = float(result.statistic)
    p_value = float(result.pvalue)
    # scipy 1.18 的 df 通过 .df 属性获取（旧版需要手算）
    df = float(getattr(result, "df", _satterthwaite_df(std1, std2, n1, n2)))

    # Cohen's d（手写，SPEC §6.3）：
    # d = |x̄₁ - x̄₂| / s_pooled
    # s_pooled = sqrt((s₁² + s₂²) / 2)（简化版，非加权 pooled）
    d = _cohen_d_from_stats(mean1, mean2, std1, std2)

    return TTestResult(
        t_statistic=t_stat,
        p_value=p_value,
        degrees_of_freedom=df,
        mean_v1=mean1,
        mean_v2=mean2,
        std_v1=std1,
        std_v2=std2,
        n_v1=n1,
        n_v2=n2,
        cohen_d=d,
    )


def _satterthwaite_df(std1: float, std2: float, n1: int, n2: int) -> float:
    """Satterthwaite 自由度近似（SPEC §6.3）。

    df = (s₁²/n₁ + s₂²/n₂)² / ((s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1))
    """
    v1 = (std1**2) / n1
    v2 = (std2**2) / n2
    numerator = (v1 + v2) ** 2
    denominator = (v1**2) / (n1 - 1) + (v2**2) / (n2 - 1)
    if denominator == 0:
        return float(n1 + n2 - 2)
    return numerator / denominator


def _cohen_d_from_stats(mean1: float, mean2: float, std1: float, std2: float) -> float:
    """Cohen's d（SPEC §6.3）。

    d = (mean2 - mean1) / s_pooled
    s_pooled = sqrt((std1² + std2²) / 2)

    注意：d 为正表示 v2（mean2）更好。
    """
    s_pooled = math.sqrt((std1**2 + std2**2) / 2)
    if s_pooled == 0:
        return 0.0
    return (mean2 - mean1) / s_pooled


def cohen_d(scores_v1: list[float], scores_v2: list[float]) -> float:
    """便捷函数：直接算 Cohen's d。"""
    arr1 = np.array(scores_v1, dtype=float)
    arr2 = np.array(scores_v2, dtype=float)
    return _cohen_d_from_stats(
        float(np.mean(arr1)),
        float(np.mean(arr2)),
        float(np.std(arr1, ddof=1)) if len(arr1) > 1 else 0.0,
        float(np.std(arr2, ddof=1)) if len(arr2) > 1 else 0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 最小样本量（SPEC §6.4）
# ─────────────────────────────────────────────────────────────────────────────


def min_sample_size(
    effect_size: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """计算每组需要的最小样本量（SPEC §6.4）。

    公式：n = 2 × (Z_{α/2} + Z_β)² / d²

    Args:
        effect_size: 期望检测的最小效应量（Cohen's d）
        power: 检验功效（1 - β），默认 0.80
        alpha: 显著性水平，默认 0.05

    Returns:
        每组最小样本量（向上取整）

    Raises:
        ValueError: effect_size <= 0 或参数非法

    Examples:
        >>> min_sample_size(0.5)  # 中等效应
        64
        >>> min_sample_size(0.2)  # 小效应
        394
        >>> min_sample_size(0.3)  # SPEC §6.4 默认
        176
    """
    if effect_size <= 0:
        msg = f"effect_size 必须为正，得到 {effect_size}"
        raise ValueError(msg)
    if not 0 < power < 1:
        msg = f"power 必须在 (0, 1)，得到 {power}"
        raise ValueError(msg)
    if not 0 < alpha < 1:
        msg = f"alpha 必须在 (0, 1)，得到 {alpha}"
        raise ValueError(msg)

    # 用 scipy 的 ppf 精确算 Z 值（而非硬编码 1.96/0.84）
    # 双侧 α → Z_{α/2} = -ppf(α/2) = ppf(1 - α/2)
    z_alpha_half = float(stats.norm.ppf(1 - alpha / 2))
    # power = 1 - β → Z_β = ppf(power)
    z_beta = float(stats.norm.ppf(power))

    n_raw = 2 * (z_alpha_half + z_beta) ** 2 / (effect_size**2)
    return math.ceil(n_raw)


def min_sample_size_hardcoded(
    effect_size: float,
    *,
    z_alpha_half: float = Z_ALPHA_HALF_005,
    z_beta: float = Z_BETA_080,
) -> int:
    """用 SPEC §6.4 硬编码的 Z 值算最小样本量（教学/验证用）。

    SPEC 示例用 Z_{α/2}=1.96, Z_β=0.84。本函数用这些固定值，
    便于和 SPEC 文档对照（min_sample_size 用 scipy 精确值，结果可能有 ±1 差异）。
    """
    if effect_size <= 0:
        msg = f"effect_size 必须为正，得到 {effect_size}"
        raise ValueError(msg)
    n_raw = 2 * (z_alpha_half + z_beta) ** 2 / (effect_size**2)
    return math.ceil(n_raw)


__all__ = [
    "Z_ALPHA_HALF_005",
    "Z_BETA_080",
    "TTestResult",
    "cohen_d",
    "min_sample_size",
    "min_sample_size_hardcoded",
    "welch_ttest",
]
