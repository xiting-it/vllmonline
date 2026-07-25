"""灰度策略状态机 + 推进判定（SPEC §5.3-5.5）。

GrayStrategy 维护灰度发布的阶段状态机：
    INIT → STAGE_10% → STAGE_30% → STAGE_100% → COMPLETED
                 ↓          ↓          ↓
              ROLLBACK   ROLLBACK   ROLLBACK

should_advance() 按 SPEC §5.5 的条件判定是否推进：
    1. 最小样本量已达成
    2. Welch's t-test p < 0.05 且 v2 ≥ v1
    3. 所有性能指标未劣化（5 个阈值）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

from vllmonline.config import CanaryConfig

# ─────────────────────────────────────────────────────────────────────────────
# 灰度阶段枚举（SPEC §5.3）
# ─────────────────────────────────────────────────────────────────────────────


class CanaryStage(StrEnum):
    """灰度发布阶段。"""

    INIT = "INIT"
    STAGE_10 = "STAGE_10%"
    STAGE_30 = "STAGE_30%"
    STAGE_100 = "STAGE_100%"
    COMPLETED = "COMPLETED"
    ROLLED_BACK = "ROLLED_BACK"


class CanaryStatus(StrEnum):
    """灰度部署的整体状态。"""

    INIT = "INIT"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    ROLLED_BACK = "ROLLED_BACK"


# ─────────────────────────────────────────────────────────────────────────────
# 版本 metrics 快照（SPEC §5.4 的 per-version 对比输入）
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VersionMetrics:
    """某版本在某时刻的 metrics 快照。

    所有时间单位秒，比率单位 [0,1]。None 表示数据不足。
    """

    ttft_p50: float | None = None
    ttft_p99: float | None = None
    tpot_mean: float | None = None
    throughput: float | None = None  # tokens/s
    error_rate: float | None = None  # [0,1]


@dataclass(frozen=True, slots=True)
class EvalSnapshot:
    """A/B 评测快照（P4 提供，灰度判定输入）。"""

    sample_count: int
    score_v1_mean: float
    score_v2_mean: float
    p_value: float | None
    significant: bool
    effect_size: float | None
    recommendation: str  # advance | hold | rollback


# ─────────────────────────────────────────────────────────────────────────────
# 推进判定
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AdvanceDecision:
    """should_advance 的结果。

    SPEC §5.5：返回 (bool, reason)，reason 必须具体。
    """

    can_advance: bool
    reason: str
    blocking_factors: list[str] = field(default_factory=list)


class GrayStrategy:
    """灰度策略机（无状态——每次 should_advance 接收当前 metrics）。

    阶段→流量配比映射（SPEC §5.3）：
        STAGE_10  → v2 占 10%
        STAGE_30  → v2 占 30%
        STAGE_100 → v2 占 100%
    """

    STAGE_TO_V2_RATIO: ClassVar[dict[CanaryStage, float]] = {
        CanaryStage.INIT: 0.0,
        CanaryStage.STAGE_10: 0.10,
        CanaryStage.STAGE_30: 0.30,
        CanaryStage.STAGE_100: 1.0,
        CanaryStage.COMPLETED: 1.0,
        CanaryStage.ROLLED_BACK: 0.0,
    }

    # 合法的阶段顺序（用于 next_stage）
    STAGE_ORDER: ClassVar[list[CanaryStage]] = [
        CanaryStage.INIT,
        CanaryStage.STAGE_10,
        CanaryStage.STAGE_30,
        CanaryStage.STAGE_100,
        CanaryStage.COMPLETED,
    ]

    def __init__(self, config: CanaryConfig | None = None) -> None:
        self._config = config or CanaryConfig()

    def get_v2_ratio(self, stage: CanaryStage) -> float:
        """某阶段对应的 v2 流量占比。"""
        return self.STAGE_TO_V2_RATIO.get(stage, 0.0)

    def get_traffic_split(self, stage: CanaryStage, v1_id: str, v2_id: str) -> dict[str, float]:
        """生成路由表用的 traffic_split dict。"""
        v2_ratio = self.get_v2_ratio(stage)
        return {v1_id: round(1.0 - v2_ratio, 4), v2_id: round(v2_ratio, 4)}

    def next_stage(self, current: CanaryStage) -> CanaryStage | None:
        """下一个阶段（None 表示已到终态）。"""
        try:
            idx = self.STAGE_ORDER.index(current)
        except ValueError:
            return None
        if idx + 1 >= len(self.STAGE_ORDER):
            return None
        return self.STAGE_ORDER[idx + 1]

    def stage_from_index(self, stages: list[float], index: int) -> CanaryStage:
        """从 stages 列表的 index 推导阶段名。

        SPEC §8.3 的 stages=[0.1, 0.3, 1.0] 对应：
            index 0 → STAGE_10%
            index 1 → STAGE_30%
            index 2 → STAGE_100%
        """
        if index >= len(stages):
            return CanaryStage.COMPLETED
        ratio = stages[index]
        if ratio >= 1.0:
            return CanaryStage.STAGE_100
        if ratio >= 0.3:
            return CanaryStage.STAGE_30
        if ratio >= 0.1:
            return CanaryStage.STAGE_10
        return CanaryStage.INIT

    # ── 核心：推进判定（SPEC §5.5）──

    def should_advance(
        self,
        stage: CanaryStage,
        metrics_v1: VersionMetrics,
        metrics_v2: VersionMetrics,
        eval_result: EvalSnapshot | None = None,
    ) -> AdvanceDecision:
        """判定是否可推进到下一阶段。

        SPEC §5.5 必须同时满足：
            1. 最小样本量（eval_result.sample_count >= min_sample_size）
            2. Welch's t-test p < 0.05 且 v2 ≥ v1（eval_result.recommendation == "advance"）
            3. 性能指标未劣化（5 个阈值）

        Args:
            stage: 当前阶段
            metrics_v1/v2: 两版本的性能 metrics 快照
            eval_result: A/B 评测结果（可 None，表示无评测数据）

        Returns:
            AdvanceDecision(can_advance, reason, blocking_factors)
        """
        # 终态不再推进
        if stage in (CanaryStage.COMPLETED, CanaryStage.ROLLED_BACK):
            return AdvanceDecision(
                can_advance=False,
                reason=f"已处于终态 {stage.value}",
                blocking_factors=["terminal_state"],
            )

        blocking: list[str] = []

        # 条件 1：最小样本量
        if eval_result is None:
            blocking.append("无 A/B 评测数据")
        elif eval_result.sample_count < self._config.min_sample_size:
            blocking.append(
                f"样本量不足：{eval_result.sample_count} < {self._config.min_sample_size}"
            )

        # 条件 2：统计显著（p < significance_alpha）+ v2 不劣于 v1
        alpha = 0.05  # SPEC §5.5 硬编码 p < 0.05
        if eval_result is not None:
            if eval_result.p_value is not None and eval_result.p_value >= alpha:
                blocking.append(f"统计不显著：p={eval_result.p_value:.4f} ≥ {alpha}")
            if eval_result.score_v2_mean < eval_result.score_v1_mean:
                blocking.append(
                    f"v2 质量低于 v1：{eval_result.score_v2_mean:.3f} < "
                    f"{eval_result.score_v1_mean:.3f}"
                )

        # 条件 3：性能指标未劣化（SPEC §5.5 阈值表）
        perf_blocks = _check_performance_thresholds(metrics_v1, metrics_v2, self._config)
        blocking.extend(perf_blocks)

        if blocking:
            reason = "；".join(blocking)
            return AdvanceDecision(can_advance=False, reason=reason, blocking_factors=blocking)

        return AdvanceDecision(
            can_advance=True,
            reason=(
                f"所有条件满足：样本量充足、统计显著、性能未劣化。"
                f"v2 ttft_p99={metrics_v2.ttft_p99}, error_rate={metrics_v2.error_rate}"
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 性能阈值检查（SPEC §5.5）
# ─────────────────────────────────────────────────────────────────────────────


def _check_performance_thresholds(
    v1: VersionMetrics,
    v2: VersionMetrics,
    config: CanaryConfig,
) -> list[str]:
    """检查 v2 相对 v1 的性能劣化（SPEC §5.5 阈值表）。

    返回劣化项列表（空表示无劣化）。
    缺数据（None）的指标跳过（不阻塞推进）。
    """
    blocks: list[str] = []

    # ttft_p50: v2 不超过 v1 +20%
    if v1.ttft_p50 is not None and v2.ttft_p50 is not None and v1.ttft_p50 > 0:
        degradation = (v2.ttft_p50 - v1.ttft_p50) / v1.ttft_p50
        if degradation > config.ttft_p50_degradation_threshold:
            blocks.append(
                f"TTFT P50 劣化 {degradation * 100:.1f}%"
                f"（阈值 +{config.ttft_p50_degradation_threshold * 100:.0f}%）"
            )

    # ttft_p99: v2 不超过 v1 +30%
    if v1.ttft_p99 is not None and v2.ttft_p99 is not None and v1.ttft_p99 > 0:
        degradation = (v2.ttft_p99 - v1.ttft_p99) / v1.ttft_p99
        if degradation > config.ttft_p99_degradation_threshold:
            blocks.append(
                f"TTFT P99 劣化 {degradation * 100:.1f}%"
                f"（阈值 +{config.ttft_p99_degradation_threshold * 100:.0f}%）"
            )

    # tpot_mean: v2 不超过 v1 +15%
    if v1.tpot_mean is not None and v2.tpot_mean is not None and v1.tpot_mean > 0:
        degradation = (v2.tpot_mean - v1.tpot_mean) / v1.tpot_mean
        if degradation > config.tpot_mean_degradation_threshold:
            blocks.append(
                f"TPOT 劣化 {degradation * 100:.1f}%"
                f"（阈值 +{config.tpot_mean_degradation_threshold * 100:.0f}%）"
            )

    # throughput: v2 不低于 v1 -10%（注意方向相反）
    if v1.throughput is not None and v2.throughput is not None and v1.throughput > 0:
        drop = (v1.throughput - v2.throughput) / v1.throughput
        if drop > config.throughput_drop_threshold:
            blocks.append(
                f"吞吐下降 {drop * 100:.1f}%（阈值 -{config.throughput_drop_threshold * 100:.0f}%）"
            )

    # error_rate: v2 不高于 v1 +2%（绝对值）
    if v1.error_rate is not None and v2.error_rate is not None:
        increase = v2.error_rate - v1.error_rate
        if increase > config.error_rate_increase_threshold:
            blocks.append(
                f"错误率上升 {increase * 100:.2f}%"
                f"（阈值 +{config.error_rate_increase_threshold * 100:.0f}%）"
            )

    return blocks
