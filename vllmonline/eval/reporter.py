"""评测报告生成（SPEC §6，Markdown 格式）。

报告含：
    - 样本量
    - 各维度均分（v1 vs v2）
    - p 值、效应量
    - 建议（advance/hold/rollback）
"""

from __future__ import annotations

from typing import Any

from vllmonline.eval.judge import AggregatedEval, ComparisonResult
from vllmonline.eval.statistics import TTestResult


def generate_report(
    eval_result: AggregatedEval | Any,
    *,
    title: str = "vLLMonline A/B 评测报告",
) -> str:
    """生成 Markdown 格式的评测报告。

    Args:
        eval_result: AggregatedEval 或兼容对象（含必要字段）
        title: 报告标题

    Returns:
        Markdown 字符串
    """
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")

    # 概要
    lines.append("## 概要")
    lines.append("")
    lines.append(f"- 样本量：{eval_result.sample_count}")
    lines.append(f"- v1 平均分：{eval_result.score_v1_mean:.4f}")
    lines.append(f"- v2 平均分：{eval_result.score_v2_mean:.4f}")

    t_test: TTestResult = eval_result.t_test
    lines.append(f"- t 统计量：{t_test.t_statistic:.4f}")
    lines.append(f"- p 值：{t_test.p_value:.4f}")
    lines.append(f"- Cohen's d（效应量）：{t_test.cohen_d:.4f}")
    lines.append(f"- 自由度：{t_test.degrees_of_freedom:.2f}")
    lines.append(f"- 统计显著：{'是' if t_test.significant else '否'}")
    lines.append(f"- 建议：**{t_test.recommendation}**")
    lines.append("")

    # 维度详情
    dim_v1 = getattr(eval_result, "dimension_scores_v1", {})
    dim_v2 = getattr(eval_result, "dimension_scores_v2", {})
    if dim_v1 or dim_v2:
        lines.append("## 各维度详情")
        lines.append("")
        lines.append("| 维度 | v1 | v2 | 差异 |")
        lines.append("|------|------|------|------|")
        for dim in ["accuracy", "completeness", "safety"]:
            v1 = dim_v1.get(dim, 0.0)
            v2 = dim_v2.get(dim, 0.0)
            diff = v2 - v1
            sign = "+" if diff >= 0 else ""
            lines.append(f"| {dim} | {v1:.4f} | {v2:.4f} | {sign}{diff:.4f} |")
        lines.append("")

    # 判定说明
    lines.append("## 判定说明")
    lines.append("")
    rec = t_test.recommendation
    if rec == "advance":
        lines.append("v2 在统计上显著优于 v1，建议**推进灰度**到下一阶段。")
    elif rec == "rollback":
        lines.append("v2 在统计上显著劣于 v1，建议**立即回滚**。")
    else:
        lines.append(
            "证据不足（p >= 0.05），无法判定 v2 显著优于或劣于 v1。"
            "建议**延长当前阶段**继续收集样本。"
        )
    lines.append("")

    # 统计细节
    lines.append("## 统计细节")
    lines.append("")
    lines.append("- 检验方法：Welch's t-test（不假设方差齐性）")
    lines.append("- 自由度：Satterthwaite 近似")
    lines.append("- 显著性水平：α = 0.05")
    lines.append("- 假设：双侧检验")
    lines.append("")

    return "\n".join(lines)


def generate_simple_report(
    v1_scores: list[float],
    v2_scores: list[float],
    *,
    title: str = "vLLMonline A/B 评测报告",
) -> str:
    """便捷函数：直接从分数列表生成报告。"""
    from vllmonline.eval.statistics import welch_ttest

    t_test = welch_ttest(v1_scores, v2_scores)

    # 包装成 AggregatedEval-like 对象
    class _Simple:
        sample_count = len(v1_scores)
        score_v1_mean = sum(v1_scores) / len(v1_scores)
        score_v2_mean = sum(v2_scores) / len(v2_scores)

    _simple = _Simple()
    _simple.t_test = t_test  # type: ignore[attr-defined]
    return generate_report(_simple, title=title)


__all__ = [
    "AggregatedEval",
    "ComparisonResult",
    "generate_report",
    "generate_simple_report",
]
