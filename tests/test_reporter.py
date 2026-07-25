"""eval/reporter.py 测试。"""

from __future__ import annotations

from vllmonline.eval.judge import (
    ComparisonResult,
    DimensionScore,
    aggregate_evaluations,
)
from vllmonline.eval.reporter import generate_report, generate_simple_report


def _make_comparison(score_a: float, score_b: float) -> ComparisonResult:
    return ComparisonResult(
        accuracy=DimensionScore("B", score_a, score_b, "x"),
        completeness=DimensionScore("B", score_a, score_b, "x"),
        safety=DimensionScore("tie", score_a, score_b, "x"),
    )


class TestGenerateReport:
    def test_contains_key_sections(self) -> None:
        results = [_make_comparison(0.7 + i * 0.01, 0.85 + i * 0.01) for i in range(10)]
        agg = aggregate_evaluations(results)
        md = generate_report(agg)
        assert "# vLLMonline A/B 评测报告" in md
        assert "## 概要" in md
        assert "## 各维度详情" in md
        assert "## 判定说明" in md
        assert "## 统计细节" in md

    def test_contains_sample_count(self) -> None:
        # 用微小扰动避免方差为 0
        results = [_make_comparison(0.7 + i * 0.001, 0.85 + i * 0.001) for i in range(5)]
        agg = aggregate_evaluations(results)
        md = generate_report(agg)
        assert "5" in md

    def test_contains_p_value_and_recommendation(self) -> None:
        results = [_make_comparison(0.7 + i * 0.01, 0.85 + i * 0.01) for i in range(10)]
        agg = aggregate_evaluations(results)
        md = generate_report(agg)
        # p 值（小数形式）
        assert "p 值" in md
        # 建议 advance（v2 明显更好）
        assert "advance" in md.lower() or "推进" in md

    def test_custom_title(self) -> None:
        results = [_make_comparison(0.7, 0.85), _make_comparison(0.71, 0.86)]
        agg = aggregate_evaluations(results)
        md = generate_report(agg, title="自定义报告")
        assert "# 自定义报告" in md

    def test_dimension_table_has_all_three(self) -> None:
        results = [_make_comparison(0.7, 0.85), _make_comparison(0.71, 0.86)]
        agg = aggregate_evaluations(results)
        md = generate_report(agg)
        assert "accuracy" in md
        assert "completeness" in md
        assert "safety" in md

    def test_rollback_message(self) -> None:
        results = [_make_comparison(0.85 + i * 0.01, 0.70 + i * 0.01) for i in range(10)]
        agg = aggregate_evaluations(results)
        md = generate_report(agg)
        assert "回滚" in md

    def test_hold_message(self) -> None:
        # 接近的分数 + 少样本 → 不显著 → hold
        results = [_make_comparison(0.80, 0.81), _make_comparison(0.79, 0.80)]
        agg = aggregate_evaluations(results)
        md = generate_report(agg)
        assert "延长" in md or "证据不足" in md


class TestGenerateSimpleReport:
    def test_from_score_lists(self) -> None:
        v1 = [0.7 + i * 0.01 for i in range(10)]
        v2 = [0.85 + i * 0.01 for i in range(10)]
        md = generate_simple_report(v1, v2)
        assert "# vLLMonline A/B 评测报告" in md
        assert "advance" in md.lower() or "推进" in md

    def test_simple_report_rollback(self) -> None:
        v1 = [0.85 + i * 0.01 for i in range(10)]
        v2 = [0.70 + i * 0.01 for i in range(10)]
        md = generate_simple_report(v1, v2)
        assert "回滚" in md
