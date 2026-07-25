"""eval/judge.py 测试（SPEC §6.1-6.2）。

测试矩阵：
    - parse_judge_response：JSON 解析 + 容错（markdown 包裹、缺字段、非法分数）
    - LLMJudge.compare：用 fake vLLM 返回构造好的 judge JSON
    - aggregate_evaluations：多结果聚合 + t-test
    - JUDGE_PROMPT_TEMPLATE 格式
"""

from __future__ import annotations

import pytest

from vllmonline.eval.judge import (
    ComparisonResult,
    DimensionScore,
    JudgeParseError,
    LLMJudge,
    aggregate_evaluations,
    parse_judge_response,
)
from vllmonline.eval.statistics import welch_ttest
from vllmonline.vllm.client import VLLMClient

# ─────────────────────────────────────────────────────────────────────────────
# parse_judge_response
# ─────────────────────────────────────────────────────────────────────────────


GOOD_JUDGE_JSON = """{
  "accuracy":   {"winner": "B", "score_a": 0.7, "score_b": 0.9, "rationale": "B 更准"},
  "completeness": {"winner": "B", "score_a": 0.6, "score_b": 0.85, "rationale": "B 更全"},
  "safety":    {"winner": "tie", "score_a": 0.95, "score_b": 0.95, "rationale": "都安全"}
}"""


class TestParseJudgeResponse:
    def test_parse_valid_json(self) -> None:
        result = parse_judge_response(GOOD_JUDGE_JSON)
        assert isinstance(result, ComparisonResult)
        assert result.accuracy.winner == "B"
        assert result.accuracy.score_b == 0.9
        assert result.completeness.score_a == 0.6
        assert result.safety.winner == "tie"

    def test_parse_json_in_markdown_code_block(self) -> None:
        content = f"以下是评测结果：\n```json\n{GOOD_JUDGE_JSON}\n```\n"
        result = parse_judge_response(content)
        assert result.accuracy.score_b == 0.9

    def test_parse_json_embedded_in_text(self) -> None:
        """JSON 前后有解释文字。"""
        content = f"好的，我来评测：\n{GOOD_JUDGE_JSON}\n以上就是结果。"
        result = parse_judge_response(content)
        assert result.accuracy.score_a == 0.7

    def test_missing_dimension_defaults_to_tie(self) -> None:
        """缺一个维度 → 默认 0.5/tie。"""
        content = """{
          "accuracy": {"winner": "B", "score_a": 0.7, "score_b": 0.9, "rationale": "x"}
        }"""
        result = parse_judge_response(content)
        assert result.accuracy.score_b == 0.9
        # 缺的 completeness / safety 用默认
        assert result.completeness.score_a == 0.5
        assert result.safety.winner == "tie"

    def test_missing_score_field_defaults_to_half(self) -> None:
        content = """{
          "accuracy": {"winner": "A"}
        }"""
        result = parse_judge_response(content)
        assert result.accuracy.score_a == 0.5
        assert result.accuracy.score_b == 0.5

    def test_score_out_of_range_clamped(self) -> None:
        """分数超 [0,1] 被 clamp。"""
        content = """{
          "accuracy": {"score_a": 1.5, "score_b": -0.3, "winner": "tie"}
        }"""
        result = parse_judge_response(content)
        assert result.accuracy.score_a == 1.0
        assert result.accuracy.score_b == 0.0

    def test_non_numeric_score_handled(self) -> None:
        """非数字分数 → 默认 0.5。"""
        content = """{
          "accuracy": {"score_a": "good", "score_b": null, "winner": "tie"}
        }"""
        result = parse_judge_response(content)
        assert result.accuracy.score_a == 0.5
        assert result.accuracy.score_b == 0.5

    def test_completely_invalid_raises(self) -> None:
        """完全无 JSON → JudgeParseError。"""
        with pytest.raises(JudgeParseError):
            parse_judge_response("这根本不是 JSON，也没大括号")

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(JudgeParseError):
            parse_judge_response("{invalid json content")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(JudgeParseError):
            parse_judge_response("")


# ─────────────────────────────────────────────────────────────────────────────
# ComparisonResult 派生属性
# ─────────────────────────────────────────────────────────────────────────────


class TestComparisonResult:
    def test_score_means(self) -> None:
        result = parse_judge_response(GOOD_JUDGE_JSON)
        # v1 mean = (0.7 + 0.6 + 0.95) / 3
        assert result.score_v1_mean == pytest.approx((0.7 + 0.6 + 0.95) / 3)
        # v2 mean = (0.9 + 0.85 + 0.95) / 3
        assert result.score_v2_mean == pytest.approx((0.9 + 0.85 + 0.95) / 3)

    def test_dimension_scores_dict(self) -> None:
        result = parse_judge_response(GOOD_JUDGE_JSON)
        assert result.dimension_scores_v1["accuracy"] == 0.7
        assert result.dimension_scores_v2["safety"] == 0.95

    def test_frozen(self) -> None:
        result = parse_judge_response(GOOD_JUDGE_JSON)
        with pytest.raises(AttributeError):
            result.accuracy = DimensionScore("A", 1.0, 0.0, "x")  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# LLMJudge（用 fake vLLM）
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMJudge:
    async def test_compare_with_fake_vllm(self) -> None:
        """fake vLLM 返回稳定内容，judge 解析它。

        注：fake vLLM 返回的内容是从 prompt 派生的 token，不是真 JSON。
        所以这里手动 mock client.chat_completion。
        """
        # 直接 mock client 的 chat_completion 返回 GOOD_JUDGE_JSON

        client = VLLMClient.__new__(VLLMClient)
        client._owns_client = False  # type: ignore[attr-defined]

        async def fake_chat_completion(endpoint, payload):
            return {"choices": [{"message": {"content": GOOD_JUDGE_JSON, "role": "assistant"}}]}

        client.chat_completion = fake_chat_completion  # type: ignore[method-assign]

        judge = LLMJudge(client, "http://judge:8000", "judge-model")
        result = await judge.compare("什么是 AI？", "AI 是...", "AI 是人工智能...")
        assert result.accuracy.winner == "B"
        assert result.score_v2_mean > result.score_v1_mean

    def test_build_prompt_contains_all_fields(self) -> None:
        """prompt 模板含 user_prompt + 两个 response。"""
        client = VLLMClient.__new__(VLLMClient)
        client._owns_client = False  # type: ignore[attr-defined]
        judge = LLMJudge(client, "http://x", "m")
        prompt = judge.build_prompt("hi", "resp A", "resp B")
        assert "hi" in prompt
        assert "resp A" in prompt
        assert "resp B" in prompt
        assert "accuracy" in prompt
        assert "completeness" in prompt
        assert "safety" in prompt
        assert "JSON" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_evaluations
# ─────────────────────────────────────────────────────────────────────────────


def _make_comparison(score_a: float, score_b: float) -> ComparisonResult:
    """构造一个简化 ComparisonResult（三维度同分）。"""
    return ComparisonResult(
        accuracy=DimensionScore(
            winner="B" if score_b > score_a else "A",
            score_a=score_a,
            score_b=score_b,
            rationale="x",
        ),
        completeness=DimensionScore(
            winner="B" if score_b > score_a else "A",
            score_a=score_a,
            score_b=score_b,
            rationale="x",
        ),
        safety=DimensionScore(winner="tie", score_a=score_a, score_b=score_b, rationale="x"),
    )


class TestAggregate:
    def test_aggregate_v2_better(self) -> None:
        results = [
            _make_comparison(0.70, 0.85),
            _make_comparison(0.72, 0.87),
            _make_comparison(0.68, 0.83),
            _make_comparison(0.71, 0.86),
        ]
        agg = aggregate_evaluations(results)
        assert agg.sample_count == 4
        assert agg.score_v2_mean > agg.score_v1_mean
        assert agg.recommendation == "advance"

    def test_aggregate_v2_worse(self) -> None:
        results = [
            _make_comparison(0.85, 0.70),
            _make_comparison(0.87, 0.72),
            _make_comparison(0.83, 0.68),
            _make_comparison(0.86, 0.71),
        ]
        agg = aggregate_evaluations(results)
        assert agg.recommendation == "rollback"

    def test_aggregate_dimension_means(self) -> None:
        results = [_make_comparison(0.6, 0.9), _make_comparison(0.7, 0.8)]
        agg = aggregate_evaluations(results)
        # accuracy v1 = (0.6 + 0.7) / 2
        assert agg.dimension_scores_v1["accuracy"] == pytest.approx(0.65)
        assert agg.dimension_scores_v2["accuracy"] == pytest.approx(0.85)

    def test_aggregate_insufficient_raises(self) -> None:
        with pytest.raises(ValueError, match="至少需要 2"):
            aggregate_evaluations([_make_comparison(0.5, 0.6)])

    def test_aggregate_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="至少需要 2"):
            aggregate_evaluations([])

    def test_aggregate_t_test_matches_direct(self) -> None:
        """聚合的 t-test 应与直接对均分序列算的一致。"""
        results = [_make_comparison(0.7 + i * 0.01, 0.85 + i * 0.01) for i in range(10)]
        agg = aggregate_evaluations(results)
        v1_means = [r.score_v1_mean for r in results]
        v2_means = [r.score_v2_mean for r in results]
        direct = welch_ttest(v1_means, v2_means)
        assert agg.t_test.t_statistic == pytest.approx(direct.t_statistic)
        assert agg.t_test.p_value == pytest.approx(direct.p_value)
