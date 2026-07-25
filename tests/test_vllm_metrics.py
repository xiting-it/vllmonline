"""vllm/metrics.py 解析 + relabel 测试（SPEC §7.3）。"""

from __future__ import annotations

from vllmonline.vllm.metrics import (
    MetricSample,
    parse_prometheus_text,
    relabel_with_version,
)

# ─────────────────────────────────────────────────────────────────────────────
# 解析
# ─────────────────────────────────────────────────────────────────────────────


class TestParse:
    def test_simple_metric_no_labels(self) -> None:
        samples = parse_prometheus_text("foo_bar 42\n")
        assert len(samples) == 1
        assert samples[0].name == "foo_bar"
        assert samples[0].value == 42.0
        assert samples[0].labels == {}

    def test_metric_with_labels(self) -> None:
        text = 'http_requests_total{method="GET",code="200"} 1027\n'
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "http_requests_total"
        assert samples[0].value == 1027.0
        assert samples[0].labels == {"method": "GET", "code": "200"}

    def test_vllm_metric_with_colon(self) -> None:
        """vLLM 用 vllm: 前缀（含冒号）。"""
        text = 'vllm:request_success{model_name="qwen-7b"} 12345\n'
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "vllm:request_success"
        assert samples[0].value == 12345.0
        assert samples[0].labels == {"model_name": "qwen-7b"}

    def test_skip_comments_and_empty(self) -> None:
        text = """
# HELP foo Total
# TYPE foo counter
foo 1

foo_bar 2
"""
        samples = parse_prometheus_text(text)
        assert len(samples) == 2
        assert {s.name for s in samples} == {"foo", "foo_bar"}

    def test_negative_and_float_values(self) -> None:
        samples = parse_prometheus_text("a -1.5\nb 3.14\nc -100\n")
        assert [s.value for s in samples] == [-1.5, 3.14, -100.0]

    def test_special_values(self) -> None:
        samples = parse_prometheus_text("a NaN\nb +Inf\nc -Inf\n")
        import math

        assert math.isnan(samples[0].value)
        assert samples[1].value == float("inf")
        assert samples[2].value == float("-inf")

    def test_label_with_escaped_quote(self) -> None:
        """label 值含转义引号。"""
        text = 'foo{name="a\\"b"} 1\n'
        samples = parse_prometheus_text(text)
        assert samples[0].labels == {"name": 'a"b'}

    def test_histogram_buckets(self) -> None:
        """histogram 的 _bucket/_sum/_count 当普通样本处理。"""
        text = """
http_duration_bucket{le="0.1"} 1
http_duration_bucket{le="+Inf"} 5
http_duration_sum 12.34
http_duration_count 5
"""
        samples = parse_prometheus_text(text)
        assert len(samples) == 4
        names = {s.name for s in samples}
        assert "http_duration_bucket" in names

    def test_malformed_line_skipped(self) -> None:
        text = "garbage line\nfoo 1\nnot a metric\n"
        samples = parse_prometheus_text(text)
        # 只解析出 foo
        assert len(samples) == 1
        assert samples[0].name == "foo"

    def test_empty_text(self) -> None:
        assert parse_prometheus_text("") == []

    def test_real_vllm_output(self) -> None:
        """模拟真实 vLLM /metrics 输出片段。"""
        text = """
# HELP vllm:request_success Total successful requests
# TYPE vllm:request_success counter
vllm:request_success{model_name="qwen-7b"} 8234
# HELP vllm:time_to_first_token_seconds TTFT
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="qwen-7b",le="0.05"} 100
vllm:time_to_first_token_seconds_bucket{model_name="qwen-7b",le="+Inf"} 200
vllm:time_to_first_token_seconds_count{model_name="qwen-7b"} 200
vllm:time_to_first_token_seconds_sum{model_name="qwen-7b"} 4.5
vllm:num_requests_running{model_name="qwen-7b"} 3
"""
        samples = parse_prometheus_text(text)
        assert len(samples) == 6
        # 第一个是 counter
        assert samples[0].name == "vllm:request_success"
        assert samples[0].value == 8234


# ─────────────────────────────────────────────────────────────────────────────
# Relabel
# ─────────────────────────────────────────────────────────────────────────────


class TestRelabel:
    def test_adds_model_version_label(self) -> None:
        samples = [
            MetricSample(
                name="vllm:request_success",
                labels={"model_name": "qwen-7b"},
                value=100,
            )
        ]
        result = relabel_with_version(samples, "v1")
        assert result[0].labels["model_version"] == "v1"
        assert result[0].labels["model_name"] == "qwen-7b"  # 保留原 label
        assert result[0].value == 100

    def test_renames_vllm_prefix(self) -> None:
        """vllm:foo → vllmonline_foo（SPEC §7.3 示例）。"""
        samples = [MetricSample(name="vllm:request_success", labels={}, value=1)]
        result = relabel_with_version(samples, "v1")
        assert result[0].name == "vllmonline_request_success"

    def test_doesnt_override_existing_model_version(self) -> None:
        """已有 model_version label 不覆盖。"""
        samples = [
            MetricSample(
                name="foo",
                labels={"model_version": "existing"},
                value=1,
            )
        ]
        result = relabel_with_version(samples, "v2")
        assert result[0].labels["model_version"] == "existing"

    def test_batch_relabel_preserves_count(self) -> None:
        samples = parse_prometheus_text('vllm:a 1\nvllm:b{x="1"} 2\nvllm:c 3\n')
        result = relabel_with_version(samples, "v1")
        assert len(result) == 3
        assert all(s.labels.get("model_version") == "v1" for s in result)

    def test_doesnt_mutate_input(self) -> None:
        """不修改原样本（frozen dataclass + 返回新列表）。"""
        original = MetricSample(name="vllm:foo", labels={"a": "b"}, value=1)
        result = relabel_with_version([original], "v1")
        assert "model_version" not in original.labels
        assert result[0].labels["model_version"] == "v1"
