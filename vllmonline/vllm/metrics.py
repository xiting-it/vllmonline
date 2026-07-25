"""vLLM /metrics 解析 + relabel（SPEC §7.3）。

vLLM 原生 /metrics 端点不区分模型版本（只有 model_name label）。
vllmonline 需要：
    1. 解析 vLLM Prometheus text format
    2. 为每个指标追加 model_version label
    3. 通过自己的 /metrics 端点暴露

本模块只做解析 + relabel（纯函数，不依赖 Prometheus client）。
合并到全局 registry 由 P5 metrics collector 协调。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricSample:
    """一个 Prometheus metric 样本。"""

    name: str  # 不含 label 的 metric 名（如 vllm:request_success）
    labels: dict[str, str]  # 已解析的 label kv
    value: float
    # 原始文本（用于调试）
    raw: str = ""


def parse_prometheus_text(text: str) -> list[MetricSample]:
    """解析 Prometheus text exposition format。

    支持：
        # HELP / # TYPE 注释行（跳过）
        metric_name{label1="v1",label2="v2"} 123.45
        metric_name 67.89  （无 label）

    不支持（vLLM 不用这些，故简化）：
        histogram 的 _sum / _count（按普通样本处理）
        quantile label（按普通 label 处理）
    """
    samples: list[MetricSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sample = _parse_metric_line(line)
        if sample is not None:
            samples.append(sample)
    return samples


# 匹配 metric_name{labels} value 或 metric_name value
# metric_name 允许冒号（vllm:xxx）
_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+"
    r"(?P<value>[-+]?[\d.eE+-]+|NaN|\+Inf|-Inf)"
    r"(?:\s+(?P<timestamp>\d+))?$"
)


def _parse_metric_line(line: str) -> MetricSample | None:
    """解析单行 metric。"""
    m = _LINE_RE.match(line)
    if not m:
        return None
    name = m.group("name")
    labels_str = m.group("labels") or ""
    value_str = m.group("value")

    labels = _parse_labels(labels_str)

    # value 处理
    if value_str in ("NaN",):
        value = float("nan")
    elif value_str in ("+Inf", "Inf"):
        value = float("inf")
    elif value_str == "-Inf":
        value = float("-inf")
    else:
        try:
            value = float(value_str)
        except ValueError:
            return None

    return MetricSample(name=name, labels=labels, value=value, raw=line)


# 匹配 label="value"，value 内的转义 \" \\ 需处理
_LABEL_RE = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:[^"\\]|\\.)*)"')


def _parse_labels(labels_str: str) -> dict[str, str]:
    """解析 labels 字符串（label1="v1", label2="v2"）。"""
    result: dict[str, str] = {}
    for m in _LABEL_RE.finditer(labels_str):
        key = m.group("key")
        # 反转义 \" 和 \\
        value = m.group("value").replace('\\"', '"').replace("\\\\", "\\")
        result[key] = value
    return result


def relabel_with_version(
    samples: list[MetricSample],
    model_version: str,
    *,
    name_prefix: str = "vllmonline_",
) -> list[MetricSample]:
    """给一批样本追加 model_version label，并重命名 metric。

    SPEC §7.3：
        vllm:request_success{model_name="qwen-7b"} 12345
        → vllmonline:request_success{model_name="qwen-7b",model_version="v1"} 12345

    注意：原始 vllm: 前缀的冒号在 Prometheus 里合法，但 vllmonline 统一用下划线
    （prometheus_client Python 库的命名规范）。本函数把 vllm:xxx 改成 vllmonline_xxx。

    Args:
        samples: 原始样本
        model_version: 要追加的版本标签值
        name_prefix: 新 metric 名前缀（默认 vllmonline_）

    Returns:
        新的 MetricSample 列表（不修改原样本）
    """
    result: list[MetricSample] = []
    for s in samples:
        # 重命名：vllm:foo → vllmonline_foo（冒号 → 下划线 + 加前缀）
        new_name = _rename_metric(s.name, name_prefix)
        # 追加 model_version label（不覆盖已有的）
        new_labels = (
            {**s.labels, "model_version": model_version}
            if "model_version" not in s.labels
            else s.labels
        )
        result.append(MetricSample(name=new_name, labels=new_labels, value=s.value, raw=s.raw))
    return result


def _rename_metric(original: str, prefix: str) -> str:
    """重命名 metric：vllm:foo → vllmonline_foo。

    规则：
        - 替换冒号为下划线
        - 如果不以 vllmonline 开头，加前缀
    """
    # 把 vllm: 替换成 vllmonline_
    if original.startswith("vllm:"):
        return prefix + original[len("vllm:") :]
    # 已经是 vllmonline_ 开头的不动
    if original.startswith("vllmonline"):
        return original
    # 其他：加前缀
    name = original.replace(":", "_")
    if not name.startswith(prefix):
        name = prefix + name
    return name
