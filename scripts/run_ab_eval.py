#!/usr/bin/env python3
"""A/B 评测脚本：对比两个 vLLM 实例上的模型，输出 Markdown 报告 + 终端表格。

用法（在 pod 里，vllmonline + 两个 vLLM 都跑着）：
    cd /mnt/workspace/vllmonline
    PYTHONPATH=. python scripts/run_ab_eval.py

输出：
    1. 终端表格（直接截图用）：每个 prompt 的 v1/v2 回答 + 三维度评分
    2. Markdown 报告：scripts/reports/ab_eval_<timestamp>.md
    3. t-test 结果：t / p / Cohen's d / recommendation

设计：
    - judge 用 v1（7B，质量较高）当裁判
    - 被评测对象：v1 自己（baseline）vs v2（1.5B，候选）
    - 同一组 5 个 prompt，每个 prompt 双发 + judge 评分
    - 最后聚合做 Welch's t-test
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

# 把项目根加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from vllmonline.eval.judge import (
    ComparisonResult,
    LLMJudge,
    aggregate_evaluations,
)
from vllmonline.eval.reporter import generate_report
from vllmonline.vllm.client import VLLMClient

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

V1_ENDPOINT = "http://localhost:8000"  # Qwen2.5-7B
V2_ENDPOINT = "http://localhost:8001"  # Qwen2.5-1.5B
JUDGE_ENDPOINT = V1_ENDPOINT  # 用 7B 当 judge（避免 1.5B 自评偏差）
MODEL_NAME = "qwen-7b"  # 两个 vLLM 都 served-model-name=qwen-7b

# 评测用的 prompt（选能明显看出质量差异的）
PROMPTS = [
    "请用 100 字解释量子纠缠现象。",
    "写一首关于秋天的七言绝句。",
    "Python 中 GIL 是什么？为什么多线程不能并行？",
    "总结《红楼梦》前 20 回的主要情节。",
    "解释 ROCm 和 CUDA 的主要区别。",
]


async def collect_response(client: VLLMClient, endpoint: str, prompt: str) -> str:
    """收集一个模型对 prompt 的回答。"""
    resp = await client.chat_completion(
        endpoint,
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.0,  # 评测要确定性
        },
    )
    return resp.get("choices", [{}])[0].get("message", {}).get("content", "")


async def main() -> None:
    print("=" * 70)
    print("vLLMonline A/B 评测")
    print(f"  v1 (baseline): {V1_ENDPOINT}")
    print(f"  v2 (candidate): {V2_ENDPOINT}")
    print(f"  judge: {JUDGE_ENDPOINT}")
    print(f"  prompts: {len(PROMPTS)} 个")
    print("=" * 70)
    print()

    client = VLLMClient()
    judge = LLMJudge(
        client=client,
        judge_endpoint=JUDGE_ENDPOINT,
        judge_model=MODEL_NAME,
    )

    results: list[ComparisonResult] = []
    raw_responses: list[dict] = []

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] 评测中: {prompt[:40]}...")
        t0 = time.time()

        # 双发
        text_v1 = await collect_response(client, V1_ENDPOINT, prompt)
        text_v2 = await collect_response(client, V2_ENDPOINT, prompt)

        # Judge 评分
        try:
            result = await judge.compare(prompt, text_v1, text_v2)
            results.append(result)
            raw_responses.append(
                {
                    "prompt": prompt,
                    "v1": text_v1,
                    "v2": text_v2,
                    "result": result,
                }
            )
            elapsed = time.time() - t0
            print(
                f"        v1={result.score_v1_mean:.3f}  v2={result.score_v2_mean:.3f}  ({elapsed:.1f}s)"
            )
        except Exception as e:
            print(f"        ✗ Judge 失败: {e}")
            print(f"        v1 回答: {text_v1[:80]}...")
            print(f"        v2 回答: {text_v2[:80]}...")

    await client.aclose()

    if len(results) < 2:
        print("\n✗ 有效评测样本不足（< 2），无法做 t-test")
        return

    # 聚合统计
    print()
    print("=" * 70)
    print("聚合统计")
    print("=" * 70)
    agg = aggregate_evaluations(results)
    print(f"  样本量:       {agg.sample_count}")
    print(f"  v1 平均分:    {agg.score_v1_mean:.4f}")
    print(f"  v2 平均分:    {agg.score_v2_mean:.4f}")
    print(f"  t 统计量:     {agg.t_test.t_statistic:.4f}")
    print(f"  p 值:         {agg.t_test.p_value:.4f}")
    print(f"  Cohen's d:    {agg.t_test.cohen_d:.4f}")
    print(f"  自由度:       {agg.t_test.degrees_of_freedom:.2f}")
    print(f"  统计显著:     {'是' if agg.t_test.significant else '否'} (p<0.05)")
    print(f"  建议:         {agg.recommendation}")
    print()

    # 终端表格（截图用）
    print("=" * 70)
    print("逐 prompt 评分明细（终端表格，可截图）")
    print("=" * 70)
    print(f"{'#':<3} {'prompt':<32} {'v1分':<8} {'v2分':<8} {'winner':<8}")
    print("-" * 70)
    for i, r in enumerate(raw_responses, 1):
        p = r["prompt"][:30]
        v1 = r["result"].score_v1_mean
        v2 = r["result"].score_v2_mean
        # 综合三维度判断 winner
        if v1 > v2:
            w = "v1"
        elif v2 > v1:
            w = "v2"
        else:
            w = "tie"
        print(f"{i:<3} {p:<32} {v1:<8.3f} {v2:<8.3f} {w:<8}")
    print()

    # 三维度均分
    print("=" * 70)
    print("三维度均分对比")
    print("=" * 70)
    print(f"{'维度':<15} {'v1':<10} {'v2':<10} {'差异':<10}")
    print("-" * 50)
    for dim in ["accuracy", "completeness", "safety"]:
        v1 = agg.dimension_scores_v1[dim]
        v2 = agg.dimension_scores_v2[dim]
        diff = v2 - v1
        sign = "+" if diff >= 0 else ""
        print(f"{dim:<15} {v1:<10.4f} {v2:<10.4f} {sign}{diff:<9.4f}")
    print()

    # 生成 Markdown 报告
    reports_dir = Path("scripts/reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"ab_eval_{timestamp}.md"

    md = generate_report(agg, title=f"vLLMonline A/B 评测报告（{timestamp}）")
    # 追加每个 prompt 的回答详情
    md += "\n## 回答详情\n\n"
    for i, r in enumerate(raw_responses, 1):
        md += f"### Prompt {i}: {r['prompt']}\n\n"
        md += f"**v1 回答**（baseline，得分 {r['result'].score_v1_mean:.3f}）:\n\n"
        md += f"> {r['v1']}\n\n"
        md += f"**v2 回答**（candidate，得分 {r['result'].score_v2_mean:.3f}）:\n\n"
        md += f"> {r['v2']}\n\n"
        md += "**Judge 三维度**:\n"
        md += f"- accuracy: v1={r['result'].accuracy.score_a:.3f} / v2={r['result'].accuracy.score_b:.3f} ({r['result'].accuracy.rationale})\n"
        md += f"- completeness: v1={r['result'].completeness.score_a:.3f} / v2={r['result'].completeness.score_b:.3f}\n"
        md += f"- safety: v1={r['result'].safety.score_a:.3f} / v2={r['result'].safety.score_b:.3f}\n\n"

    report_path.write_text(md, encoding="utf-8")
    print(f"✓ Markdown 报告已生成: {report_path}")
    print()
    print("=" * 70)
    print(f"结论: {agg.recommendation.upper()}")
    if agg.recommendation == "advance":
        print("v2 在统计上显著优于 v1，建议推进灰度。")
    elif agg.recommendation == "rollback":
        print("v2 在统计上显著劣于 v1，建议立即回滚。")
    else:
        print("证据不足，建议延长当前阶段继续收集样本。")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
