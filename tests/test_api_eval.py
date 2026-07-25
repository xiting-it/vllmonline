"""评测 API 测试（SPEC §8.2 评测部分）。

POST /api/eval/compare 需要 fake vLLM 返回可解析的 judge JSON。
但 vllmonline 的 eval 路由内部自建 VLLMClient（不接 set_test_backend），
所以这里用更直接的方式：测路由的契约（404/422），不测完整 judge 流程
（judge 的逻辑已在 test_judge.py 覆盖）。
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


async def test_eval_compare_unknown_models_404(client: AsyncClient) -> None:
    """未注册的模型 → 404。"""
    r = await client.post(
        "/api/eval/compare",
        json={
            "model_v1": "nope-v1",
            "model_v2": "nope-v2",
            "prompts": ["hi"],
        },
    )
    assert r.status_code == 404


async def test_eval_compare_empty_prompts_422(client: AsyncClient) -> None:
    """prompts 为空 → 422 schema 校验。"""
    r = await client.post(
        "/api/eval/compare",
        json={"model_v1": "x-v1", "model_v2": "x-v2", "prompts": []},
    )
    assert r.status_code == 422


async def test_eval_compare_missing_field_422(client: AsyncClient) -> None:
    """缺 model_v1 → 422。"""
    r = await client.post(
        "/api/eval/compare",
        json={"model_v2": "x-v2", "prompts": ["hi"]},
    )
    assert r.status_code == 422


async def test_eval_report_not_found(client: AsyncClient) -> None:
    r = await client.get("/api/eval/nonexistent/report")
    assert r.status_code == 404


async def test_eval_compare_requires_active_models(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模型必须注册——这里注册但走完整流程会因 fake vLLM 返回非 JSON 而失败。

    验证路由能正确进入 judge 流程（至少不是 500 内部错误，而是 422 样本不足）。
    用 monkeypatch 让 LLMJudge 直接返回构造好的结果。
    """
    # 注册两个模型（IDLE 即可，eval 路由不强校验 state）
    for ver in ["v1", "v2"]:
        await client.post(
            "/api/models/register",
            json={
                "model_name": "qwen-7b",
                "version": ver,
                "endpoint": f"http://{ver}:8000",
                "params_billion": 7.0,
                "dtype": "fp16",
            },
        )

    # monkeypatch LLMJudge.evaluate_pair 返回有变化的结果（避免方差为 0）
    from vllmonline.eval.judge import (
        ComparisonResult,
        DimensionScore,
    )

    call_count = {"n": 0}

    async def fake_eval(self, user_prompt, v1_endpoint, v2_endpoint, model_v1, model_v2):
        call_count["n"] += 1
        # 每次返回略有不同的分数（让 t-test 有方差）
        delta = call_count["n"] * 0.01
        return ComparisonResult(
            accuracy=DimensionScore("B", 0.70 + delta, 0.85 + delta, "v2 better"),
            completeness=DimensionScore("B", 0.65 + delta, 0.80 + delta, "v2 more complete"),
            safety=DimensionScore("tie", 0.90 + delta, 0.92 + delta, "both safe"),
        )

    monkeypatch.setattr("vllmonline.eval.judge.LLMJudge.evaluate_pair", fake_eval)

    r = await client.post(
        "/api/eval/compare",
        json={
            "model_v1": "qwen-7b-v1",
            "model_v2": "qwen-7b-v2",
            "prompts": ["什么是 AI？", "解释机器学习", "Python 是什么"],
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["sample_count"] == 3
    assert body["recommendation"] == "advance"  # v2 三维度都更好
    assert body["significant"] is True

    # 报告可查
    eval_id = body["id"]
    r2 = await client.get(f"/api/eval/{eval_id}/report")
    assert r2.status_code == 200
    report = r2.json()
    assert report["sample_count"] == 3
    assert report["recommendation"] == "advance"
