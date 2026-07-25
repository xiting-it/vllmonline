"""灰度管理 API 测试（SPEC §8.2-8.3 灰度部分）。

测试矩阵：
    - POST /api/canary/start 启动（含校验：模型必须 ACTIVE）
    - GET /api/canary/{id}/status
    - POST /api/canary/{id}/advance 推进阶段
    - POST /api/canary/{id}/rollback 回滚
    - 错误：404 / 409
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


async def _register_and_activate(client: AsyncClient, model_id: str) -> None:
    """注册并激活一个模型。"""
    name, ver = model_id.rsplit("-", 1)
    await client.post(
        "/api/models/register",
        json={
            "model_name": name,
            "version": ver,
            "endpoint": f"http://{model_id}:8000",
            "params_billion": 7.0,
            "dtype": "fp16",
        },
    )
    await client.post(f"/api/models/{model_id}/load")


async def test_canary_start_success(client: AsyncClient) -> None:
    await _register_and_activate(client, "qwen-7b-v1")
    await _register_and_activate(client, "qwen-7b-v2")
    r = await client.post(
        "/api/canary/start",
        json={
            "model_v1": "qwen-7b-v1",
            "model_v2": "qwen-7b-v2",
            "strategy": "gradual",
            "stages": [0.1, 0.3, 1.0],
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"].startswith("canary-")
    assert "STAGE_10" in body["current_stage"]
    assert body["traffic_split"]["qwen-7b-v2"] == pytest.approx(0.1)


async def test_canary_start_requires_active_models(client: AsyncClient) -> None:
    """v1/v2 必须 ACTIVE。"""
    # 只注册不激活
    await client.post(
        "/api/models/register",
        json={
            "model_name": "qwen-7b",
            "version": "v1",
            "endpoint": "http://x",
            "params_billion": 7.0,
            "dtype": "fp16",
        },
    )
    await client.post(
        "/api/models/register",
        json={
            "model_name": "qwen-7b",
            "version": "v2",
            "endpoint": "http://x",
            "params_billion": 7.0,
            "dtype": "fp16",
        },
    )
    r = await client.post(
        "/api/canary/start",
        json={"model_v1": "qwen-7b-v1", "model_v2": "qwen-7b-v2"},
    )
    assert r.status_code == 409


async def test_canary_start_unknown_model_404(client: AsyncClient) -> None:
    r = await client.post(
        "/api/canary/start",
        json={"model_v1": "nope-v1", "model_v2": "nope-v2"},
    )
    assert r.status_code == 404


async def test_canary_status(client: AsyncClient) -> None:
    await _register_and_activate(client, "qwen-7b-v1")
    await _register_and_activate(client, "qwen-7b-v2")
    start = await client.post(
        "/api/canary/start",
        json={"model_v1": "qwen-7b-v1", "model_v2": "qwen-7b-v2"},
    )
    deployment_id = start.json()["id"]

    r = await client.get(f"/api/canary/{deployment_id}/status")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == deployment_id
    assert body["status"] == "IN_PROGRESS"
    assert "STAGE" in body["current_stage"]


async def test_canary_status_not_found(client: AsyncClient) -> None:
    r = await client.get("/api/canary/nonexistent/status")
    assert r.status_code == 404


async def test_canary_advance(client: AsyncClient) -> None:
    await _register_and_activate(client, "qwen-7b-v1")
    await _register_and_activate(client, "qwen-7b-v2")
    start = await client.post(
        "/api/canary/start",
        json={
            "model_v1": "qwen-7b-v1",
            "model_v2": "qwen-7b-v2",
            "stages": [0.1, 0.3, 1.0],
        },
    )
    deployment_id = start.json()["id"]

    # 推进到 STAGE_30%
    r = await client.post(f"/api/canary/{deployment_id}/advance")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["advanced"] is True
    assert "STAGE_30" in body["current_stage"]
    assert body["traffic_split"]["qwen-7b-v2"] == pytest.approx(0.3)


async def test_canary_advance_to_completion(client: AsyncClient) -> None:
    await _register_and_activate(client, "qwen-7b-v1")
    await _register_and_activate(client, "qwen-7b-v2")
    start = await client.post(
        "/api/canary/start",
        json={
            "model_v1": "qwen-7b-v1",
            "model_v2": "qwen-7b-v2",
            "stages": [0.1, 0.3, 1.0],
        },
    )
    deployment_id = start.json()["id"]

    # 推进 3 次到完成
    await client.post(f"/api/canary/{deployment_id}/advance")  # → 30%
    await client.post(f"/api/canary/{deployment_id}/advance")  # → 100%
    r = await client.post(f"/api/canary/{deployment_id}/advance")  # → 完成
    assert r.status_code == 200
    body = r.json()
    assert body["current_stage"] == "COMPLETED"

    # 再推进应 409
    r2 = await client.post(f"/api/canary/{deployment_id}/advance")
    assert r2.status_code == 409


async def test_canary_rollback(client: AsyncClient) -> None:
    await _register_and_activate(client, "qwen-7b-v1")
    await _register_and_activate(client, "qwen-7b-v2")
    start = await client.post(
        "/api/canary/start",
        json={"model_v1": "qwen-7b-v1", "model_v2": "qwen-7b-v2"},
    )
    deployment_id = start.json()["id"]

    r = await client.post(f"/api/canary/{deployment_id}/rollback", params={"reason": "测试回滚"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ROLLED_BACK"

    # 验证 v2 被回滚（sleep）
    v2 = await client.get("/api/models/qwen-7b-v2")
    assert v2.json()["state"] == "SLEEPING"


async def test_canary_rollback_already_rolled_back(client: AsyncClient) -> None:
    await _register_and_activate(client, "qwen-7b-v1")
    await _register_and_activate(client, "qwen-7b-v2")
    start = await client.post(
        "/api/canary/start",
        json={"model_v1": "qwen-7b-v1", "model_v2": "qwen-7b-v2"},
    )
    deployment_id = start.json()["id"]

    await client.post(f"/api/canary/{deployment_id}/rollback")
    # 再次回滚应 409
    r = await client.post(f"/api/canary/{deployment_id}/rollback")
    assert r.status_code == 409


async def test_canary_metrics(client: AsyncClient) -> None:
    await _register_and_activate(client, "qwen-7b-v1")
    await _register_and_activate(client, "qwen-7b-v2")
    start = await client.post(
        "/api/canary/start",
        json={"model_v1": "qwen-7b-v1", "model_v2": "qwen-7b-v2"},
    )
    deployment_id = start.json()["id"]

    r = await client.get(f"/api/canary/{deployment_id}/metrics")
    assert r.status_code == 200
    assert "traffic_split" in r.json()
