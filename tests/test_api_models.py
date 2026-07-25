"""模型管理 API 单测（SPEC §8.2-8.3）。

测试矩阵：
    - POST /api/models/register 注册 + 计算显存
    - GET /api/models 列表（含过滤）
    - GET /api/models/{id} 详情
    - DELETE /api/models/{id}（必须 IDLE）
    - POST /api/models/{id}/load | unload | sleep | wake 状态转移
    - 错误场景：404、409（重复注册/非法转移）、422（schema 校验）
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

# ─────────────────────────────────────────────────────────────────────────────
# 注册
# ─────────────────────────────────────────────────────────────────────────────


async def test_register_model_success(client: AsyncClient) -> None:
    """注册成功，返回 201 + 计算的显存字段。"""
    r = await client.post(
        "/api/models/register",
        json={
            "model_name": "qwen-7b",
            "version": "v1",
            "endpoint": "http://vllm:8000/v1",
            "params_billion": 7.0,
            "dtype": "fp16",
            "quantization": None,
            "gpu_id": 0,
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] == "qwen-7b-v1"
    assert body["state"] == "IDLE"
    assert body["weight_gb"] == pytest.approx(14.0)
    assert body["kv_cache_budget_gb"] == pytest.approx(14.0 * 0.25 * 0.9)


async def test_register_model_with_quantization(client: AsyncClient) -> None:
    """带量化的模型显存计算正确。"""
    r = await client.post(
        "/api/models/register",
        json={
            "model_name": "qwen-72b",
            "version": "v1",
            "endpoint": "http://vllm:8000/v1",
            "params_billion": 72.0,
            "dtype": "int4",
            "quantization": "gptq",
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert body["weight_gb"] == pytest.approx(37.8)


async def test_register_duplicate_returns_409(client: AsyncClient) -> None:
    """重复注册同一 id 返回 409。"""
    payload = {
        "model_name": "qwen-7b",
        "version": "v1",
        "endpoint": "http://vllm:8000/v1",
        "params_billion": 7.0,
        "dtype": "fp16",
    }
    r1 = await client.post("/api/models/register", json=payload)
    assert r1.status_code == 201
    r2 = await client.post("/api/models/register", json=payload)
    assert r2.status_code == 409


async def test_register_invalid_dtype_returns_422(client: AsyncClient) -> None:
    """未知 dtype → 422 schema 校验失败。"""
    r = await client.post(
        "/api/models/register",
        json={
            "model_name": "x",
            "version": "v1",
            "endpoint": "http://x",
            "params_billion": 7.0,
            "dtype": "fp8",  # 不支持
        },
    )
    assert r.status_code == 422


async def test_register_invalid_quantization_returns_422(client: AsyncClient) -> None:
    r = await client.post(
        "/api/models/register",
        json={
            "model_name": "x",
            "version": "v1",
            "endpoint": "http://x",
            "params_billion": 7.0,
            "dtype": "fp16",
            "quantization": "unknown",
        },
    )
    assert r.status_code == 422


async def test_register_negative_params_returns_422(client: AsyncClient) -> None:
    r = await client.post(
        "/api/models/register",
        json={
            "model_name": "x",
            "version": "v1",
            "endpoint": "http://x",
            "params_billion": -1.0,
            "dtype": "fp16",
        },
    )
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 列表 + 详情
# ─────────────────────────────────────────────────────────────────────────────


async def test_list_models_empty(client: AsyncClient) -> None:
    r = await client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["models"] == []


async def test_list_models_with_filter(client: AsyncClient) -> None:
    """按 model_name 过滤。"""
    for name, ver in [("qwen-7b", "v1"), ("qwen-7b", "v2"), ("llama", "v1")]:
        await client.post(
            "/api/models/register",
            json={
                "model_name": name,
                "version": ver,
                "endpoint": "http://x",
                "params_billion": 7.0,
                "dtype": "fp16",
            },
        )
    r = await client.get("/api/models", params={"model_name": "qwen-7b"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert all(m["model_name"] == "qwen-7b" for m in body["models"])


async def test_get_model_detail(client: AsyncClient) -> None:
    await client.post(
        "/api/models/register",
        json={
            "model_name": "qwen-7b",
            "version": "v1",
            "endpoint": "http://vllm:8000",
            "params_billion": 7.0,
            "dtype": "fp16",
        },
    )
    r = await client.get("/api/models/qwen-7b-v1")
    assert r.status_code == 200
    assert r.json()["id"] == "qwen-7b-v1"


async def test_get_model_not_found(client: AsyncClient) -> None:
    r = await client.get("/api/models/nonexistent")
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 状态转移
# ─────────────────────────────────────────────────────────────────────────────


async def _register_and_get_state(client: AsyncClient, model_id: str = "qwen-7b-v1") -> str:
    """注册一个模型并返回当前 state。"""
    await client.post(
        "/api/models/register",
        json={
            "model_name": model_id.rsplit("-", 1)[0],
            "version": model_id.rsplit("-", 1)[1],
            "endpoint": "http://vllm:8000",
            "params_billion": 7.0,
            "dtype": "fp16",
        },
    )
    return (await client.get(f"/api/models/{model_id}")).json()["state"]


async def test_load_model_transitions_to_active(client: AsyncClient) -> None:
    """POST /load：IDLE → LOADING → ACTIVE。"""
    await _register_and_get_state(client)
    r = await client.post("/api/models/qwen-7b-v1/load")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "ACTIVE"


async def test_sleep_model(client: AsyncClient) -> None:
    """POST /sleep：ACTIVE → SLEEPING。"""
    await _register_and_get_state(client)
    await client.post("/api/models/qwen-7b-v1/load")
    r = await client.post("/api/models/qwen-7b-v1/sleep")
    assert r.status_code == 200
    assert r.json()["state"] == "SLEEPING"


async def test_wake_model(client: AsyncClient) -> None:
    """POST /wake：SLEEPING → LOADING → ACTIVE。"""
    await _register_and_get_state(client)
    await client.post("/api/models/qwen-7b-v1/load")
    await client.post("/api/models/qwen-7b-v1/sleep")
    r = await client.post("/api/models/qwen-7b-v1/wake")
    assert r.status_code == 200
    assert r.json()["state"] == "ACTIVE"


async def test_unload_model_full_cycle(client: AsyncClient) -> None:
    """POST /unload：ACTIVE → DRAINING → SLEEPING → UNLOADING → IDLE。"""
    await _register_and_get_state(client)
    await client.post("/api/models/qwen-7b-v1/load")
    r = await client.post("/api/models/qwen-7b-v1/unload")
    assert r.status_code == 200
    assert r.json()["state"] == "IDLE"


async def test_illegal_transition_returns_409(client: AsyncClient) -> None:
    """非法转移（如 IDLE 直接 sleep）→ 409。"""
    await _register_and_get_state(client)
    # IDLE → SLEEPING 非法（必须先 ACTIVE）
    r = await client.post("/api/models/qwen-7b-v1/sleep")
    assert r.status_code == 409
    assert "状态转移失败" in r.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# 删除
# ─────────────────────────────────────────────────────────────────────────────


async def test_delete_idle_model_success(client: AsyncClient) -> None:
    await _register_and_get_state(client)
    r = await client.delete("/api/models/qwen-7b-v1")
    assert r.status_code == 204
    # 再 GET 应 404
    assert (await client.get("/api/models/qwen-7b-v1")).status_code == 404


async def test_delete_active_model_returns_409(client: AsyncClient) -> None:
    """非 IDLE 态不能删。"""
    await _register_and_get_state(client)
    await client.post("/api/models/qwen-7b-v1/load")
    r = await client.delete("/api/models/qwen-7b-v1")
    assert r.status_code == 409


async def test_delete_nonexistent_returns_404(client: AsyncClient) -> None:
    r = await client.delete("/api/models/nope")
    assert r.status_code == 404
