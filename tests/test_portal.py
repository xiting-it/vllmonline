"""门户相关测试：列表 API + 静态页可访问。

覆盖：
    - GET /api/canary（空列表 + 有数据）
    - GET /api/eval（空列表 + 有数据）
    - GET /portal 返回 HTML
    - 静态资源（app.js / style.css / chart.umd.js）可访问
"""

from __future__ import annotations

from httpx import AsyncClient

# ─────────────────────────────────────────────────────────────────────────────
# GET /api/canary 列表
# ─────────────────────────────────────────────────────────────────────────────


async def test_list_canaries_empty(client: AsyncClient) -> None:
    r = await client.get("/api/canary")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["deployments"] == []


async def test_list_canaries_after_start(client: AsyncClient) -> None:
    """注册两个模型 + 起灰度后，列表应包含该部署。"""
    for ver, port in [("v1", 8000), ("v2", 8001)]:
        await client.post(
            "/api/models/register",
            json={
                "model_name": "qwen-7b",
                "version": ver,
                "endpoint": f"http://localhost:{port}",
                "params_billion": 7.0,
                "dtype": "fp16",
            },
        )
        await client.post(f"/api/models/qwen-7b-{ver}/load")

    start = await client.post(
        "/api/canary/start",
        json={"model_v1": "qwen-7b-v1", "model_v2": "qwen-7b-v2", "stages": [0.1, 0.3, 1.0]},
    )
    assert start.status_code == 201

    r = await client.get("/api/canary")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    dep = body["deployments"][0]
    assert dep["id"].startswith("canary-")
    assert dep["status"] == "IN_PROGRESS"
    assert dep["model_v1_id"] == "qwen-7b-v1"
    assert dep["model_v2_id"] == "qwen-7b-v2"
    assert dep["traffic_split"]["qwen-7b-v2"] > 0
    assert dep["started_at"] is not None

    # 按开始时间倒序：最新的在前
    if body["total"] > 1:
        assert body["deployments"][0]["started_at"] >= body["deployments"][1]["started_at"]


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/eval 列表
# ─────────────────────────────────────────────────────────────────────────────


async def test_list_evals_empty(client: AsyncClient) -> None:
    r = await client.get("/api/eval")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["evals"] == []


async def test_list_evals_with_data(client: AsyncClient) -> None:
    """直接往 DB 插一条评测记录（绕过需要 LLM 的 compare 端点），列表应返回。"""
    # 拿一个 session——通过依赖注入的工厂写数据
    from vllmonline.db.session import _global_session_factory

    assert _global_session_factory is not None, "lifespan 应已初始化 session factory"
    from vllmonline.db.models import EvalResult

    async with _global_session_factory() as session:
        session.add(
            EvalResult(
                id="eval-test-001",
                sample_count=49,
                score_v1_mean=0.9207,
                score_v2_mean=0.8578,
                p_value=0.0001,
                significant=True,
                effect_size=-0.8315,
                recommendation="rollback",
                dimension_scores={
                    "v1": {"accuracy": 0.8765, "completeness": 0.8857, "safety": 1.0},
                    "v2": {"accuracy": 0.7816, "completeness": 0.7918, "safety": 1.0},
                },
            )
        )
        await session.commit()

    r = await client.get("/api/eval")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    ev = body["evals"][0]
    assert ev["id"] == "eval-test-001"
    assert ev["sample_count"] == 49
    assert ev["p_value"] == 0.0001
    assert ev["significant"] is True
    assert ev["recommendation"] == "rollback"
    assert ev["dimension_scores"]["v2"]["accuracy"] == 0.7816
    assert ev["created_at"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# 静态门户
# ─────────────────────────────────────────────────────────────────────────────


async def test_portal_serves_html(client: AsyncClient) -> None:
    """/portal 应返回 index.html（307 重定向到 /portal/，浏览器自动跟随）。"""
    r = await client.get("/portal", follow_redirects=True)
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "vLLMonline" in r.text
    assert "灰度控制台" in r.text  # 核心面板存在


async def test_portal_static_assets(client: AsyncClient) -> None:
    """门户的静态资源（JS/CSS/Chart.js）可访问。"""
    for path, marker in [
        ("/portal/app.js", "refreshCanary"),
        ("/portal/style.css", "--bg"),
        ("/portal/chart.umd.js", "Chart"),
    ]:
        r = await client.get(path)
        assert r.status_code == 200, f"{path} 应可访问，实际 {r.status_code}"
        assert marker in r.text, f"{path} 内容异常"
