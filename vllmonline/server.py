"""FastAPI 主入口。

Phase 0 最简实现：
    - POST /v1/chat/completions：透传到默认 vLLM backend（暂不做路由分流，留待 P3）
    - GET /healthz：存活探针
    - GET /readyz：就绪探针（检查 vLLM backend 连通性）
    - GET /metrics：Prometheus metrics 端点
    - GET /：服务信息

后续 Phase 会在此基础上挂载：
    - P2: /api/models/* 模型管理
    - P3: /api/canary/* 灰度管理 + 路由表 + per-version metrics 中间件
    - P4: /api/eval/* 评测

启动方式：
    开发：`make run`（uvicorn --reload）
    生产：`uvicorn vllmonline.server:app --host 0.0.0.0 --port 8080 --workers 4`
    Docker：`docker compose up`
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from vllmonline.config import Settings, get_settings
from vllmonline.metrics import REGISTRY, init_metrics, record_proxy_request
from vllmonline.version import __version__

# hop-by-hop headers 不应被代理转发（RFC 7230 §6.1）
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# 日志初始化
# ─────────────────────────────────────────────────────────────────────────────


def _configure_logging(settings: Settings) -> structlog.BoundLogger:
    """配置 structlog。生产用 JSON，开发用 console。"""
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.logging.level.upper(), logging.INFO),
    )
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if settings.logging.json_format or settings.environment == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    structlog.configure(processors=processors)
    logger: structlog.BoundLogger = structlog.get_logger(settings.logging.service_name)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# 应用级共享状态
# ─────────────────────────────────────────────────────────────────────────────


class AppState:
    """挂载在 app.state.vllmonline 上的共享对象。

    保持类型清晰，避免到处访问 request.app.state.xxx 的弱类型模式。
    """

    def __init__(self) -> None:
        self.settings: Settings
        self.http_client: httpx.AsyncClient
        self.logger: structlog.BoundLogger
        # 默认 vLLM backend（Phase 0 用；P3 起改为路由表查询）
        self.default_backend_url: str
        # P2 起：模型注册表
        from vllmonline.scheduler.lifecycle import ModelRegistry

        self.registry: ModelRegistry

    @classmethod
    def from_app(cls, app: FastAPI) -> AppState:
        state: AppState = app.state.vllmonline
        return state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时初始化资源，关闭时清理。"""
    settings = get_settings()
    logger = _configure_logging(settings)

    init_metrics(settings)

    # Phase 0：默认 backend 写死为环境变量或 localhost:8000。
    # 后续 P2 起从 ModelRegistry 动态查询。
    default_backend = os.environ.get("VLLM_BACKEND_URL", "http://localhost:8000").rstrip("/")

    state = AppState()
    state.settings = settings
    state.logger = logger
    state.default_backend_url = default_backend

    # 测试 hook：如果设置了 _test_client_factory，用它构造 http_client
    # （指向 fake vLLM via ASGITransport，零网络）。生产路径不触发。
    if _test_client_factory is not None:
        http_client = _test_client_factory()
        # 测试 fixture 可能同时想覆盖 backend URL
        if _test_backend_url is not None:
            state.default_backend_url = _test_backend_url
    else:
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=settings.vllm.connect_timeout_seconds,
                read=settings.vllm.read_timeout_seconds,
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )
    state.http_client = http_client
    app.state.vllmonline = state

    # 初始化 DB engine + ModelRegistry（P2 起）
    from vllmonline.api import routes as api_routes
    from vllmonline.db import session as db_session
    from vllmonline.scheduler.lifecycle import ModelRegistry

    db_engine = db_session.create_engine(settings)
    db_session.set_global_engine(db_engine)
    # 开发/测试：直接建表（生产走 alembic upgrade head）
    if settings.database.is_sqlite or settings.environment != "production":
        await db_session.init_db(db_engine)

    registry = ModelRegistry()
    api_routes.set_global_registry(registry)
    state.registry = registry

    logger.info(
        "vllmonline starting",
        version=__version__,
        environment=settings.environment,
        backend=state.default_backend_url,
        gpu_backend=settings.gpu.backend.value,
        db_url=settings.database.url.split("@")[-1]
        if "@" in settings.database.url
        else settings.database.url,
    )

    try:
        yield
    finally:
        await http_client.aclose()
        db_session.clear_global_engine()
        await db_engine.dispose()
        logger.info("vllmonline stopped")


# 测试 hook：单元测试通过 set_test_backend() 注入 fake vLLM。
# 生产代码路径里这两个变量永远为 None。
_test_client_factory: Any = None  # 类型：Callable[[], httpx.AsyncClient] | None
_test_backend_url: str | None = None


def set_test_backend(
    client_factory: Any,
    backend_url: str,
) -> None:
    """测试专用：注入 fake vLLM client 工厂。生产代码不要调用。"""
    global _test_client_factory, _test_backend_url
    _test_client_factory = client_factory
    _test_backend_url = backend_url


def clear_test_backend() -> None:
    """测试结束清理。"""
    global _test_client_factory, _test_backend_url
    _test_client_factory = None
    _test_backend_url = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app 工厂
# ─────────────────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """构造 FastAPI 应用（含全部路由注册）。

    测试和 uvicorn 入口都通过本函数拿 app，确保路由完整。
    """
    app = FastAPI(
        title="vLLMonline",
        description=(
            "vLLM 推理引擎上层管理平台：零停机模型热切换 + 灰度发布 + A/B 自动评测 + 劣化自动回滚。"
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 注册 Phase 0 路由
    app.add_api_route("/", root, methods=["GET"], tags=["meta"])
    app.add_api_route("/healthz", healthz, methods=["GET"], tags=["meta"])
    app.add_api_route("/readyz", readyz, methods=["GET"], tags=["meta"])
    app.add_api_route("/metrics", metrics_endpoint, methods=["GET"], tags=["meta"])
    app.add_api_route(
        "/v1/chat/completions",
        proxy_chat_completion,
        methods=["POST"],
        tags=["proxy"],
        name="chat_completions",
    )

    # Phase 2 起的业务路由
    from vllmonline.api.routes import register_model_routes

    register_model_routes(app)
    # P3: register_canary_routes(app)
    # P4: register_eval_routes(app)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# 依赖注入
# ─────────────────────────────────────────────────────────────────────────────


def get_app_state(request: Request) -> AppState:
    return AppState.from_app(request.app)


# ─────────────────────────────────────────────────────────────────────────────
# 路由处理函数
# ─────────────────────────────────────────────────────────────────────────────


async def root() -> dict[str, str]:
    """GET / —— 服务信息。"""
    return {"service": "vllmonline", "version": __version__, "docs": "/docs"}


async def healthz() -> dict[str, str]:
    """GET /healthz —— 存活探针。只要进程在跑就返回 ok。"""
    return {"status": "ok"}


async def readyz(state: AppState = Depends(get_app_state)) -> JSONResponse:
    """GET /readyz —— 就绪探针。检查 vLLM backend 是否可达。"""
    checks: dict[str, str] = {}
    overall_ok = True

    try:
        resp = await state.http_client.get(
            f"{state.default_backend_url}/health",
            timeout=2.0,
        )
        if resp.is_success:
            checks["vllm"] = "ok"
        else:
            checks["vllm"] = f"unhealthy (http {resp.status_code})"
            overall_ok = False
    except httpx.HTTPError as e:
        checks["vllm"] = f"unreachable: {type(e).__name__}"
        overall_ok = False

    code = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        content={"status": "ok" if overall_ok else "not_ready", "checks": checks},
        status_code=code,
    )


async def metrics_endpoint() -> PlainTextResponse:
    """GET /metrics —— Prometheus 抓取端点。"""
    return PlainTextResponse(
        content=generate_latest(REGISTRY).decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


async def proxy_chat_completion(
    request: Request,
    state: AppState = Depends(get_app_state),
) -> Response:
    """POST /v1/chat/completions —— OpenAI 兼容代理。

    Phase 0 行为：透传请求体到默认 vLLM backend，streaming 逐 chunk 透传。
    不修改请求体，不注入额外 header（P3 起加 x-model-version）。
    """
    body = await request.body()

    # 检测是否 streaming 请求（保守：解析失败按非 streaming 处理）
    try:
        payload = json.loads(body) if body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {}
    is_stream = bool(payload.get("stream", False))

    # 透传 header（去掉 hop-by-hop）
    forwarded_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS
    }

    backend_url = f"{state.default_backend_url}/v1/chat/completions"
    client = state.http_client

    if is_stream:
        return await _proxy_stream(client, backend_url, body, forwarded_headers, state)
    return await _proxy_nonstream(client, backend_url, body, forwarded_headers, state)


async def _proxy_nonstream(
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    headers: dict[str, str],
    state: AppState,
) -> JSONResponse:
    """非 streaming：等完整响应后返回。"""
    try:
        resp = await client.post(url, content=body, headers=headers)
    except httpx.HTTPError as e:
        state.logger.error("proxy upstream error", url=url, error=str(e))
        record_proxy_request(status="error", error_type="upstream_unreachable")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"upstream vLLM unreachable: {e}",
        ) from e

    record_proxy_request(
        status="success" if resp.is_success else "error",
        error_type=None if resp.is_success else f"http_{resp.status_code}",
    )
    return JSONResponse(
        content=resp.json() if resp.content else {},
        status_code=resp.status_code,
    )


async def _proxy_stream(
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    headers: dict[str, str],
    state: AppState,
) -> StreamingResponse:
    """streaming：逐 chunk 透传（保持 SSE 不变形）。"""
    headers = {**headers, "accept": "text/event-stream"}

    async def chunk_generator() -> AsyncIterator[bytes]:
        ok = True
        try:
            async with client.stream("POST", url, content=body, headers=headers) as upstream:
                async for chunk in upstream.aiter_raw():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as e:
            ok = False
            state.logger.error("proxy stream error", url=url, error=str(e))
        finally:
            record_proxy_request(
                status="success" if ok else "error",
                error_type=None if ok else "stream_error",
            )

    return StreamingResponse(chunk_generator(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# 模块级 app（uvicorn 入口：vllmonline.server:app）
# ─────────────────────────────────────────────────────────────────────────────

app = create_app()


def main() -> None:
    """命令行入口（python -m vllmonline）。"""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "vllmonline.server:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.environment == "development",
        log_level=settings.logging.level.lower(),
    )


if __name__ == "__main__":
    main()
