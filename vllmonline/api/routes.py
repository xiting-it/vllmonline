"""模型管理 REST API（SPEC §8.2 模型管理部分）。

P2 实现：
    POST   /api/models/register
    POST   /api/models/{id}/load
    POST   /api/models/{id}/unload
    POST   /api/models/{id}/sleep
    POST   /api/models/{id}/wake
    GET    /api/models
    GET    /api/models/{id}
    DELETE /api/models/{id}

P3/P4 的灰度/评测路由在各自 Phase 挂载。
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from vllmonline.api.schemas import (
    ModelListResponse,
    ModelRegisterRequest,
    ModelResponse,
)
from vllmonline.db.models import ModelVersion
from vllmonline.db.session import get_session
from vllmonline.scheduler.gpu_memory import build_model_profile
from vllmonline.scheduler.lifecycle import Model, ModelRegistry
from vllmonline.scheduler.types import ModelState

logger = structlog.get_logger("vllmonline.api.routes")


# ─────────────────────────────────────────────────────────────────────────────
# Registry 访问
# ─────────────────────────────────────────────────────────────────────────────

# 全局 ModelRegistry（lifespan 启动时设）
_global_registry: ModelRegistry | None = None


def set_global_registry(registry: ModelRegistry) -> None:
    """lifespan 启动时调用。"""
    global _global_registry
    _global_registry = registry


def get_registry() -> ModelRegistry:
    if _global_registry is None:
        msg = "ModelRegistry 未初始化"
        raise RuntimeError(msg)
    return _global_registry


# ─────────────────────────────────────────────────────────────────────────────
# 持久化辅助
# ─────────────────────────────────────────────────────────────────────────────


def _model_to_response(model: Model) -> ModelResponse:
    """把内存 Model 转成 API 响应。"""
    return ModelResponse(
        id=model.id,
        model_name=model.model_name,
        version=model.version,
        endpoint=model.endpoint,
        params_billion=model.memory_profile.params_billion,
        dtype=model.memory_profile.dtype,
        quantization=model.memory_profile.quantization,
        weight_gb=model.memory_profile.weight_gb,
        kv_cache_budget_gb=model.memory_profile.kv_cache_budget_gb,
        gpu_id=model.gpu_id,
        state=model.state.value,
        pending_requests=model.pending_requests,
        total_requests_served=model.total_requests_served,
        total_errors=model.total_errors,
        state_changed_at=model.state_changed_at,
        created_at=model.state_changed_at,  # Model 没存 created_at，用 state_changed 近似
    )


def _dbrow_to_response(row: ModelVersion) -> ModelResponse:
    """把 DB 行转成 API 响应。"""
    return ModelResponse(
        id=row.id,
        model_name=row.model_name,
        version=row.version,
        endpoint=row.endpoint,
        params_billion=row.params_billion,
        dtype=row.dtype,
        quantization=row.quantization,
        weight_gb=row.weight_gb,
        kv_cache_budget_gb=row.kv_cache_budget_gb,
        gpu_id=row.gpu_id,
        state=row.status,
        pending_requests=row.pending_requests,
        total_requests_served=row.total_requests_served,
        total_errors=row.total_errors,
        state_changed_at=row.state_changed_at,
        created_at=row.created_at,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 路由
# ─────────────────────────────────────────────────────────────────────────────


def create_model_router() -> APIRouter:
    """构造模型管理路由。"""
    router = APIRouter(prefix="/api/models", tags=["models"])

    @router.post(
        "/register",
        response_model=ModelResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def register_model(
        req: ModelRegisterRequest,
        session: AsyncSession = Depends(get_session),
    ) -> ModelResponse:
        """注册新模型版本（SPEC §8.3）。

        同时写入 DB 和内存 ModelRegistry。
        计算权重/KV cache 显存并存储。
        """
        model_id = req.derived_id
        profile = build_model_profile(req.params_billion, req.dtype, req.quantization)

        # 写 DB
        row = ModelVersion(
            id=model_id,
            model_name=req.model_name,
            version=req.version,
            endpoint=req.endpoint,
            params_billion=req.params_billion,
            dtype=req.dtype,
            quantization=req.quantization,
            weight_gb=profile.weight_gb,
            kv_cache_budget_gb=profile.kv_cache_budget_gb,
            gpu_id=req.gpu_id,
            status=ModelState.IDLE.value,
        )
        session.add(row)
        try:
            await session.commit()
        except IntegrityError as e:
            await session.rollback()
            msg = f"模型 {model_id} 已存在"
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=msg) from e

        # 写内存 Registry
        registry = get_registry()
        model = Model(
            id=model_id,
            model_name=req.model_name,
            version=req.version,
            endpoint=req.endpoint,
            gpu_id=req.gpu_id,
            memory_profile=profile,
        )
        try:
            await registry.register(model)
        except ValueError as e:
            # DB 已写但内存冲突——记录但不回滚 DB（罕见）
            logger.warning("registry conflict after DB insert", model=model_id, error=str(e))

        await session.refresh(row)
        logger.info(
            "model registered",
            id=model_id,
            weight_gb=profile.weight_gb,
            kv_gb=profile.kv_cache_budget_gb,
        )
        return _dbrow_to_response(row)

    @router.get("", response_model=ModelListResponse)
    async def list_models(
        session: AsyncSession = Depends(get_session),
        model_name: str | None = None,
    ) -> ModelListResponse:
        """列出所有模型（可按 model_name 过滤）。"""
        stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc())
        if model_name:
            stmt = stmt.where(ModelVersion.model_name == model_name)
        result = await session.execute(stmt)
        rows = list(result.scalars().all())
        return ModelListResponse(models=[_dbrow_to_response(r) for r in rows], total=len(rows))

    @router.get("/{model_id}", response_model=ModelResponse)
    async def get_model(
        model_id: str,
        session: AsyncSession = Depends(get_session),
    ) -> ModelResponse:
        """获取单个模型详情。"""
        row = await session.get(ModelVersion, model_id)
        if row is None:
            msg = f"模型 {model_id} 不存在"
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
        return _dbrow_to_response(row)

    @router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_model(
        model_id: str,
        session: AsyncSession = Depends(get_session),
    ) -> None:
        """删除模型（必须为 IDLE 终态）。"""
        row = await session.get(ModelVersion, model_id)
        if row is None:
            msg = f"模型 {model_id} 不存在"
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
        if row.status != ModelState.IDLE.value:
            msg = f"模型 {model_id} 必须是 IDLE 态才能删除（当前 {row.status}）"
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=msg)
        await session.delete(row)
        await session.commit()

        # 同步删除内存 registry
        registry = get_registry()
        if model_id in registry:
            try:
                await registry.unregister(model_id)
            except (ValueError, KeyError) as e:
                logger.warning("registry unregister failed", model=model_id, error=str(e))

    # ── 状态机操作 ──

    @router.post("/{model_id}/load", response_model=ModelResponse)
    async def load_model(
        model_id: str,
        session: AsyncSession = Depends(get_session),
    ) -> ModelResponse:
        """加载模型到 GPU：IDLE → LOADING → ACTIVE。"""
        return await _do_transition(model_id, [ModelState.LOADING, ModelState.ACTIVE], session)

    @router.post("/{model_id}/unload", response_model=ModelResponse)
    async def unload_model(
        model_id: str,
        session: AsyncSession = Depends(get_session),
    ) -> ModelResponse:
        """卸载模型：ACTIVE → DRAINING → SLEEPING → UNLOADING → IDLE。"""
        return await _do_transition(
            model_id,
            [ModelState.DRAINING, ModelState.SLEEPING, ModelState.UNLOADING, ModelState.IDLE],
            session,
        )

    @router.post("/{model_id}/sleep", response_model=ModelResponse)
    async def sleep_model(
        model_id: str,
        session: AsyncSession = Depends(get_session),
    ) -> ModelResponse:
        """休眠模型：ACTIVE → SLEEPING。"""
        return await _do_transition(model_id, [ModelState.SLEEPING], session)

    @router.post("/{model_id}/wake", response_model=ModelResponse)
    async def wake_model(
        model_id: str,
        session: AsyncSession = Depends(get_session),
    ) -> ModelResponse:
        """唤醒模型：SLEEPING → LOADING → ACTIVE。"""
        return await _do_transition(model_id, [ModelState.LOADING, ModelState.ACTIVE], session)

    return router


async def _do_transition(
    model_id: str,
    path: list[ModelState],
    session: AsyncSession,
) -> ModelResponse:
    """执行一组状态转移，同步到 DB 和 registry。

    验证：模型必须存在，且起始状态与 path[0] 的前驱匹配。
    """
    registry = get_registry()
    if model_id not in registry:
        msg = f"模型 {model_id} 不存在"
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
    model = registry.get(model_id)

    try:
        for target in path:
            await model.transition(target)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"状态转移失败（当前 {model.state.value}）：{e}",
        ) from e

    # 同步到 DB
    row = await session.get(ModelVersion, model_id)
    if row is not None:
        row.status = model.state.value
        row.state_changed_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(row)
        return _dbrow_to_response(row)
    # DB 没有就只返回内存态
    return _model_to_response(model)


# ─────────────────────────────────────────────────────────────────────────────
# 注册函数（server.create_app 调用）
# ─────────────────────────────────────────────────────────────────────────────


def register_model_routes(app: object) -> None:
    """把模型管理路由挂到 FastAPI app 上。"""
    router = create_model_router()
    app.include_router(router)  # type: ignore[attr-defined]
