"""Pydantic request/response 模型（对应 SPEC §8.3）。

严格按 SPEC 的 schema 定义。所有 datetime 用 UTC ISO 格式。
枚举字段用 str 类型（避免 pydantic v2 对自定义 enum 的边界问题）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# 模型管理（SPEC §8.3）
# ─────────────────────────────────────────────────────────────────────────────


class ModelRegisterRequest(BaseModel):
    """POST /api/models/register 请求体（SPEC §8.3）。"""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(..., examples=["qwen-7b"])
    version: str = Field(..., examples=["v2"])
    endpoint: str = Field(..., examples=["http://vllm-v2:8001/v1"])
    params_billion: float = Field(..., gt=0, examples=[7.0])
    dtype: str = Field(default="fp16", examples=["fp16", "bf16", "int4"])
    quantization: str | None = Field(default=None, examples=["gptq", "awq", None])
    gpu_id: int = Field(default=0, ge=0)

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        allowed = {"fp32", "fp16", "bf16", "int8", "int4"}
        if v not in allowed:
            msg = f"dtype 必须是 {allowed}，得到 {v!r}"
            raise ValueError(msg)
        return v

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str | None) -> str | None:
        if v is None:
            return None
        allowed = {"gptq", "awq", "gguf"}
        if v not in allowed:
            msg = f"quantization 必须是 {allowed} 或 None，得到 {v!r}"
            raise ValueError(msg)
        return v

    @property
    def derived_id(self) -> str:
        """按 SPEC §9.2 推导的 model_id：{model_name}-{version}。"""
        return f"{self.model_name}-{self.version}"


class ModelResponse(BaseModel):
    """GET /api/models/{id} 响应体。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    model_name: str
    version: str
    endpoint: str
    params_billion: float
    dtype: str
    quantization: str | None
    weight_gb: float | None
    kv_cache_budget_gb: float | None
    gpu_id: int | None
    state: str  # ModelState 字符串值
    pending_requests: int
    total_requests_served: int
    total_errors: int
    state_changed_at: datetime
    created_at: datetime


class ModelListResponse(BaseModel):
    """GET /api/models 响应体。"""

    models: list[ModelResponse]
    total: int


# ─────────────────────────────────────────────────────────────────────────────
# 灰度管理（SPEC §8.3）
# ─────────────────────────────────────────────────────────────────────────────


class CanaryStartRequest(BaseModel):
    """POST /api/canary/start 请求体（SPEC §8.3）。"""

    model_config = ConfigDict(extra="forbid")

    model_v1: str = Field(..., examples=["qwen-7b-v1"])
    model_v2: str = Field(..., examples=["qwen-7b-v2"])
    strategy: str = Field(default="gradual", examples=["gradual"])
    stages: list[float] = Field(default=[0.10, 0.30, 1.0])
    min_duration_per_stage_seconds: int = Field(default=300, ge=0)

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, v: list[float]) -> list[float]:
        if not v:
            msg = "stages 不能为空"
            raise ValueError(msg)
        if not all(0.0 < s <= 1.0 for s in v):
            msg = "每个 stage 必须在 (0, 1]"
            raise ValueError(msg)
        if v != sorted(v):
            msg = "stages 必须递增"
            raise ValueError(msg)
        if v[-1] != 1.0:
            msg = "最后一个 stage 必须是 1.0（全量）"
            raise ValueError(msg)
        return v


class CanaryResponse(BaseModel):
    """POST /api/canary/start 响应体（SPEC §8.3）。"""

    id: str
    current_stage: str
    traffic_split: dict[str, float]


class CanaryStatusResponse(BaseModel):
    """GET /api/canary/{id}/status 响应体（SPEC §8.3）。"""

    id: str
    current_stage: str
    traffic_split: dict[str, float]
    stage_started_at: datetime | None = None
    metrics: dict[str, Any] | None = None
    eval: dict[str, Any] | None = None
    can_advance: bool = False
    can_rollback: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# 评测（SPEC §8.3，P4 用）
# ─────────────────────────────────────────────────────────────────────────────


class EvalCompareRequest(BaseModel):
    """POST /api/eval/compare 请求体（P4 实现）。"""

    model_config = ConfigDict(extra="forbid")

    model_v1: str
    model_v2: str
    prompts: list[str] = Field(..., min_length=1)
    judge_model: str | None = None


class EvalReportResponse(BaseModel):
    """GET /api/eval/{id}/report 响应体（P4 实现）。"""

    id: str
    sample_count: int
    score_v1_mean: float
    score_v2_mean: float
    t_statistic: float | None
    p_value: float | None
    significant: bool | None
    effect_size: float | None
    recommendation: str | None
    dimension_scores: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ─────────────────────────────────────────────────────────────────────────────
# 通用
# ─────────────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """统一的错误响应。"""

    detail: str
    error_type: str | None = None
