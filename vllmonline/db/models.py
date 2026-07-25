"""SQLAlchemy 2.0 ORM 模型（严格对应 SPEC §9.1）。

4 张表：
    model_versions      模型版本（核心表）
    canary_deployments  灰度部署
    canary_events       灰度事件日志
    eval_results        A/B 评测结果

设计原则（SPEC §9.2）：
    - model_versions.id 格式 {model_name}-{version}，如 qwen-7b-v2
    - JSON 字段（PG 是 JSONB，SQLite 自动降级为 TEXT）用于灵活结构化数据
    - 所有时间戳 UTC
    - status 字段存字符串枚举值（Python 侧用 ModelState）

SQLite 兼容：ForeignKey 在 SQLite 下不强制约束（PRAGMA foreign_keys=OFF 默认），
对测试无影响；PG 生产环境生效。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。type_annotation_map 处理自定义类型。"""


def _utcnow() -> datetime:
    """datetime 默认值工厂（避免 NOW() 在 SQLite 行为不一致）。"""
    return datetime.now(UTC)


class ModelVersion(Base):
    """模型版本（核心表，SPEC §9.1）。

    一行 = 一个注册的模型版本（如 qwen-7b-v2）。
    status 字段镜像内存中 Model.state（崩溃恢复用）。
    """

    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(512), nullable=False)
    params_billion: Mapped[float] = mapped_column(Float, nullable=False)
    dtype: Mapped[str] = mapped_column(String(16), nullable=False, default="fp16")
    quantization: Mapped[str | None] = mapped_column(String(32), nullable=True)
    weight_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    kv_cache_budget_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="IDLE")
    pending_requests: Mapped[int] = mapped_column(Integer, default=0)
    total_requests_served: Mapped[int] = mapped_column(BigInteger, default=0)
    total_errors: Mapped[int] = mapped_column(BigInteger, default=0)

    state_changed_at: Mapped[datetime] = mapped_column(default=_utcnow)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)

    # 关系（P3 灰度用）
    as_v1_deployments: Mapped[list[CanaryDeployment]] = relationship(
        back_populates="model_v1", foreign_keys="CanaryDeployment.model_v1_id"
    )
    as_v2_deployments: Mapped[list[CanaryDeployment]] = relationship(
        back_populates="model_v2", foreign_keys="CanaryDeployment.model_v2_id"
    )

    __table_args__ = (
        Index("ix_model_versions_status", "status"),
        Index("ix_model_versions_model_name", "model_name"),
    )


class CanaryDeployment(Base):
    """灰度部署（SPEC §9.1）。

    一行 = 一次灰度发布（v1 → v2 的渐进切换）。
    """

    __tablename__ = "canary_deployments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_v1_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_versions.id"), nullable=False
    )
    model_v2_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_versions.id"), nullable=False
    )
    strategy: Mapped[str] = mapped_column(String(32), default="gradual")
    stages: Mapped[list[float]] = mapped_column(JSON, default=lambda: [0.1, 0.3, 1.0])
    current_stage_index: Mapped[int] = mapped_column(Integer, default=0)
    traffic_split: Mapped[dict[str, float]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="INIT")
    started_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)

    model_v1: Mapped[ModelVersion] = relationship(
        back_populates="as_v1_deployments", foreign_keys=[model_v1_id]
    )
    model_v2: Mapped[ModelVersion] = relationship(
        back_populates="as_v2_deployments", foreign_keys=[model_v2_id]
    )
    events: Mapped[list[CanaryEvent]] = relationship(
        back_populates="deployment", cascade="all, delete-orphan"
    )
    eval_results: Mapped[list[EvalResult]] = relationship(
        back_populates="deployment", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_canary_deployments_status", "status"),)


class CanaryEvent(Base):
    """灰度事件日志（SPEC §9.1）。

    每次 ADVANCE/ROLLBACK/HOLD 都记一行，含当时的 metrics 快照。
    """

    __tablename__ = "canary_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deployment_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("canary_deployments.id"), nullable=False
    )
    stage_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(16), nullable=False)  # ADVANCE|ROLLBACK|HOLD
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    deployment: Mapped[CanaryDeployment] = relationship(back_populates="events")

    __table_args__ = (Index("ix_canary_events_deployment", "deployment_id"),)


class EvalResult(Base):
    """A/B 评测结果（SPEC §9.1）。"""

    __tablename__ = "eval_results"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    deployment_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("canary_deployments.id"), nullable=True
    )
    sample_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_v1_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_v2_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    t_statistic: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    significant: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    effect_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    dimension_scores: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    recommendation: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    deployment: Mapped[CanaryDeployment | None] = relationship(back_populates="eval_results")

    __table_args__ = (Index("ix_eval_results_deployment", "deployment_id"),)


__all__ = [
    "Base",
    "CanaryDeployment",
    "CanaryEvent",
    "EvalResult",
    "ModelVersion",
]
