"""配置管理：pydantic-settings 双源（YAML 文件 + 环境变量）。

加载顺序（后者覆盖前者）：
    1. 代码内默认值（下方 DEFAULT_* 常量）
    2. YAML 文件（path 由 VLLMONLINE_CONFIG 环境变量指定，默认 config.yaml）
    3. 环境变量（前缀 VLLMONLINE_，嵌套用 __ 分隔）

所有可配置项都有默认值——开箱即用。生产部署通过环境变量或 YAML 覆盖。

MI300X 部署注意事项：
    - GPU backend 自动从 rocm-smi / nvidia-smi 可用性探测，也可显式 config.gpu_backend
    - cuda_context_overhead_gb 在 MI300X 上推荐 4.0-5.0（SPEC 默认 2.0 适用于 NVIDIA）
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GpuBackend(str, Enum):
    """GPU 厂商栈。决定如何查询显存（rocm-smi vs nvidia-smi）。"""

    AUTO = "auto"  # 自动探测
    ROCM = "rocm"  # AMD ROCm（MI300X 等）
    NVIDIA = "nvidia"  # NVIDIA CUDA
    NONE = "none"  # 无 GPU（开发/测试模式，所有查询返回 0）


# ─────────────────────────────────────────────────────────────────────────────
# 配置子模型
# ─────────────────────────────────────────────────────────────────────────────


class ServerConfig(BaseModel):
    """HTTP 服务监听配置。"""

    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1  # 生产用 gunicorn/uvicorn 多 worker，开发用 1
    request_timeout_seconds: float = 120.0  # 客户端请求最长等待


class DatabaseConfig(BaseModel):
    """数据库连接。

    - 生产：PostgreSQL（asyncpg）
    - 开发/单测：SQLite（aiosqlite，in-memory 或文件）

    通过 url scheme 自动判断 driver。例：
        postgresql+asyncpg://...  -> PostgreSQL
        sqlite+aiosqlite:///./test.db  -> SQLite
    """

    url: str = "sqlite+aiosqlite:///:memory:"
    echo: bool = False  # 打印 SQL（调试用）
    pool_size: int = 10
    max_overflow: int = 20
    pool_pre_ping: bool = True  # 连接池借出前 ping（防断连）

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return "postgres" in self.url


class RedisConfig(BaseModel):
    """Redis 连接（路由表热更新 + metrics 缓存）。"""

    url: str = "redis://localhost:6379/0"
    enabled: bool = True  # 设 false 时降级为纯内存路由表（开发/测试）
    key_prefix: str = "vllmonline:"


class GpuConfig(BaseModel):
    """GPU 相关配置。

    cuda_context_overhead_gb：GPU 驱动/runtime 常驻显存开销（GB）。
        - SPEC 默认 2.0（适用于 NVIDIA A100/H100）
        - MI300X 推荐 4.0-5.0（HBM stack + ROCm runtime 更大）
        本字段名保留 "cuda" 是为与 SPEC §3 措辞一致，实际对 ROCm 同样适用。
    """

    backend: GpuBackend = GpuBackend.AUTO
    cuda_context_overhead_gb: float = 2.0
    default_memory_utilization: float = 0.90  # vLLM 默认 gpu_memory_utilization

    @field_validator("cuda_context_overhead_gb")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            msg = "cuda_context_overhead_gb 不能为负"
            raise ValueError(msg)
        return v

    @field_validator("default_memory_utilization")
    @classmethod
    def _util_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            msg = "default_memory_utilization 必须在 (0, 1]"
            raise ValueError(msg)
        return v


class VllmConfig(BaseModel):
    """vLLM backend 客户端配置。"""

    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 120.0
    # sleep/wake 后的 warmup 请求等待秒数（SPEC §7.2 workaround）
    wake_warmup_timeout_seconds: float = 30.0
    # sleep 后等待显存释放的轮询间隔
    sleep_settle_poll_seconds: float = 1.0
    sleep_settle_max_wait_seconds: float = 10.0
    # tenacity 重试
    retry_max_attempts: int = 3
    retry_initial_wait_seconds: float = 0.5


class CanaryConfig(BaseModel):
    """灰度策略默认参数。"""

    default_stages: list[float] = Field(default=[0.10, 0.30, 1.0])
    min_duration_per_stage_seconds: int = 300
    min_sample_size: int = 64  # 默认按 d=0.5/power=0.8 计算
    rollback_check_interval_seconds: int = 10
    # 推进判定阈值（SPEC §5.5）
    ttft_p50_degradation_threshold: float = 0.20
    ttft_p99_degradation_threshold: float = 0.30
    tpot_mean_degradation_threshold: float = 0.15
    throughput_drop_threshold: float = 0.10
    error_rate_increase_threshold: float = 0.02


class EvalConfig(BaseModel):
    """A/B 评测配置。"""

    sampling_rate: float = 0.05  # 从流量中采样的比例
    judge_model_endpoint: str | None = None  # judge LLM 的端点；None 时复用某被测版本
    judge_timeout_seconds: float = 60.0
    significance_alpha: float = 0.05  # Welch's t-test p 值阈值
    min_effect_size: float = 0.2  # Cohen's d 最小可接受效应量
    target_effect_size: float = 0.3  # 最小样本量计算用的目标效应量
    target_power: float = 0.80


class MetricsConfig(BaseModel):
    """Prometheus metrics 配置。"""

    namespace: str = ""  # metric 名前缀（空表示不加前缀）
    # TTFT histogram buckets（SPEC §5.4）
    ttft_buckets: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
    tpot_buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)


class RoutingConfig(BaseModel):
    """流量路由配置。"""

    routing_table_refresh_seconds: int = 5  # 从 DB 同步路由表的间隔


class LoggingConfig(BaseModel):
    """日志配置。"""

    level: str = "INFO"
    json_format: bool = False  # 生产用 true（结构化日志）
    service_name: str = "vllmonline"


# ─────────────────────────────────────────────────────────────────────────────
# 顶层 Settings
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = "config.yaml"
ENV_PREFIX = "VLLMONLINE_"
ENV_CONFIG_FILE_VAR = "VLLMONLINE_CONFIG"


class Settings(BaseSettings):
    """vLLMonline 全局配置。

    优先级：环境变量 > YAML 文件 > 类内默认值。
    嵌套字段在环境变量里用 `__` 分隔，例如：
        VLLMONLINE_DATABASE__URL=postgresql+asyncpg://...
        VLLMONLINE_GPU__CUDA_CONTEXT_OVERHEAD_GB=4.0
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # 顶层开关
    debug: bool = False
    environment: str = "development"  # development | staging | production

    # 子配置
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    vllm: VllmConfig = Field(default_factory=VllmConfig)
    canary: CanaryConfig = Field(default_factory=CanaryConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ─────────────────────────────────────────────────────────────────────────────
# 加载逻辑
# ─────────────────────────────────────────────────────────────────────────────


def _load_yaml_overrides(path: str | Path) -> dict[str, Any]:
    """读 YAML 配置文件。不存在或为空返回空 dict（不报错）。"""
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_settings(
    config_path: str | Path | None = None,
    *,
    env_override: dict[str, str] | None = None,
) -> Settings:
    """加载配置。

    Args:
        config_path: YAML 文件路径。None 时读 VLLMONLINE_CONFIG 环境变量，
            再回退到默认 config.yaml。
        env_override: 测试用——直接传入环境变量字典，避免污染 os.environ。

    Returns:
        Settings 实例。
    """
    if config_path is None:
        config_path = os.environ.get(ENV_CONFIG_FILE_VAR, DEFAULT_CONFIG_PATH)

    yaml_data = _load_yaml_overrides(config_path)

    # pydantic-settings 的环境变量扫描：如果传了 env_override，仅用那个；
    # 否则用真实 os.environ。Settings() 自动读 env_prefix + nested delimiter。
    if env_override is not None:
        # 临时设置环境变量（最干净的方式是构造时传 _env，但 BaseSettings
        # 通过 settings_customise_sources 支持复杂场景——这里用更简单的方式）
        return Settings.model_validate({**yaml_data, **_env_dict_to_nested(env_override)})

    # 真实路径：先实例化（读 env），再用 yaml 覆盖（yaml 优先级低于 env，
    # 所以反向：yaml 是默认值的扩展，env 仍可覆盖它）
    settings = Settings.model_validate(yaml_data)
    # 重新走一遍 env，让 env 覆盖 yaml
    return Settings(settings.__dict__ | _env_dict_to_nested(dict(os.environ)))


def _env_dict_to_nested(env: dict[str, str]) -> dict[str, Any]:
    """把扁平的环境变量（前缀 VLLMONLINE_，分隔符 __）转成嵌套 dict。

    例：VLLMONLINE_DATABASE__URL=x -> {"database": {"url": "x"}}
    非 VLLMONLINE_ 前缀的变量忽略。
    """
    result: dict[str, Any] = {}
    for key, value in env.items():
        if not key.startswith(ENV_PREFIX) or key == ENV_CONFIG_FILE_VAR:
            continue
        remainder = key[len(ENV_PREFIX) :].lower()
        parts = remainder.split("__")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局单例 Settings（FastAPI Depends 用）。

    首次调用走完整加载流程；之后命中缓存。
    测试里用 `get_settings.cache_clear()` 重置。
    """
    return load_settings()
