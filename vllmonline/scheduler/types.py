"""调度器核心类型定义。

严格遵循 SPEC §4.1（状态机）和 §3.4（LoadDecision）。
所有跨模块共享的数据结构都在这里，避免循环导入。

注意：本模块不可随意修改字段——它是整个系统的数据契约，
任何改动都会波及 gpu_memory / lifecycle / engine / api 多个模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# ─────────────────────────────────────────────────────────────────────────────
# 模型状态机（SPEC §4.1）
# ─────────────────────────────────────────────────────────────────────────────


class ModelState(StrEnum):
    """模型实例的生命周期状态。

    状态机完整定义见 SPEC §4.1。合法转移表见 lifecycle.TRANSITIONS。
    """

    IDLE = "IDLE"  # 已注册但未加载到 GPU
    LOADING = "LOADING"  # 正在加载到 GPU
    ACTIVE = "ACTIVE"  # 正常服务（接收请求）
    DRAINING = "DRAINING"  # 排空中：停止接新请求，等待在飞请求完成
    SLEEPING = "SLEEPING"  # 权重在 GPU，KV Cache 已释放
    UNLOADING = "UNLOADING"  # 正在从 GPU 卸载
    ERROR = "ERROR"  # 错误态（加载失败/OOM/健康检查连续失败）

    # ── 辅助判定（SPEC §4.4）──
    def can_serve(self) -> bool:
        """能否接受新请求。仅 ACTIVE 态可服务。"""
        return self is ModelState.ACTIVE

    def is_on_gpu(self) -> bool:
        """是否占用 GPU 显存。

        LOADING/ACTIVE/DRAINING 占完整显存（权重 + KV cache）。
        SLEEPING 只占权重（KV cache 已释放）。
        IDLE/ERROR 不占显存。
        """
        return self in {
            ModelState.LOADING,
            ModelState.ACTIVE,
            ModelState.DRAINING,
            ModelState.SLEEPING,
        }

    def is_terminal(self) -> bool:
        """是否是终态（不会自动转出）。

        IDLE 是"未加载"终态，ERROR 是"故障"终态。
        两者都需要外部 action（load / reset）才能离开。
        """
        return self in {ModelState.IDLE, ModelState.ERROR}


# ─────────────────────────────────────────────────────────────────────────────
# 加载策略（SPEC §3.4）
# ─────────────────────────────────────────────────────────────────────────────


class LoadStrategy(StrEnum):
    """can_load() 决策树的四种结果。

    优先级（从最便宜到最贵）：
        DIRECT < SLEEP_OLD < UNLOAD_OLD < INSUFFICIENT（不可行）
    """

    DIRECT = "DIRECT"  # 显存够，直接加载，不影响现有模型
    SLEEP_OLD = "SLEEP_OLD"  # 需要先 sleep 已加载模型释放 KV cache
    UNLOAD_OLD = "UNLOAD_OLD"  # 需要 unload 已加载模型释放全部显存
    INSUFFICIENT = "INSUFFICIENT"  # 即使空 GPU 也放不下（模型太大）


# ─────────────────────────────────────────────────────────────────────────────
# 显存数据结构
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ModelMemoryProfile:
    """模型的显存占用画像（计算值，不依赖运行时）。

    根据 SPEC §3.2-3.3 的公式，从模型元信息推导。
    不可变（frozen）——同一模型的画像在生命周期内不变。
    """

    params_billion: float  # 参数量（billions），如 7.0 表示 7B
    dtype: str  # fp32/fp16/bf16/int8/int4
    quantization: str | None  # gptq/awq/gguf 或 None
    weight_gb: float  # 权重显存（GB）
    kv_cache_budget_gb: float  # KV cache 预算（GB，按 utilization 估算）

    @property
    def total_gb(self) -> float:
        """满载时的总显存占用（权重 + KV cache）。"""
        return self.weight_gb + self.kv_cache_budget_gb

    @property
    def sleeping_gb(self) -> float:
        """sleep 态的显存占用（仅权重，KV cache 已释放）。"""
        return self.weight_gb


@dataclass(frozen=True, slots=True)
class GPUState:
    """某张 GPU 的当前显存状态快照。

    由 GpuInfoProvider 采集（rocm-smi / nvidia-smi 解析）。
    frozen：作为 can_load 输入时不应被修改。
    """

    gpu_id: int
    total_gb: float  # 物理 HBM/VRAM 总量
    used_gb: float  # 当前已用（含 CUDA/ROCm context + 其他模型）

    @property
    def free_gb(self) -> float:
        """剩余可用显存。"""
        return max(0.0, self.total_gb - self.used_gb)


@dataclass(frozen=True, slots=True)
class LoadDecision:
    """can_load() 的判定结果。

    根据 SPEC §3.4，必须包含 detail 字段——人类可读的决策解释，
    方便调试和日志（"为什么决定 sleep 旧模型"）。
    """

    strategy: LoadStrategy
    free_after_load_gb: float  # 加载后剩余显存（GB）
    required_gb: float  # 新模型需要的显存（GB）
    detail: str  # 人类可读的决策逻辑解释

    @property
    def can_proceed(self) -> bool:
        """加载是否可行（任何非 INSUFFICIENT 策略都可）。"""
        return self.strategy is not LoadStrategy.INSUFFICIENT


# ─────────────────────────────────────────────────────────────────────────────
# 路由表条目（P3 用，先定义避免循环）
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RouteEntry:
    """路由表里指向某个模型版本的条目。

    weight 是该版本在所属 model_name 的所有 ACTIVE 版本中的流量占比（0-1）。
    """

    model_id: str  # 如 "qwen-7b-v2"
    endpoint: str  # vLLM endpoint URL
    weight: float = 0.0  # 0-1，所有同 model_name 的 weight 之和应为 1.0


@dataclass(slots=True)
class RoutingTable:
    """流量路由表：model_name -> 多版本加权列表。

    P3 实现完整逻辑；此处仅定义数据结构。
    """

    routes: dict[str, list[RouteEntry]] = field(default_factory=dict)

    def add(self, model_name: str, entry: RouteEntry) -> None:
        self.routes.setdefault(model_name, []).append(entry)

    def get(self, model_name: str) -> list[RouteEntry]:
        return self.routes.get(model_name, [])
