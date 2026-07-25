"""GPU 显存计算（SPEC §3，最关键模块）。

⚠️ 本模块的所有计算必须精确。显存算错 1GB → 生产环境 OOM → 整个 GPU 上的
   所有模型全挂。任何公式修改都要相应更新测试。

公式来源（SPEC §3.2-3.3）：
    weight_gb    = params_billion × bytes_per_param × (1 + quant_overhead)
    kv_cache_gb  = weight_gb × 0.25 × utilization   （简化版，架构参数未知时）

can_load 决策树（SPEC §3.4）：
    DIRECT     显存够，直接加载
    SLEEP_OLD  显存不够，但 sleep 已加载模型释放 KV cache 后够
    UNLOAD_OLD 显存不够，但 unload 已加载模型释放全部显存后够
    INSUFFICIENT 即使空 GPU 也放不下
"""

from __future__ import annotations

from vllmonline.scheduler.types import GPUState, LoadDecision, LoadStrategy, ModelMemoryProfile

# ─────────────────────────────────────────────────────────────────────────────
# 常量表（SPEC §3.2，硬编码，不可随意修改）
# ─────────────────────────────────────────────────────────────────────────────

# dtype → 每参数字节数
DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# 量化方法 → 额外开销比（叠加在 bytes/param 之上）
QUANT_OVERHEAD: dict[str, float] = {
    "gptq": 0.05,
    "awq": 0.05,
    "gguf": 0.03,
}

# SPEC §3.3：KV cache 约占权重的 20-30%，保守取 0.25
KV_CACHE_WEIGHT_RATIO = 0.25

# SPEC §3.4：CUDA/ROCm context 默认开销（GB）
# 注意：MI300X 上实际更大（4-5GB），通过 config.cuda_context_overhead_gb 覆盖。
DEFAULT_CUDA_CONTEXT_OVERHEAD_GB = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 计算 API
# ─────────────────────────────────────────────────────────────────────────────


def calculate_weight_memory(
    params_billion: float,
    dtype: str,
    quantization: str | None = None,
) -> float:
    """计算模型权重显存（GB）。

    公式（SPEC §3.2）：
        W = P × B × (1 + Q_overhead)

    Args:
        params_billion: 参数量（billions）。如 7.0 表示 7B。
        dtype: fp32/fp16/bf16/int8/int4。
        quantization: gptq/awq/gguf 或 None（不量化）。

    Returns:
        显存（GB），保留完整精度（不四舍五入）。
        SPEC §3.4 的"保留 1 位小数"用于显示/日志；计算保持精度避免累积误差。

    Raises:
        ValueError: dtype 或 quantization 不在已知表里；params_billion 为负。

    Examples:
        >>> calculate_weight_memory(7.0, "fp16")
        14.0
        >>> calculate_weight_memory(70.0, "int4", "gptq")
        36.75
        >>> calculate_weight_memory(72.0, "int4", "gptq")  # SPEC §3.2 示例
        37.8
    """
    if params_billion < 0:
        msg = f"params_billion 不能为负，得到 {params_billion}"
        raise ValueError(msg)
    if dtype not in DTYPE_BYTES:
        msg = f"未知 dtype: {dtype!r}，支持的 dtype: {sorted(DTYPE_BYTES)}"
        raise ValueError(msg)
    if quantization is not None and quantization not in QUANT_OVERHEAD:
        msg = f"未知 quantization: {quantization!r}，支持的: {sorted(QUANT_OVERHEAD)} 或 None"
        raise ValueError(msg)

    bytes_per_param = DTYPE_BYTES[dtype]
    overhead = QUANT_OVERHEAD.get(quantization, 0.0) if quantization else 0.0
    return params_billion * bytes_per_param * (1.0 + overhead)


def estimate_kv_cache_from_weight(
    weight_gb: float,
    utilization: float = 0.90,
) -> float:
    """估算 KV cache 显存（GB）。

    简化版（SPEC §3.3，架构参数未知时）：
        K = W × 0.25 × U

    Args:
        weight_gb: 权重显存（GB）。
        utilization: vLLM 的 gpu_memory_utilization，默认 0.90。

    Returns:
        KV cache 预算（GB），保留完整精度。

    Raises:
        ValueError: utilization 不在 (0, 1] 或 weight_gb 为负。
    """
    if weight_gb < 0:
        msg = f"weight_gb 不能为负，得到 {weight_gb}"
        raise ValueError(msg)
    if not 0.0 < utilization <= 1.0:
        msg = f"utilization 必须在 (0, 1]，得到 {utilization}"
        raise ValueError(msg)
    return weight_gb * KV_CACHE_WEIGHT_RATIO * utilization


def build_model_profile(
    params_billion: float,
    dtype: str,
    quantization: str | None = None,
    *,
    utilization: float = 0.90,
) -> ModelMemoryProfile:
    """构造模型的完整显存画像。

    一步到位：计算权重 + KV cache + 装进不可变 dataclass。

    Args:
        同 calculate_weight_memory + utilization。

    Returns:
        ModelMemoryProfile（frozen dataclass）。
    """
    weight = calculate_weight_memory(params_billion, dtype, quantization)
    kv = estimate_kv_cache_from_weight(weight, utilization)
    return ModelMemoryProfile(
        params_billion=params_billion,
        dtype=dtype,
        quantization=quantization,
        weight_gb=weight,
        kv_cache_budget_gb=kv,
    )


# ─────────────────────────────────────────────────────────────────────────────
# can_load 决策树（SPEC §3.4，核心算法）
# ─────────────────────────────────────────────────────────────────────────────


def can_load(
    gpu_state: GPUState,
    new_model: ModelMemoryProfile,
    *,
    loaded_models: list[ModelMemoryProfile] | None = None,
    cuda_context_overhead_gb: float = DEFAULT_CUDA_CONTEXT_OVERHEAD_GB,
) -> LoadDecision:
    """判定能否在当前 GPU 上加载新模型，返回策略。

    决策树（SPEC §3.4）：
        1. required = new_model.weight + new_model.kv_cache
        2. current_free >= required → DIRECT
        3. current_free + sum(sleep 释放的 KV) >= required → SLEEP_OLD
        4. total - cuda_overhead >= required → UNLOAD_OLD
        5. 否则 → INSUFFICIENT

    Args:
        gpu_state: GPU 当前显存状态。
        new_model: 待加载模型的显存画像。
        loaded_models: 当前已加载在 GPU 上的其他模型（用于计算可释放量）。
        cuda_context_overhead_gb: CUDA/ROCm context 开销。
            MI300X 上推荐 4-5（通过 config 注入）。

    Returns:
        LoadDecision（含 strategy + free_after_load + detail）。

    Note:
        - DIRECT/SLEEP_OLD/UNLOAD_OLD 都是"可加载"，can_proceed=True。
        - 本函数**不执行**加载，只做决策。
        - loaded_models 仅用于 SLEEP_OLD 分支计算可释放 KV cache。
          如果有多个已加载模型，sum 它们的 kv_cache_budget_gb。
    """
    loaded_models = loaded_models or []
    required = new_model.total_gb
    current_free = gpu_state.free_gb

    # 分支 1：DIRECT
    if current_free >= required:
        free_after = current_free - required
        return LoadDecision(
            strategy=LoadStrategy.DIRECT,
            free_after_load_gb=free_after,
            required_gb=required,
            detail=(
                f"显存充足，直接加载。当前剩余 {current_free:.1f}GB，"
                f"需要 {required:.1f}GB，加载后剩 {free_after:.1f}GB。"
            ),
        )

    # 分支 2：SLEEP_OLD（sleep 已加载模型释放 KV cache）
    # 每个 loaded_model sleep 后释放其 kv_cache_budget_gb（权重仍占用）
    releasable_kv = sum(m.kv_cache_budget_gb for m in loaded_models)
    if current_free + releasable_kv >= required:
        free_after = current_free + releasable_kv - required
        models_desc = (
            f"{len(loaded_models)} 个已加载模型" if len(loaded_models) != 1 else "1 个已加载模型"
        )
        return LoadDecision(
            strategy=LoadStrategy.SLEEP_OLD,
            free_after_load_gb=free_after,
            required_gb=required,
            detail=(
                f"显存不足（剩 {current_free:.1f}GB，需 {required:.1f}GB），"
                f"但 sleep {models_desc} 可释放 {releasable_kv:.1f}GB KV cache，"
                f"释放后剩 {free_after:.1f}GB 足够。"
            ),
        )

    # 分支 3：UNLOAD_OLD（unload 已加载模型释放全部显存，含权重）
    # unload 后，整个 GPU 只剩 CUDA context + 新模型
    total_releasable = sum(m.total_gb for m in loaded_models)
    free_after_unload = current_free + total_releasable
    if free_after_unload >= required:
        free_after = free_after_unload - required
        return LoadDecision(
            strategy=LoadStrategy.UNLOAD_OLD,
            free_after_load_gb=free_after,
            required_gb=required,
            detail=(
                f"显存严重不足（剩 {current_free:.1f}GB，需 {required:.1f}GB），"
                f"必须 unload 已加载模型释放 {total_releasable:.1f}GB 全部显存，"
                f"卸载后加载剩 {free_after:.1f}GB。"
            ),
        )

    # 分支 4：INSUFFICIENT（即使空 GPU 也放不下）
    # 空 GPU 可用 = total - cuda_overhead
    bare_gpu_capacity = gpu_state.total_gb - cuda_context_overhead_gb
    return LoadDecision(
        strategy=LoadStrategy.INSUFFICIENT,
        free_after_load_gb=0.0,
        required_gb=required,
        detail=(
            f"模型太大，无法加载。需要 {required:.1f}GB，"
            f"当前剩余 {current_free:.1f}GB，即使清空 GPU"
            f"（保留 {cuda_context_overhead_gb:.1f}GB context）"
            f"也只有 {bare_gpu_capacity:.1f}GB 可用。"
        ),
    )
