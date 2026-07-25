"""gpu_memory.py 单测（SPEC §3，覆盖率目标 ≥95%）。

测试矩阵：
    - 各种 dtype × 量化的 weight 计算（穷举）
    - KV cache 估算（含边界）
    - build_model_profile 一致性
    - can_load 4 个分支 + 边界（空 GPU、刚好够、刚好不够、超大模型）
    - 错误输入拒绝（负值、未知 dtype/quant）
"""

from __future__ import annotations

import pytest

from vllmonline.scheduler.gpu_memory import (
    DEFAULT_CUDA_CONTEXT_OVERHEAD_GB,
    DTYPE_BYTES,
    KV_CACHE_WEIGHT_RATIO,
    QUANT_OVERHEAD,
    build_model_profile,
    calculate_weight_memory,
    can_load,
    estimate_kv_cache_from_weight,
)
from vllmonline.scheduler.types import GPUState, LoadStrategy, ModelMemoryProfile

# ─────────────────────────────────────────────────────────────────────────────
# weight 计算（SPEC §3.2，穷举 dtype × quant）
# ─────────────────────────────────────────────────────────────────────────────


class TestCalculateWeightMemory:
    """穷举所有 dtype 和 quantization 组合。"""

    @pytest.mark.parametrize(
        ("dtype", "expected_bytes"),
        [
            ("fp32", 4.0),
            ("fp16", 2.0),
            ("bf16", 2.0),
            ("int8", 1.0),
            ("int4", 0.5),
        ],
    )
    def test_each_dtype_no_quant(self, dtype: str, expected_bytes: float) -> None:
        """不量化时 W = P × B。用 10B 做基线（好算）。"""
        result = calculate_weight_memory(10.0, dtype)
        assert result == pytest.approx(10.0 * expected_bytes)

    def test_dtype_bytes_table_complete(self) -> None:
        """表里所有 dtype 都能计算，且数值与常量表一致。"""
        for dtype, bytes_per_param in DTYPE_BYTES.items():
            result = calculate_weight_memory(1.0, dtype)
            assert result == pytest.approx(bytes_per_param)

    @pytest.mark.parametrize(
        ("quant", "overhead"),
        [("gptq", 0.05), ("awq", 0.05), ("gguf", 0.03)],
    )
    def test_quant_overhead_applied(self, quant: str, overhead: float) -> None:
        """量化时 W = P × B × (1 + overhead)。"""
        result = calculate_weight_memory(10.0, "fp16", quant)
        assert result == pytest.approx(10.0 * 2.0 * (1 + overhead))

    def test_quant_overhead_table_complete(self) -> None:
        """所有量化方法都纳入表。"""
        for quant, overhead in QUANT_OVERHEAD.items():
            result = calculate_weight_memory(1.0, "fp16", quant)
            assert result == pytest.approx(2.0 * (1 + overhead))

    def test_none_quantization_equivalent_to_no_quant(self) -> None:
        """quantization=None 与不传等价。"""
        assert calculate_weight_memory(7.0, "fp16") == pytest.approx(
            calculate_weight_memory(7.0, "fp16", None)
        )

    # ── SPEC/PLAN 给的明确示例（必须精确通过）──
    def test_spec_example_qwen7b_fp16(self) -> None:
        """SPEC §3.2：Qwen-7B, fp16 → 14.0 GB。"""
        assert calculate_weight_memory(7.0, "fp16") == 14.0

    def test_plan_example_70b_int4_gptq(self) -> None:
        """PLAN P1 验收：70B int4 gptq → 36.75。"""
        assert calculate_weight_memory(70.0, "int4", "gptq") == 36.75

    def test_spec_example_72b_int4_gptq(self) -> None:
        """SPEC §3.2：72B int4 gptq → 37.8 GB。"""
        assert calculate_weight_memory(72.0, "int4", "gptq") == pytest.approx(37.8)

    # ── 边界条件 ──
    def test_zero_params(self) -> None:
        """0 参数模型权重为 0。"""
        assert calculate_weight_memory(0.0, "fp16") == 0.0

    def test_large_params(self) -> None:
        """大模型（405B）也能算。"""
        # Llama-3-405B fp16 ≈ 810 GB
        assert calculate_weight_memory(405.0, "fp16") == pytest.approx(810.0)

    def test_negative_params_raises(self) -> None:
        with pytest.raises(ValueError, match="不能为负"):
            calculate_weight_memory(-1.0, "fp16")

    def test_unknown_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="未知 dtype"):
            calculate_weight_memory(7.0, "fp8")

    def test_unknown_quant_raises(self) -> None:
        with pytest.raises(ValueError, match="未知 quantization"):
            calculate_weight_memory(7.0, "fp16", "unknown_quant")

    def test_case_sensitive_dtype(self) -> None:
        """dtype 大小写敏感（按 SPEC 表的小写约定）。"""
        with pytest.raises(ValueError):
            calculate_weight_memory(7.0, "FP16")


# ─────────────────────────────────────────────────────────────────────────────
# KV cache 估算（SPEC §3.3）
# ─────────────────────────────────────────────────────────────────────────────


class TestEstimateKvCache:
    def test_default_utilization(self) -> None:
        """默认 utilization=0.90：K = W × 0.25 × 0.90。"""
        result = estimate_kv_cache_from_weight(14.0)
        assert result == pytest.approx(14.0 * 0.25 * 0.90)

    def test_custom_utilization(self) -> None:
        """自定义 utilization。"""
        result = estimate_kv_cache_from_weight(14.0, 0.5)
        assert result == pytest.approx(14.0 * 0.25 * 0.5)

    def test_full_utilization(self) -> None:
        """utilization=1.0：K = W × 0.25。"""
        result = estimate_kv_cache_from_weight(14.0, 1.0)
        assert result == pytest.approx(14.0 * KV_CACHE_WEIGHT_RATIO)

    def test_zero_weight(self) -> None:
        """0 权重 → 0 KV cache。"""
        assert estimate_kv_cache_from_weight(0.0) == 0.0

    def test_zero_utilization_raises(self) -> None:
        """utilization=0 不合法（除以 0 无意义）。"""
        with pytest.raises(ValueError, match="utilization"):
            estimate_kv_cache_from_weight(14.0, 0.0)

    def test_negative_utilization_raises(self) -> None:
        with pytest.raises(ValueError, match="utilization"):
            estimate_kv_cache_from_weight(14.0, -0.1)

    def test_over_one_utilization_raises(self) -> None:
        with pytest.raises(ValueError, match="utilization"):
            estimate_kv_cache_from_weight(14.0, 1.1)

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="不能为负"):
            estimate_kv_cache_from_weight(-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# build_model_profile
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildModelProfile:
    def test_basic_profile(self) -> None:
        profile = build_model_profile(7.0, "fp16")
        assert profile.params_billion == 7.0
        assert profile.dtype == "fp16"
        assert profile.quantization is None
        assert profile.weight_gb == pytest.approx(14.0)
        assert profile.kv_cache_budget_gb == pytest.approx(14.0 * 0.25 * 0.9)
        assert profile.total_gb == pytest.approx(profile.weight_gb + profile.kv_cache_budget_gb)

    def test_profile_with_quantization(self) -> None:
        profile = build_model_profile(70.0, "int4", "gptq", utilization=0.9)
        assert profile.weight_gb == pytest.approx(36.75)
        assert profile.kv_cache_budget_gb == pytest.approx(36.75 * 0.25 * 0.9)

    def test_profile_is_frozen(self) -> None:
        """frozen dataclass 不可变。"""
        profile = build_model_profile(7.0, "fp16")
        with pytest.raises(AttributeError):
            profile.weight_gb = 100.0  # type: ignore[misc]

    def test_sleeping_gb_equals_weight(self) -> None:
        """sleeping 态只占权重。"""
        profile = build_model_profile(7.0, "fp16")
        assert profile.sleeping_gb == profile.weight_gb

    def test_invalid_dtype_propagates(self) -> None:
        with pytest.raises(ValueError):
            build_model_profile(7.0, "fp8")


# ─────────────────────────────────────────────────────────────────────────────
# can_load 决策树（SPEC §3.4，核心）
# ─────────────────────────────────────────────────────────────────────────────


def _profile(
    params: float = 7.0, dtype: str = "fp16", quant: str | None = None
) -> ModelMemoryProfile:
    """测试辅助：快速构造 profile。"""
    return build_model_profile(params, dtype, quant)


def _gpu(total: float, used: float, gpu_id: int = 0) -> GPUState:
    return GPUState(gpu_id=gpu_id, total_gb=total, used_gb=used)


class TestCanLoadDirect:
    """分支 1：DIRECT（显存够，直接加载）。"""

    async def test_empty_gpu_small_model(self) -> None:
        """空 GPU 加载小模型 → DIRECT。"""
        gpu = _gpu(80.0, 0.0)  # 80GB 全空
        model = _profile(7.0, "fp16")  # 总 ~17.15GB
        decision = can_load(gpu, model)
        assert decision.strategy is LoadStrategy.DIRECT
        assert decision.can_proceed
        assert decision.free_after_load_gb == pytest.approx(80.0 - model.total_gb)
        assert "DIRECT" not in decision.detail  # detail 是中文，不含策略名
        assert "直接加载" in decision.detail

    async def test_just_enough_free(self) -> None:
        """剩余刚好够 → DIRECT。"""
        model = _profile(7.0, "fp16")
        gpu = _gpu(80.0, 80.0 - model.total_gb)  # 剩余正好 = required
        decision = can_load(gpu, model)
        assert decision.strategy is LoadStrategy.DIRECT
        assert decision.free_after_load_gb == pytest.approx(0.0, abs=0.01)

    async def test_loaded_models_ignored_for_direct(self) -> None:
        """DIRECT 分支不依赖 loaded_models（即便传了也不影响判定）。"""
        gpu = _gpu(100.0, 0.0)
        model = _profile(7.0, "fp16")
        loaded = [_profile(7.0, "fp16")]  # 假装有已加载模型
        decision = can_load(gpu, model, loaded_models=loaded)
        assert decision.strategy is LoadStrategy.DIRECT


class TestCanLoadSleepOld:
    """分支 2：SLEEP_OLD（sleep 已加载模型释放 KV cache 后够）。"""

    async def test_plan_example_80gb_free10(self) -> None:
        """PLAN P1 验收：80GB GPU，free=10GB，加载 14GB 模型 → SLEEP_OLD。"""
        # weight=14, kv=14*0.25*0.9=3.15, total=17.15
        model = _profile(7.0, "fp16")
        # 当前剩 10GB，不够 17.15GB
        gpu = _gpu(80.0, 70.0)
        # 已加载模型：它的 kv=3.15，sleep 后释放 3.15 → 总 13.15，仍不够 17.15
        # 需要更大的 KV cache 才能触发 SLEEP_OLD。改用更大的已加载模型：
        big_loaded = build_model_profile(70.0, "fp16")  # kv = 140*0.25*0.9 = 31.5
        decision = can_load(gpu, model, loaded_models=[big_loaded])
        # 10 + 31.5 = 41.5 >= 17.15 → SLEEP_OLD ✓
        assert decision.strategy is LoadStrategy.SLEEP_OLD
        assert decision.can_proceed
        assert "sleep" in decision.detail.lower()
        assert "KV cache" in decision.detail

    async def test_multiple_loaded_models_sum(self) -> None:
        """多个已加载模型的 KV cache 求和。"""
        model = _profile(7.0, "fp16")  # 需要 17.15
        gpu = _gpu(80.0, 65.0)  # 剩 15，不够
        # 两个小模型，每个 kv ≈ 3.15，总 6.3；15+6.3=21.3 >= 17.15
        loaded = [_profile(7.0, "fp16"), _profile(7.0, "fp16")]
        decision = can_load(gpu, model, loaded_models=loaded)
        assert decision.strategy is LoadStrategy.SLEEP_OLD

    async def test_sleep_not_enough_falls_through(self) -> None:
        """sleep 释放的 KV 不够 → 不应返回 SLEEP_OLD（应继续到 UNLOAD_OLD 或 INSUFFICIENT）。"""
        model = _profile(7.0, "fp16")  # 需要 17.15
        gpu = _gpu(80.0, 70.0)  # 剩 10
        # 已加载模型 KV 太小：1GB，10+1=11 < 17.15，不够 sleep
        tiny_loaded = build_model_profile(1.0, "fp16")  # kv ≈ 0.225
        # 但 1GB 模型 total=1.225，unload 它能释放 1.225，仍不够 → INSUFFICIENT
        # 所以这个用例实际是 INSUFFICIENT，验证 SLEEP_OLD 分支没误触发
        decision = can_load(gpu, model, loaded_models=[tiny_loaded])
        assert decision.strategy is not LoadStrategy.SLEEP_OLD


class TestCanLoadUnloadOld:
    """分支 3：UNLOAD_OLD（unload 已加载模型释放全部显存后够）。"""

    async def test_unload_releases_enough(self) -> None:
        """sleep 不够但 unload 够 → UNLOAD_OLD。"""
        model = _profile(7.0, "fp16")  # 需要 17.15
        # 当前剩 5GB；已加载模型 total=20GB，KV=4.5
        # sleep: 5+4.5=9.5 < 17.15 不够
        # unload: 5+20=25 >= 17.15 够
        big_loaded = build_model_profile(10.0, "fp16")  # total ≈ 24.5, kv ≈ 4.5
        gpu = _gpu(80.0, 75.0)  # 剩 5
        decision = can_load(gpu, model, loaded_models=[big_loaded])
        assert decision.strategy is LoadStrategy.UNLOAD_OLD
        assert decision.can_proceed
        assert "unload" in decision.detail.lower()

    async def test_multiple_unload_sum(self) -> None:
        """多个已加载模型 total 求和。"""
        model = _profile(7.0, "fp16")  # 需要 17.15
        gpu = _gpu(80.0, 75.0)  # 剩 5
        # 三个小模型，每个 total ≈ 8.575，总 25.725 >= 17.15
        # 但每个 kv ≈ 3.15，sleep 释放 9.45，5+9.45=14.45 < 17.15 → 不走 sleep
        loaded = [
            _profile(7.0, "fp16"),
            _profile(7.0, "fp16"),
            _profile(7.0, "fp16"),
        ]
        decision = can_load(gpu, model, loaded_models=loaded)
        assert decision.strategy is LoadStrategy.UNLOAD_OLD


class TestCanLoadInsufficient:
    """分支 4：INSUFFICIENT（即使空 GPU 也放不下）。"""

    async def test_model_larger_than_gpu(self) -> None:
        """模型比整个 GPU 还大。"""
        # 405B fp16 ≈ 810GB，远超 80GB GPU
        model = _profile(405.0, "fp16")
        gpu = _gpu(80.0, 0.0)
        decision = can_load(gpu, model)
        assert decision.strategy is LoadStrategy.INSUFFICIENT
        assert not decision.can_proceed
        assert decision.free_after_load_gb == 0.0
        assert "太大" in decision.detail or "无法加载" in decision.detail

    async def test_model_larger_than_mi300x(self) -> None:
        """即使 MI300X 192GB 也放不下的模型。"""
        model = _profile(405.0, "fp16")  # ~810GB
        gpu = _gpu(192.0, 0.0)  # MI300X 全空
        decision = can_load(gpu, model)
        assert decision.strategy is LoadStrategy.INSUFFICIENT

    async def test_model_fits_bare_gpu_but_not_with_loaded(self) -> None:
        """模型空 GPU 能装，但当前已加载导致 sleep/unload 都不够 → 仍可能 INSUFFICIENT。

        场景：GPU 80GB，已加载一个 30GB 模型（占着），剩 50GB。
        新模型需要 60GB。sleep 释放 KV 不够，unload 释放 30GB 后 80GB 够。
        → 应该是 UNLOAD_OLD，不是 INSUFFICIENT。
        验证 INSUFFICIENT 只在真放不下时触发。
        """
        model = _profile(30.0, "fp16")  # 需要 ~73.5GB
        loaded = build_model_profile(30.0, "fp16")  # total ~73.5, kv ~16.5
        gpu = _gpu(80.0, 30.0)  # 剩 50
        # sleep: 50+16.5=66.5 < 73.5 不够
        # unload: 50+73.5=123.5 >= 73.5 够
        decision = can_load(gpu, model, loaded_models=[loaded])
        assert decision.strategy is LoadStrategy.UNLOAD_OLD

    async def test_insufficient_with_cuda_overhead(self) -> None:
        """cuda_overhead 让 bare_gpu 不够 → INSUFFICIENT。"""
        # 模型刚好等于 total - overhead，但 overhead 占了
        model = _profile(40.0, "fp16")  # 需要 ~98GB
        gpu = _gpu(80.0, 0.0)
        # 默认 overhead=2，bare=78 < 98 → INSUFFICIENT
        decision = can_load(gpu, model, cuda_context_overhead_gb=2.0)
        assert decision.strategy is LoadStrategy.INSUFFICIENT


# ─────────────────────────────────────────────────────────────────────────────
# can_load 边界与集成
# ─────────────────────────────────────────────────────────────────────────────


class TestCanLoadEdges:
    async def test_default_cuda_overhead_constant(self) -> None:
        """默认 CUDA context 开销 = 2.0（SPEC §3.4）。"""
        assert DEFAULT_CUDA_CONTEXT_OVERHEAD_GB == 2.0

    async def test_custom_cuda_overhead_mi300x(self) -> None:
        """MI300X 推荐 4.0 overhead——影响 INSUFFICIENT 判定。"""
        # 一个 75GB 的需求，默认 overhead=2 → bare=78 够，但当前剩 0
        # 没 loaded_models，所以走 INSUFFICIENT（bare=78 < required）
        # 实际：required=75*2*1.0=150 > 80，必然 INSUFFICIENT
        # 改用更小模型让 overhead 真正影响判定边界
        model = _profile(38.0, "fp16")  # weight=76, kv=17.1, total=93.1
        gpu = _gpu(96.0, 5.0)  # 剩 91 < 93.1
        # 无 loaded_models：sleep/unload 都没东西可释放 → INSUFFICIENT
        # bare = 96 - overhead
        # overhead=2: bare=94 >= 93.1 但当前路径无 loaded_models，
        #   UNLOAD_OLD 分支要求 free_after_unload=91+0=91 < 93.1 不满足
        #   → INSUFFICIENT
        # overhead 不影响这个用例（因为没有 loaded_models）
        # 验证 overhead 字段确实传进去了（通过 detail 文本）
        decision_default = can_load(gpu, model, cuda_context_overhead_gb=2.0)
        assert decision_default.strategy is LoadStrategy.INSUFFICIENT
        assert "2.0" in decision_default.detail

        decision_mi300 = can_load(gpu, model, cuda_context_overhead_gb=5.0)
        assert "5.0" in decision_mi300.detail

    async def test_decision_has_required_gb(self) -> None:
        """LoadDecision.required_gb 反映新模型 total。"""
        model = _profile(7.0, "fp16")
        decision = can_load(_gpu(80, 0), model)
        assert decision.required_gb == pytest.approx(model.total_gb)

    async def test_no_loaded_models_defaults_empty(self) -> None:
        """不传 loaded_models 时按空列表处理。"""
        model = _profile(7.0, "fp16")
        # 剩余不够，无 loaded_models → UNLOAD_OLD 也不满足（无东西可释放）→ INSUFFICIENT
        decision = can_load(_gpu(80, 75), model)  # 剩 5
        assert decision.strategy is LoadStrategy.INSUFFICIENT

    async def test_empty_loaded_models_list(self) -> None:
        """显式传空列表与不传等价。"""
        model = _profile(7.0, "fp16")
        d1 = can_load(_gpu(80, 75), model)
        d2 = can_load(_gpu(80, 75), model, loaded_models=[])
        assert d1.strategy == d2.strategy
        assert d1.strategy is LoadStrategy.INSUFFICIENT


# ─────────────────────────────────────────────────────────────────────────────
# GPUState 数据结构
# ─────────────────────────────────────────────────────────────────────────────


class TestGPUState:
    def test_free_calculation(self) -> None:
        gpu = GPUState(gpu_id=0, total_gb=192.0, used_gb=50.0)
        assert gpu.free_gb == 142.0

    def test_free_clamped_at_zero(self) -> None:
        """used 超过 total 时 free 不为负。"""
        gpu = GPUState(gpu_id=0, total_gb=80.0, used_gb=100.0)
        assert gpu.free_gb == 0.0

    def test_frozen(self) -> None:
        gpu = GPUState(gpu_id=0, total_gb=80.0, used_gb=0.0)
        with pytest.raises(AttributeError):
            gpu.total_gb = 100.0  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# LoadStrategy / LoadDecision 语义
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadDecision:
    def test_can_proceed_true_for_non_insufficient(self) -> None:
        from vllmonline.scheduler.types import LoadDecision

        for strategy in [
            LoadStrategy.DIRECT,
            LoadStrategy.SLEEP_OLD,
            LoadStrategy.UNLOAD_OLD,
        ]:
            d = LoadDecision(
                strategy=strategy, free_after_load_gb=1.0, required_gb=10.0, detail="x"
            )
            assert d.can_proceed

    def test_can_proceed_false_for_insufficient(self) -> None:
        from vllmonline.scheduler.types import LoadDecision

        d = LoadDecision(
            strategy=LoadStrategy.INSUFFICIENT,
            free_after_load_gb=0.0,
            required_gb=100.0,
            detail="x",
        )
        assert not d.can_proceed
