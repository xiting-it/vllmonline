"""gpu_info.py 单测。

主要测试解析器（_parse_rocm_smi / _parse_nvidia_smi），
provider 的 subprocess 调用靠 mock（CI/开发环境通常无 rocm-smi/nvidia-smi）。
"""

from __future__ import annotations

import pytest

from vllmonline.config import GpuBackend
from vllmonline.gpu_info import (
    NoneGpuProvider,
    _parse_nvidia_smi,
    _parse_rocm_smi,
    make_gpu_provider,
)

# ─────────────────────────────────────────────────────────────────────────────
# rocm-smi 解析器
# ─────────────────────────────────────────────────────────────────────────────


ROCM_SMI_OUTPUT_MI300X = """================================ ROCm SMI =================================
GPU[0]          : VRAM Total Memory (B): 201761212416
GPU[0]          : VRAM Total Used Memory (B): 53687091200
==============================================================================
"""


class TestParseRocmSmi:
    def test_mi300x_192gb(self) -> None:
        """MI300X 192GB，用了 50GB。"""
        total_b, used_b = _parse_rocm_smi(ROCM_SMI_OUTPUT_MI300X, gpu_id=0)
        assert total_b == 201761212416  # ≈ 187.8 GB
        assert used_b == 53687091200  # 50 GB

    def test_different_gpu_id(self) -> None:
        """指定 gpu_id 过滤。"""
        output = """GPU[0]          : VRAM Total Memory (B): 1000
GPU[0]          : VRAM Total Used Memory (B): 100
GPU[1]          : VRAM Total Memory (B): 2000
GPU[1]          : VRAM Total Used Memory (B): 200
"""
        total_b, used_b = _parse_rocm_smi(output, gpu_id=1)
        assert total_b == 2000
        assert used_b == 200

    def test_empty_output_raises(self) -> None:
        with pytest.raises(ValueError, match="无法从 rocm-smi"):
            _parse_rocm_smi("", gpu_id=0)

    def test_malformed_output_raises(self) -> None:
        with pytest.raises(ValueError, match="无法从 rocm-smi"):
            _parse_rocm_smi("garbage without numbers", gpu_id=0)

    def test_partial_output_raises(self) -> None:
        """只有 total 没有 used → 解析失败。"""
        output = "GPU[0]          : VRAM Total Memory (B): 1000\n"
        with pytest.raises(ValueError):
            _parse_rocm_smi(output, gpu_id=0)


# ─────────────────────────────────────────────────────────────────────────────
# nvidia-smi 解析器
# ─────────────────────────────────────────────────────────────────────────────


class TestParseNvidiaSmi:
    def test_basic(self) -> None:
        total_mb, used_mb = _parse_nvidia_smi("81920, 5120\n")
        assert total_mb == 81920  # 80GB
        assert used_mb == 5120  # 5GB

    def test_strips_whitespace(self) -> None:
        total_mb, used_mb = _parse_nvidia_smi("  81920 ,  5120  \n")
        assert total_mb == 81920
        assert used_mb == 5120

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="无法从 nvidia-smi"):
            _parse_nvidia_smi("")

    def test_single_value_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_nvidia_smi("81920")


# ─────────────────────────────────────────────────────────────────────────────
# NoneGpuProvider
# ─────────────────────────────────────────────────────────────────────────────


class TestNoneGpuProvider:
    async def test_returns_fake_snapshot(self) -> None:
        provider = NoneGpuProvider(fake_total_gb=80.0, fake_used_gb=10.0)
        snapshot = await provider.snapshot()
        assert snapshot.total_gb == 80.0
        assert snapshot.used_gb == 10.0

    async def test_to_gpu_state(self) -> None:
        provider = NoneGpuProvider(fake_total_gb=192.0, fake_used_gb=50.0)
        state = await provider.to_gpu_state()
        assert state.total_gb == 192.0
        assert state.used_gb == 50.0
        assert state.free_gb == 142.0


# ─────────────────────────────────────────────────────────────────────────────
# make_gpu_provider（AUTO 探测）
# ─────────────────────────────────────────────────────────────────────────────


class TestMakeGpuProvider:
    def test_explicit_rocm(self) -> None:
        from vllmonline.gpu_info import RocmSmiProvider

        provider = make_gpu_provider(GpuBackend.ROCM)
        assert isinstance(provider, RocmSmiProvider)

    def test_explicit_nvidia(self) -> None:
        from vllmonline.gpu_info import NvidiaSmiProvider

        provider = make_gpu_provider(GpuBackend.NVIDIA)
        assert isinstance(provider, NvidiaSmiProvider)

    def test_explicit_none(self) -> None:
        provider = make_gpu_provider(GpuBackend.NONE)
        assert isinstance(provider, NoneGpuProvider)

    def test_auto_falls_back_to_none_when_no_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """无 GPU 环境（开发机）→ NoneGpuProvider。"""
        monkeypatch.setattr("vllmonline.gpu_info.shutil.which", lambda _: None)
        provider = make_gpu_provider(GpuBackend.AUTO)
        assert isinstance(provider, NoneGpuProvider)

    def test_auto_picks_rocm_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """有 rocm-smi → 选 ROCm。"""
        from vllmonline.gpu_info import RocmSmiProvider

        def fake_which(cmd: str) -> str | None:
            return "/usr/bin/rocm-smi" if cmd == "rocm-smi" else None

        monkeypatch.setattr("vllmonline.gpu_info.shutil.which", fake_which)
        provider = make_gpu_provider(GpuBackend.AUTO)
        assert isinstance(provider, RocmSmiProvider)

    def test_auto_picks_nvidia_when_no_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from vllmonline.gpu_info import NvidiaSmiProvider

        def fake_which(cmd: str) -> str | None:
            return "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else None

        monkeypatch.setattr("vllmonline.gpu_info.shutil.which", fake_which)
        provider = make_gpu_provider(GpuBackend.AUTO)
        assert isinstance(provider, NvidiaSmiProvider)
