"""GPU 信息查询抽象层。

SPEC 通篇假设 NVIDIA/CUDA（nvidia-smi），但本项目目标是 MI300X（ROCm，
rocm-smi）。本模块抽象出统一接口，按 config.gpu.backend 选择实现：

    ROCM   → rocm-smi --showmeminfo vram
    NVIDIA → nvidia-smi --query-gpu=... --format=csv
    NONE   → 返回全零（开发/测试，无 GPU）

MI300X 关键事实（2026-07 调研）：
    - 192 GB HBM3，架构 gfx942，ROCm 6.2+ 原生支持（无需 HSA_OVERRIDE）
    - rocm-smi --showmeminfo vram 输出每 GPU VRAM 使用
    - 已知坑：非 SPX 分区模式下 rocm-smi 把每分区显存显示成总量
"""

from __future__ import annotations

import abc
import asyncio
import re
import shutil
from dataclasses import dataclass

from vllmonline.config import GpuBackend
from vllmonline.scheduler.types import GPUState


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    """整张 GPU 的快照（多个 GPU 时取第一张，本项目假设单 GPU）。"""

    gpu_id: int
    total_gb: float
    used_gb: float

    def to_gpu_state(self) -> GPUState:
        return GPUState(gpu_id=self.gpu_id, total_gb=self.total_gb, used_gb=self.used_gb)


class GpuInfoProvider(abc.ABC):
    """GPU 信息提供者抽象。"""

    @abc.abstractmethod
    async def snapshot(self, gpu_id: int = 0) -> GpuSnapshot:
        """采集指定 GPU 的当前显存快照。"""

    async def to_gpu_state(self, gpu_id: int = 0) -> GPUState:
        """便捷方法：直接返回 GPUState。"""
        return (await self.snapshot(gpu_id)).to_gpu_state()


class NoneGpuProvider(GpuInfoProvider):
    """无 GPU 的占位实现（开发/测试）。

    返回一个虚拟的 80GB GPU（模拟 A100），显存空闲。
    仅供无 GPU 环境跑通流程，不能用于真实显存决策。
    """

    def __init__(self, fake_total_gb: float = 80.0, fake_used_gb: float = 0.0) -> None:
        self._total = fake_total_gb
        self._used = fake_used_gb

    async def snapshot(self, gpu_id: int = 0) -> GpuSnapshot:
        return GpuSnapshot(gpu_id=gpu_id, total_gb=self._total, used_gb=self._used)


class RocmSmiProvider(GpuInfoProvider):
    """AMD ROCm 实现，解析 rocm-smi 输出。

    rocm-smi --showmeminfo vram 输出示例：
        ================================ ROCm SMI =================================
        GPU[0]          : VRAM Total Memory (B): 201761212416
        GPU[0]          : VRAM Total Used Memory (B): 12345678900
        ======================================================================
    """

    COMMAND: tuple[str, ...] = ("rocm-smi", "--showmeminfo", "vram")

    async def snapshot(self, gpu_id: int = 0) -> GpuSnapshot:
        proc = await asyncio.create_subprocess_exec(
            *self.COMMAND,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            msg = (
                f"rocm-smi 失败 (exit {proc.returncode}): {stderr.decode().strip() or 'no stderr'}"
            )
            raise RuntimeError(msg)
        total_b, used_b = _parse_rocm_smi(stdout.decode(), gpu_id)
        return GpuSnapshot(
            gpu_id=gpu_id,
            total_gb=round(total_b / 1024**3, 2),
            used_gb=round(used_b / 1024**3, 2),
        )


class NvidiaSmiProvider(GpuInfoProvider):
    """NVIDIA CUDA 实现，解析 nvidia-smi 输出（备用，本项目主用 ROCm）。

    命令：nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits
    输出：24564, 3210  （单位 MB）
    """

    COMMAND: tuple[str, ...] = (
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used",
        "--format=csv,noheader,nounits",
    )

    async def snapshot(self, gpu_id: int = 0) -> GpuSnapshot:
        # nvidia-smi 用 -i 选 GPU
        cmd = [*self.COMMAND, "-i", str(gpu_id)]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            msg = (
                f"nvidia-smi 失败 (exit {proc.returncode}): "
                f"{stderr.decode().strip() or 'no stderr'}"
            )
            raise RuntimeError(msg)
        total_mb, used_mb = _parse_nvidia_smi(stdout.decode())
        return GpuSnapshot(
            gpu_id=gpu_id,
            total_gb=round(total_mb / 1024, 2),
            used_gb=round(used_mb / 1024, 2),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 解析器（独立函数，便于单测）
# ─────────────────────────────────────────────────────────────────────────────


def _parse_rocm_smi(output: str, gpu_id: int = 0) -> tuple[int, int]:
    """解析 rocm-smi --showmeminfo vram 输出，返回 (total_bytes, used_bytes)。

    Raises:
        ValueError: 输出格式不符合预期。
    """
    total_b: int | None = None
    used_b: int | None = None
    # 匹配 GPU[0]          : VRAM Total Memory (B): 201761212416
    total_re = re.compile(rf"GPU\[{gpu_id}\]\s*:\s*VRAM Total Memory \(B\):\s*(\d+)")
    used_re = re.compile(rf"GPU\[{gpu_id}\]\s*:\s*VRAM Total Used Memory \(B\):\s*(\d+)")
    for line in output.splitlines():
        if m := total_re.search(line):
            total_b = int(m.group(1))
        if m := used_re.search(line):
            used_b = int(m.group(1))
    if total_b is None or used_b is None:
        msg = f"无法从 rocm-smi 输出解析显存（gpu_id={gpu_id}）：{output!r}"
        raise ValueError(msg)
    return total_b, used_b


def _parse_nvidia_smi(output: str) -> tuple[int, int]:
    """解析 nvidia-smi csv 输出，返回 (total_mb, used_mb)。"""
    line = output.strip().splitlines()[0] if output.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        msg = f"无法从 nvidia-smi 输出解析显存：{output!r}"
        raise ValueError(msg)
    return int(parts[0]), int(parts[1])


# ─────────────────────────────────────────────────────────────────────────────
# 工厂
# ─────────────────────────────────────────────────────────────────────────────


def make_gpu_provider(
    backend: GpuBackend = GpuBackend.AUTO,
    *,
    rocm_command: str | None = None,
    nvidia_command: str | None = None,
) -> GpuInfoProvider:
    """按 backend 创建 provider。

    AUTO 模式：检测系统上哪个命令可用（rocm-smi 优先，nvidia-smi 次之，都没有→NONE）。
    """
    if backend is GpuBackend.ROCM:
        return RocmSmiProvider()
    if backend is GpuBackend.NVIDIA:
        return NvidiaSmiProvider()
    if backend is GpuBackend.NONE:
        return NoneGpuProvider()
    # AUTO
    if shutil.which("rocm-smi"):
        return RocmSmiProvider()
    if shutil.which("nvidia-smi"):
        return NvidiaSmiProvider()
    return NoneGpuProvider()
