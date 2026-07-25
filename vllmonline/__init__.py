"""vLLMonline — vLLM 推理引擎上层管理平台。

特性：
- 零停机模型热切换（同一 GPU 上多个模型实例，drain + sleep/unload）
- 灰度发布（阶梯放量，统计驱动推进）
- A/B 自动评测（LLM-as-Judge + Welch's t-test）
- 劣化自动回滚

详见 SPEC.md。
"""

# 单一版本源：从 version.py 导入，避免在多处定义。
from vllmonline.version import __version__

__all__ = ["__version__"]
