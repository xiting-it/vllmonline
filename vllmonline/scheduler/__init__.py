"""调度器子包：GPU 显存管理 + 模型状态机 + 热切换编排。

模块：
    types.py       枚举（ModelState, LoadStrategy）+ dataclass（数据契约）
    gpu_memory.py  显存计算（权重/KV cache）+ can_load 决策树
    lifecycle.py   Model + ModelRegistry + 状态转移
    engine.py      热切换编排（hot_swap，P2 实现）
"""
