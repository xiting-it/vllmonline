"""vllm 子包：与 vLLM backend 的交互。

模块：
    client.py   OpenAI 兼容 API + 高级端点封装
    adapter.py  sleep/wake/load_lora（SPEC §7.1 + 已知问题 workaround）
    metrics.py  /metrics 解析 + relabel（P3 实现）
"""
