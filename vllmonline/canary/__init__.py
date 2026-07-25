"""canary 子包：灰度发布策略 + metrics 采集 + 自动回滚。

模块：
    strategy.py          灰度状态机 + should_advance 推进判定（SPEC §5.3-5.5）
    metrics_collector.py 从 Prometheus 拉 per-version metrics（P3 后续）
    rollback.py          劣化检测 + 自动回滚（P5）
"""
