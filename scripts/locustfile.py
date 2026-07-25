"""Locust 压测脚本（PLAN P6.3）。

模拟多用户并发请求 vllmonline 代理，测试：
    - 单版本最大 QPS
    - 热切换期间的成功率（应 100%）
    - 灰度期间 v1/v2 流量分布

用法：
    pip install locust
    locust -f scripts/locustfile.py --host http://localhost:8080

然后浏览器打开 http://localhost:8089 配置并发数。
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

PROMPTS = [
    "什么是人工智能？",
    "解释机器学习的基本概念。",
    "Python 和 Java 的区别？",
    "写一个快速排序的实现。",
    "总结《三体》的核心思想。",
    "如何优化数据库查询性能？",
    "解释 CAP 定理。",
    "REST 和 GraphQL 的取舍？",
]


class VllmonlineUser(HttpUser):
    """模拟一个客户端用户。"""

    wait_time = between(0.5, 2.0)  # 每次请求间隔

    @task(10)
    def chat_nonstream(self) -> None:
        """非 streaming chat（权重 10）。"""
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen-7b",
                "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
                "max_tokens": 64,
                "temperature": 0.7,
                "stream": False,
            },
        )

    @task(3)
    def chat_stream(self) -> None:
        """streaming chat（权重 3）。"""
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen-7b",
                "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
                "max_tokens": 32,
                "stream": True,
            },
            stream=True,
            catch_response=True,
        ) as resp:
            # 消费完整个流，否则连接不释放
            chunks = 0
            for line in resp.iter_lines():
                if line and line.strip().startswith("data: "):
                    payload = line.strip()[6:]
                    if payload == "[DONE]":
                        break
                    chunks += 1
            if chunks == 0:
                resp.failure("stream 返回 0 chunks")
            else:
                resp.success()

    @task(1)
    def health_check(self) -> None:
        """健康检查（权重 1）。"""
        self.client.get("/healthz")


class CanaryDriver(HttpUser):
    """灰度驱动：周期性查询灰度状态（独立用户类，低并发）。

    用法：locust -f scripts/locustfile.py --host http://localhost:8080 \
              --class-locust VllmonlineUser 50 --class-locust CanaryDriver 1
    """

    wait_time = between(5, 10)
    weight = 0  # 默认不跑，需手动指定

    @task
    def check_canary(self) -> None:
        """查询灰度状态（需先有 deployment）。"""
        # 这里只做示例——真实需先知道 deployment_id
        self.client.get("/api/models")


# 热切换压测：在持续请求期间触发 hot_swap，验证零丢失。
# 用法：locust -f scripts/locustfile.py --host http://localhost:8080 \
#          --headless -u 100 -r 10 -t 60s
# 同时另开终端：curl -X POST :8080/api/canary/start ...
# 检查 Locust 报告的 failure rate 应为 0%。
