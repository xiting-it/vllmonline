#!/usr/bin/env python3
"""热切换期间 source rate 压测：持续发请求 + 中途触发回滚，验证零请求丢失。

用法（在 pod 里，vllmonline + 两个 vLLM 都跑着）：
    cd /mnt/workspace/vllmonline
    PYTHONPATH=. python scripts/bench_hotswap.py

流程：
    1. 先确保 v1 + v2 都 ACTIVE + 有一个进行中的灰度部署
    2. 后台持续发请求（20 并发，30 秒，目标 QPS ~50）
    3. 跑到 50% 时触发手动回滚
    4. 结束后统计：总请求 / 成功 / 失败 / 实际 QPS / 回滚前后成功率对比

输出：
    - 终端实时进度
    - 最终统计表格（截图用）
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

VLLMONLINE_URL = "http://localhost:8080"
DURATION_SECONDS = 30  # 总压测时长
CONCURRENCY = 20  # 并发数
ROLLBACK_AT_SECONDS = 15  # 第 15 秒触发回滚
TARGET_QPS_INFO = "理论 QPS ≈ 并发数 × 单请求速度（vLLM 推理是瓶颈）"


@dataclass
class RequestResult:
    sent_at: float
    success: bool
    latency_ms: float
    model_version: str | None = None
    error: str | None = None


@dataclass
class Stats:
    total: int = 0
    success: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)
    before_rollback: list[RequestResult] = field(default_factory=list)
    after_rollback: list[RequestResult] = field(default_factory=list)
    versions: dict[str, int] = field(default_factory=dict)


async def find_active_deployment(client: httpx.AsyncClient) -> str | None:
    """找一个 IN_PROGRESS 的灰度部署。"""
    # vllmonline 没有列出全部部署的 API，我们试着从已知 deployment 找
    # 简化：让用户从命令行传，或者用最新的一个
    return None  # 后面用环境变量或参数


async def trigger_rollback(client: httpx.AsyncClient, deployment_id: str) -> bool:
    """触发回滚。"""
    try:
        r = await client.post(
            f"{VLLMONLINE_URL}/api/canary/{deployment_id}/rollback",
            params={"reason": "benchmark_triggered_rollback"},
            timeout=10.0,
        )
        return r.is_success
    except Exception:
        return False


async def send_one_request(
    client: httpx.AsyncClient,
    prompt_id: int,
) -> RequestResult:
    """发一个请求，返回结果。"""
    t0 = time.monotonic()
    try:
        r = await client.post(
            f"{VLLMONLINE_URL}/v1/chat/completions",
            json={
                "model": "qwen-7b",
                "messages": [{"role": "user", "content": f"bench {prompt_id}"}],
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=30.0,
        )
        latency = (time.monotonic() - t0) * 1000
        if r.is_success:
            return RequestResult(t0, True, latency)
        return RequestResult(t0, False, latency, error=f"http_{r.status_code}")
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return RequestResult(t0, False, latency, error=type(e).__name__)


async def worker(
    client: httpx.AsyncClient,
    stop_at: float,
    prompt_counter: AsyncIterator[int],
    stats: Stats,
    rollback_at: float,
) -> None:
    async for pid in prompt_counter:
        if time.monotonic() >= stop_at:
            return
        result = await send_one_request(client, pid)
        stats.total += 1
        if result.success:
            stats.success += 1
        else:
            stats.failed += 1
        stats.latencies.append(result.latency)
        if time.monotonic() < rollback_at:
            stats.before_rollback.append(result)
        else:
            stats.after_rollback.append(result)


async def prompt_generator() -> AsyncIterator[int]:
    n = 0
    while True:
        yield n
        n += 1


async def main(deployment_id: str | None = None) -> None:
    print("=" * 70)
    print("vLLMonline 热切换 source rate 压测")
    print(
        f"  并发: {CONCURRENCY}   时长: {DURATION_SECONDS}s   回滚触发: 第 {ROLLBACK_AT_SECONDS}s"
    )
    print(f"  {TARGET_QPS_INFO}")
    print("=" * 70)

    if not deployment_id:
        print("\n⚠ 未传 deployment_id，跳过自动回滚（仅压测，不触发切换）")
        print("  用法: PYTHONPATH=. python scripts/bench_hotswap.py canary-xxxx")
        print("  你可以从 /api/canary/start 的返回里拿到 id")
        rollback_at = float("inf")
    else:
        print(f"\n✓ 将在第 {ROLLBACK_AT_SECONDS}s 回滚部署 {deployment_id}")
        rollback_at = time.monotonic() + ROLLBACK_AT_SECONDS

    stats = Stats()
    start = time.monotonic()
    stop_at = start + DURATION_SECONDS

    async with httpx.AsyncClient() as client:
        # 健康检查
        try:
            h = await client.get(f"{VLLMONLINE_URL}/healthz", timeout=3.0)
            if not h.is_success:
                print("✗ vllmonline 不可达")
                return
        except Exception as e:
            print(f"✗ vllmonline 不可达: {e}")
            return

        print("\n开始压测...\n")

        # 启动 workers
        gen = prompt_generator()
        workers = [
            asyncio.create_task(worker(client, stop_at, gen, stats, rollback_at))
            for _ in range(CONCURRENCY)
        ]

        # 后台触发回滚
        async def do_rollback():
            if deployment_id is None:
                return
            await asyncio.sleep(ROLLBACK_AT_SECONDS)
            print(f"\n>>> [{ROLLBACK_AT_SECONDS}s] 触发回滚 <<<\n")
            ok = await trigger_rollback(client, deployment_id)
            print(f">>> 回滚结果: {'成功' if ok else '失败'} <<<\n")

        rollback_task = asyncio.create_task(do_rollback())

        # 进度显示
        while time.monotonic() < stop_at:
            await asyncio.sleep(2.0)
            elapsed = time.monotonic() - start
            print(
                f"  [{elapsed:5.1f}s] 总请求={stats.total}  成功={stats.success}  失败={stats.failed}"
            )

        # 停止
        await rollback_task
        await asyncio.gather(*workers, return_exceptions=True)

    # 统计
    print()
    print("=" * 70)
    print("压测结果")
    print("=" * 70)
    actual_duration = time.monotonic() - start
    qps = stats.success / actual_duration if actual_duration > 0 else 0

    avg_latency = sum(stats.latencies) / len(stats.latencies) if stats.latencies else 0
    sorted_lat = sorted(stats.latencies) if stats.latencies else [0]
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else 0

    print(f"{'指标':<25} {'值':<20}")
    print("-" * 50)
    print(f"{'总请求数':<25} {stats.total:<20}")
    print(f"{'成功请求数':<25} {stats.success:<20}")
    print(f"{'失败请求数':<25} {stats.failed:<20}")
    print(f"{'成功率':<25} {stats.success / max(1, stats.total) * 100:.2f}%")
    print(f"{'实际 QPS':<25} {qps:.2f}")
    print(f"{'平均延迟 (ms)':<25} {avg_latency:.1f}")
    print(f"{'P50 延迟 (ms)':<25} {p50:.1f}")
    print(f"{'P99 延迟 (ms)':<25} {p99:.1f}")
    print(f"{'压测时长 (s)':<25} {actual_duration:.1f}")

    # 回滚前后对比（关键）
    if deployment_id:
        print()
        print("=" * 70)
        print("回滚前后对比（验证零请求丢失）")
        print("=" * 70)
        before = stats.before_rollback
        after = stats.after_rollback
        b_succ = sum(1 for r in before if r.success)
        a_succ = sum(1 for r in after if r.success)
        print(f"{'阶段':<15} {'请求数':<10} {'成功':<10} {'失败':<10} {'成功率':<10}")
        print("-" * 55)
        print(
            f"{'回滚前':<15} {len(before):<10} {b_succ:<10} {len(before) - b_succ:<10} {b_succ / max(1, len(before)) * 100:.1f}%"
        )
        print(
            f"{'回滚后':<15} {len(after):<10} {a_succ:<10} {len(after) - a_succ:<10} {a_succ / max(1, len(after)) * 100:.1f}%"
        )
        print()
        if stats.failed == 0:
            print("✓ 结论：热切换期间零请求丢失")
        else:
            print(f"⚠ 结论：热切换期间丢失 {stats.failed} 个请求（需排查）")

    print()
    print("=" * 70)


if __name__ == "__main__":
    dep_id = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(dep_id))
