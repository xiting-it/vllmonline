#!/usr/bin/env python3
"""热切换期间 source rate 压测：持续发请求 + 中途触发回滚，验证零请求丢失。

用法（在 pod 里）：
    cd /mnt/workspace/vllmonline
    PYTHONPATH=. python scripts/bench_hotswap.py <deployment_id>
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field

import httpx

VLLMONLINE_URL = "http://localhost:8080"
DURATION_SECONDS = 30
CONCURRENCY = 20
ROLLBACK_AT_SECONDS = 15


@dataclass
class Stats:
    total: int = 0
    success: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)
    # 按"回滚前/后"分类（用绝对时间戳判断）
    results: list[tuple[float, bool, float]] = field(
        default_factory=list
    )  # (sent_at, success, latency_ms)


async def send_one_request(client: httpx.AsyncClient, prompt_id: int) -> tuple[bool, float]:
    """发一个请求，返回 (success, latency_ms)。"""
    t0 = time.monotonic()
    try:
        r = await client.post(
            f"{VLLMONLINE_URL}/v1/chat/completions",
            json={
                "model": "qwen-7b",
                "messages": [{"role": "user", "content": f"bench {prompt_id}"}],
                "max_tokens": 5,
                "temperature": 0.0,
            },
            timeout=30.0,
        )
        latency = (time.monotonic() - t0) * 1000
        return (r.is_success, latency)
    except Exception:
        latency = (time.monotonic() - t0) * 1000
        return (False, latency)


async def worker(
    client: httpx.AsyncClient,
    worker_id: int,
    stop_at: float,
    rollback_at: float,
    stats: Stats,
    counter: list[int],  # 共享计数器 [current_value]
) -> None:
    """每个 worker 独立循环发请求，直到 stop_at。"""
    while time.monotonic() < stop_at:
        # 取一个唯一 prompt_id
        pid = counter[0]
        counter[0] += 1

        sent_at = time.monotonic()
        success, latency = await send_one_request(client, pid)

        stats.total += 1
        if success:
            stats.success += 1
        else:
            stats.failed += 1
        stats.latencies.append(latency)
        stats.results.append((sent_at, success, latency))


async def trigger_rollback(client: httpx.AsyncClient, deployment_id: str) -> bool:
    try:
        r = await client.post(
            f"{VLLMONLINE_URL}/api/canary/{deployment_id}/rollback",
            params={"reason": "benchmark_triggered"},
            timeout=30.0,
        )
        return r.is_success
    except Exception:
        return False


async def main(deployment_id: str | None = None) -> None:
    print("=" * 70)
    print("vLLMonline 热切换 source rate 压测")
    print(
        f"  并发: {CONCURRENCY}   时长: {DURATION_SECONDS}s   回滚触发: 第 {ROLLBACK_AT_SECONDS}s"
    )
    print("=" * 70)

    stats = Stats()
    start = time.monotonic()
    stop_at = start + DURATION_SECONDS
    rollback_at = start + ROLLBACK_AT_SECONDS
    counter = [0]  # 共享 prompt 计数器

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

        # 启动 workers（每个独立循环）
        workers = [
            asyncio.create_task(worker(client, i, stop_at, rollback_at, stats, counter))
            for i in range(CONCURRENCY)
        ]

        # 后台触发回滚
        async def do_rollback():
            if deployment_id is None:
                return
            # 等到回滚时间点（轮询比 Event 简单）
            while time.monotonic() < rollback_at:  # noqa: ASYNC110
                await asyncio.sleep(0.1)
            print(f"\n>>> [{ROLLBACK_AT_SECONDS}s] 触发回滚 <<<")
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

        # 停止 workers
        await asyncio.gather(*workers, return_exceptions=True)
        await rollback_task

    # 统计
    actual_duration = time.monotonic() - start
    qps = stats.success / actual_duration if actual_duration > 0 else 0

    sorted_lat = sorted(stats.latencies) if stats.latencies else [0]
    avg_latency = sum(stats.latencies) / len(stats.latencies) if stats.latencies else 0
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0
    p99 = sorted_lat[min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)] if sorted_lat else 0

    print()
    print("=" * 70)
    print("压测结果")
    print("=" * 70)
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

    # 回滚前后对比
    if deployment_id:
        before = [(s, ok, lat) for s, ok, lat in stats.results if s < rollback_at]
        after = [(s, ok, lat) for s, ok, lat in stats.results if s >= rollback_at]
        b_succ = sum(1 for _, ok, _ in before if ok)
        a_succ = sum(1 for _, ok, _ in after if ok)

        print()
        print("=" * 70)
        print("回滚前后对比（验证零请求丢失）")
        print("=" * 70)
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
            print(f"⚠ 结论：热切换期间丢失 {stats.failed} 个请求")

    print()
    print("=" * 70)


if __name__ == "__main__":
    dep_id = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(dep_id))
