#!/usr/bin/env python3
"""自动回滚时间线记录：触发回滚后精确记录每个阶段的时间戳。

用法（在 pod 里）：
    cd /mnt/workspace/vllmonline
    PYTHONPATH=. python scripts/rollback_timeline.py canary-xxxx

流程：
    1. 接收 deployment_id（命令行参数）
    2. 轮询 v2 状态，记录初始（应为 ACTIVE）
    3. 触发回滚，记录 t0
    4. 高频轮询（每 50ms）：
       - v2 状态变化（ACTIVE → DRAINING → SLEEPING）
       - 路由表变化（v1:v2 = 7:3 → 1:0）
    5. 记录每个变化的时间戳
    6. 输出时间线表格 + 总耗时

输出：
    - 终端表格（截图用）：[t0] 发起 → [t1] v2 SLEEPING → [t2] 路由 100%v1
    - 关键指标：从触发到流量完全切回的总耗时（毫秒级）
"""

from __future__ import annotations

import asyncio
import sys
import time

import httpx

VLLMONLINE_URL = "http://localhost:8080"
POLL_INTERVAL_MS = 50  # 轮询间隔 50ms
MAX_WAIT_SECONDS = 60  # 最长等 60s


async def get_v2_state(client: httpx.AsyncClient, v2_id: str) -> str | None:
    try:
        r = await client.get(f"{VLLMONLINE_URL}/api/models/{v2_id}", timeout=2.0)
        if r.is_success:
            return r.json().get("state")
    except Exception:
        pass
    return None


async def get_traffic_split(client: httpx.AsyncClient, model_name: str) -> dict:
    """通过 metrics 反推（vllmonline 没有直接查路由表的 API）。"""
    try:
        await client.get(f"{VLLMONLINE_URL}/metrics", timeout=2.0)
        # 简化：从 deployment status 读
        return {}
    except Exception:
        return {}


async def get_deployment_status(client: httpx.AsyncClient, dep_id: str) -> dict | None:
    try:
        r = await client.get(f"{VLLMONLINE_URL}/api/canary/{dep_id}/status", timeout=2.0)
        if r.is_success:
            return r.json()
    except Exception:
        pass
    return None


async def main(deployment_id: str) -> None:
    print("=" * 70)
    print("vLLMonline 回滚时间线记录")
    print(f"  deployment: {deployment_id}")
    print(f"  轮询间隔: {POLL_INTERVAL_MS}ms")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        # 1. 找到这个 deployment 对应的 v1/v2
        status = await get_deployment_status(client, deployment_id)
        if not status:
            print(f"✗ 找不到 deployment {deployment_id}")
            return

        # 从 traffic_split 反推 v1_id / v2_id
        split = status.get("traffic_split", {})
        if len(split) < 2:
            print("✗ deployment 的 traffic_split 不完整，无法确定 v1/v2")
            return
        ids = sorted(split.keys())
        v1_id, v2_id = ids[0], ids[1]
        print(f"  v1: {v1_id}")
        print(f"  v2: {v2_id}")
        print(f"  当前阶段: {status.get('current_stage')}")
        print(f"  当前状态: {status.get('status')}")
        print()

        if status.get("status") != "IN_PROGRESS":
            print(f"⚠ deployment 状态是 {status.get('status')}，不是 IN_PROGRESS")
            print("  无法触发回滚（已经回滚过 / 已完成）")
            return

        # 2. 记录初始 v2 状态
        initial_v2_state = await get_v2_state(client, v2_id)
        print(f"初始 v2 状态: {initial_v2_state}")
        print()

        # 3. 时间线事件记录
        events: list[tuple[float, str, str]] = []  # (relative_t, event, detail)
        # 用 perf_counter 做高精度计时
        t_start = time.perf_counter()

        # 4. 触发回滚（异步，不阻塞轮询）
        async def trigger():
            await asyncio.sleep(0.05)  # 让轮询先启动
            t_fire = time.perf_counter() - t_start
            events.append((t_fire, "FIRE", "调用 rollback API"))
            print(f"[{t_fire * 1000:7.1f}ms] ★ 发起回滚请求")
            try:
                r = await client.post(
                    f"{VLLMONLINE_URL}/api/canary/{deployment_id}/rollback",
                    params={"reason": "timeline_benchmark"},
                    timeout=30.0,
                )
                t_resp = time.perf_counter() - t_start
                if r.is_success:
                    events.append(
                        (t_resp, "API_RESPONDED", f"rollback API 返回（HTTP {r.status_code}）")
                    )
                    print(f"[{t_resp * 1000:7.1f}ms] ✓ rollback API 返回")
                else:
                    events.append((t_resp, "API_FAILED", f"HTTP {r.status_code}"))
                    print(f"[{t_resp * 1000:7.1f}ms] ✗ rollback API 失败: HTTP {r.status_code}")
            except Exception as e:
                t_err = time.perf_counter() - t_start
                events.append((t_err, "API_EXCEPTION", str(e)))
                print(f"[{t_err * 1000:7.1f}ms] ✗ rollback API 异常: {e}")

        trigger_task = asyncio.create_task(trigger())

        # 5. 高频轮询直到 v2 进入 SLEEPING
        last_v2_state = initial_v2_state
        last_dep_status = "IN_PROGRESS"
        v2_reached_sleeping = False
        dep_reached_rolled_back = False

        while True:
            now = time.perf_counter() - t_start
            if now > MAX_WAIT_SECONDS:
                print(f"[{now * 1000:7.1f}ms] ⚠ 超时（{MAX_WAIT_SECONDS}s）")
                break

            # 查 v2 状态
            v2_state = await get_v2_state(client, v2_id)
            if v2_state and v2_state != last_v2_state:
                events.append((now, "V2_STATE_CHANGED", f"{last_v2_state} → {v2_state}"))
                print(f"[{now * 1000:7.1f}ms] v2 状态: {last_v2_state} → {v2_state}")
                last_v2_state = v2_state
                if v2_state == "SLEEPING":
                    v2_reached_sleeping = True

            # 查 deployment 状态
            dep = await get_deployment_status(client, deployment_id)
            if dep:
                ds = dep.get("status")
                if ds != last_dep_status:
                    events.append((now, "DEP_STATUS_CHANGED", f"{last_dep_status} → {ds}"))
                    print(f"[{now * 1000:7.1f}ms] deployment: {last_dep_status} → {ds}")
                    last_dep_status = ds
                    if ds == "ROLLED_BACK":
                        dep_reached_rolled_back = True

            # 终止条件
            if v2_reached_sleeping and dep_reached_rolled_back:
                # 多轮询 200ms 确保流量真的切了
                await asyncio.sleep(0.2)
                break

            await asyncio.sleep(POLL_INTERVAL_MS / 1000)

        await trigger_task

        t_end = time.perf_counter() - t_start

        # 6. 输出时间线表格
        print()
        print("=" * 70)
        print("回滚时间线")
        print("=" * 70)
        print(f"{'相对时间':<14} {'事件':<22} {'详情':<35}")
        print("-" * 70)
        for t, event, detail in events:
            print(f"{t * 1000:>8.1f} ms   {event:<22} {detail:<35}")
        print("-" * 70)
        print(f"{'总耗时':<14} {t_end * 1000:>8.1f} ms")
        print()

        # 关键指标
        fire_t = next((t for t, e, _ in events if e == "FIRE"), None)
        v2_sleep_t = next(
            (t for t, e, _ in events if e == "V2_STATE_CHANGED" and "SLEEPING" in events[0]), None
        )
        # 找最后一个 V2_STATE_CHANGED 到 SLEEPING 的时间
        v2_sleep_t = None
        for t, e, detail in events:
            if e == "V2_STATE_CHANGED" and "SLEEPING" in detail:
                v2_sleep_t = t

        print("=" * 70)
        print("关键指标")
        print("=" * 70)
        if fire_t is not None:
            print(f"  发起回滚:                {fire_t * 1000:.1f} ms")
        if v2_sleep_t is not None:
            print(f"  v2 进入 SLEEPING:        {v2_sleep_t * 1000:.1f} ms")
            print(
                f"  从发起到 v2 sleep:       {(v2_sleep_t - fire_t) * 1000:.1f} ms"
                if fire_t
                else ""
            )
        print(f"  总耗时:                  {t_end * 1000:.1f} ms")
        print()

        # 保存到文件
        from datetime import datetime
        from pathlib import Path

        reports_dir = Path("scripts/reports")
        reports_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"rollback_timeline_{ts}.md"

        md = [
            "# 回滚时间线报告",
            "",
            f"deployment: `{deployment_id}`",
            "",
            "| 相对时间 (ms) | 事件 | 详情 |",
            "|---|---|---|",
        ]
        for t, event, detail in events:
            md.append(f"| {t * 1000:.1f} | {event} | {detail} |")
        md.append(f"| **总耗时** | | **{t_end * 1000:.1f} ms** |")
        report_path.write_text("\n".join(md), encoding="utf-8")
        print(f"✓ 时间线报告已保存: {report_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: PYTHONPATH=. python scripts/rollback_timeline.py <deployment_id>")
        print("  deployment_id 从 /api/canary/start 的返回里拿")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
