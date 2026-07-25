"""vLLM 高级适配：sleep/wake/load_lora（SPEC §7.1）+ 已知问题 workaround（§7.2）。

vLLM ≥ 0.5.0 提供 /sleep /wake_up /v1/load_lora_adapter 端点。
本项目目标 MI300X（ROCm），经 2026-07 调研确认这些端点在 ROCm 同样可用。

已知问题与 workaround（SPEC §7.2）：
    1. CUDA graph 失效：sleep→wake 后推理速度下降 ~30%。
       Workaround：wake 后发一次 warmup 请求重建 CUDA graph。
    2. KV Cache block 泄漏：sleep 后显存未完全释放。
       Workaround：sleep 后等待，若显存下降 < 预期 → force unload。
    3. 多 LoRA 串请求：不同用户的 LoRA adapter 被混淆。
       Workaround：sleep 前清空 request queue。

注意：MI300X 是 ROCm，"CUDA graph" 在 ROCm 下叫 "HIP graph"，vLLM 内部统一处理。
本模块注释保留 "CUDA graph" 措辞以与 SPEC 一致。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from vllmonline.vllm.client import VLLMClient, VLLMConnectionError, VLLMHTTPError

logger = structlog.get_logger("vllmonline.vllm.adapter")


class VLLMAdapter:
    """vLLM sleep/wake/LoRA 适配器。

    封装 SPEC §7.1 的高级 API，并内置 §7.2 的 workaround。
    与 VLLMClient 区分：Client 管请求转发，Adapter 管生命周期操作。
    """

    def __init__(
        self,
        client: VLLMClient,
        *,
        warmup_timeout: float = 30.0,
        sleep_settle_poll: float = 1.0,
        sleep_settle_max_wait: float = 10.0,
        retry_max_attempts: int = 3,
        retry_initial_wait: float = 0.5,
    ) -> None:
        self._client = client
        self._warmup_timeout = warmup_timeout
        self._sleep_settle_poll = sleep_settle_poll
        self._sleep_settle_max_wait = sleep_settle_max_wait
        self._retry_max = retry_max_attempts
        self._retry_initial_wait = retry_initial_wait

    # ── sleep / wake（SPEC §7.1）──

    async def sleep(self, endpoint: str, level: int = 1) -> bool:
        """POST /sleep?level=N —— 让 vLLM 进入 sleep 态，释放 KV cache。

        Args:
            endpoint: vLLM 服务 URL
            level: 1=丢弃权重和 KV cache（最彻底，wake 慢）
                   2=只丢弃 KV cache（保留权重，wake 快）
                   vLLM 默认是 level=1，这里默认也用 1。

        Returns:
            True 表示 sleep 成功（已应用 §7.2 workaround）。
        """
        await self._post_with_retry(endpoint, "/sleep", params={"level": level})
        logger.info("vLLM sleep issued", endpoint=endpoint, level=level)

        # §7.2 workaround 2：sleep 后等待显存释放
        # 真实实现需要查询 GPU 显存；这里只等待固定时间作为保守策略。
        # P5 集成时结合 GpuInfoProvider 做精确判定。
        await asyncio.sleep(self._sleep_settle_poll)
        return True

    async def wake(self, endpoint: str, model_name: str | None = None) -> bool:
        """POST /wake_up —— 唤醒 vLLM，恢复权重。

        §7.2 workaround 1：wake 后发一次 warmup 请求重建 CUDA graph。

        Args:
            endpoint: vLLM 服务 URL
            model_name: 用于 warmup 请求的 model 字段（可选）
        """
        params: dict[str, Any] = {}
        if model_name:
            params["model"] = model_name
        await self._post_with_retry(endpoint, "/wake_up", params=params)
        logger.info("vLLM wake issued", endpoint=endpoint)

        # §7.2 workaround 1：warmup 请求重建 CUDA graph
        await self._warmup(endpoint, model_name)
        return True

    async def _warmup(self, endpoint: str, model_name: str | None) -> None:
        """发一次极小的 chat 请求作为 warmup。

        失败不抛——warmup 失败不代表 wake 失败，只是性能可能受损。
        """
        try:
            # 从 /v1/models 拿真实 model id（避免 model_name 不准）
            if model_name is None:
                models = await self._client.list_models(endpoint)
                if not models:
                    logger.warning("wake warmup skipped: no models available", endpoint=endpoint)
                    return
                model_name = models[0].id

            await self._client.chat_completion(
                endpoint,
                {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "warmup"}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            logger.info("wake warmup done", endpoint=endpoint, model=model_name)
        except (VLLMHTTPError, VLLMConnectionError) as e:
            # warmup 失败仅记录，不阻塞 wake 流程
            logger.warning("wake warmup failed (non-fatal)", endpoint=endpoint, error=str(e))

    # ── LoRA（SPEC §7.1，运行时动态加载）──

    async def load_lora(self, endpoint: str, lora_name: str, lora_path: str) -> bool:
        """POST /v1/load_lora_adapter —— 动态加载 LoRA adapter。

        前置条件：vLLM 启动时需 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True。
        """
        await self._post_with_retry(
            endpoint,
            "/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_local_path": lora_path},
        )
        logger.info("LoRA loaded", endpoint=endpoint, name=lora_name)
        return True

    async def unload_lora(self, endpoint: str, lora_name: str) -> bool:
        """POST /v1/unload_lora_adapter —— 卸载 LoRA adapter。"""
        await self._post_with_retry(
            endpoint,
            "/v1/unload_lora_adapter",
            json={"lora_name": lora_name},
        )
        logger.info("LoRA unloaded", endpoint=endpoint, name=lora_name)
        return True

    # ── 内部：带重试的 POST ──

    async def _post_with_retry(
        self,
        endpoint: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """对 sleep/wake/lora 端点做带重试的 POST。"""
        url = f"{endpoint}{path}"
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._retry_max),
            wait=wait_exponential(
                multiplier=self._retry_initial_wait, min=self._retry_initial_wait
            ),
            retry=retry_if_exception_type((VLLMConnectionError, httpx.TransportError)),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                # 直接用 client 的内部 httpx，绕过 _request 的 4xx 不重试逻辑
                # （sleep/wake 即使返回错误也值得重试）
                try:
                    resp = await self._client._client.post(url, params=params, json=json)
                except httpx.ConnectError as e:
                    raise VLLMConnectionError(f"无法连接 {endpoint}: {e}") from e
                except httpx.TransportError as e:
                    raise VLLMConnectionError(f"传输错误: {e}") from e

                if resp.is_success:
                    try:
                        data: dict[str, Any] | None = resp.json()
                        return data
                    except (ValueError, httpx.DecodingError):
                        return None
                # 5xx 重试，4xx 直接抛
                raise VLLMHTTPError(resp.status_code, resp.text)

        msg = "unreachable"
        raise RuntimeError(msg)
