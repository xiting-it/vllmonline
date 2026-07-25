"""测试用的 fake vLLM server。

目标：在没有真实 GPU/vLLM 的开发/CI 环境下，提供与 vLLM OpenAI 兼容服务
**完全相同**的 HTTP 契约，让被测代码（vllmonline.proxy / vllm.client）的
行为与生产一致。

实现的端点：
    - POST /v1/chat/completions        （支持 stream:true 的 SSE）
    - POST /v1/completions              （text completion）
    - GET  /v1/models
    - GET  /health                      （存活）
    - POST /sleep                       （SPEC §7.1 sleep mode）
    - POST /wake_up
    - GET  /metrics                     （Prometheus text format，带 model_name label）

行为特征：
    - 响应内容按 prompt 哈希派生（让相同输入产生稳定输出，便于断言）
    - 可注入人工延迟（模拟 TTFT / TPOT）
    - 可注入故障率（模拟 backend 不稳定）
    - sleep 状态下 /v1/chat/completions 返回 503

通过 httpx.ASGITransport 把被测 client 指向本 app，零网络、零端口、零 GPU。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse


def make_fake_vllm_app(
    *,
    model_name: str = "fake-model",
    ttft_seconds: float = 0.0,
    tpot_seconds: float = 0.0,
    error_rate: float = 0.0,
    sleep_initial: bool = False,
) -> FastAPI:
    """构造一个 fake vLLM FastAPI app。

    Args:
        model_name: GET /v1/models 返回的模型名
        ttft_seconds: 首 token 延迟（模拟）
        tpot_seconds: 每 token 延迟（模拟）
        error_rate: [0,1] 请求失败概率（模拟 backend 不稳定）
        sleep_initial: 启动时是否处于 sleep 状态
    """
    app = FastAPI(title="fake-vllm")
    # 把可变状态挂到 app.state（测试期通过 app.state.xxx 修改）
    app.state.model_name = model_name
    app.state.ttft_seconds = ttft_seconds
    app.state.tpot_seconds = tpot_seconds
    app.state.error_rate = error_rate
    app.state.is_sleeping = sleep_initial
    app.state.sleep_call_count = 0
    app.state.wake_call_count = 0
    app.state.request_count = 0

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "fake",
                }
            ],
        }

    def _maybe_fail() -> bool:
        """按 error_rate 决定本次请求是否失败。"""
        if app.state.error_rate <= 0:
            return False
        # 用时间戳做伪随机（避免引入 random 全局状态影响测试可重复性）
        h = hashlib.sha256(f"{time.time_ns()}".encode()).digest()
        return (h[0] / 255) < app.state.error_rate

    def _derive_text(prompt: str, n_tokens: int = 16) -> list[str]:
        """从 prompt 派生稳定 token 序列（相同输入相同输出）。"""
        h = hashlib.sha256(prompt.encode()).digest()
        words = []
        for i in range(n_tokens):
            b = h[(i * 4) : (i * 4 + 4)]
            words.append(f"tok{int.from_bytes(b, 'big') % 10000}")
        return words

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        app.state.request_count += 1

        if app.state.is_sleeping:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "model is sleeping", "type": "sleep_mode"}},
            )

        if _maybe_fail():
            return JSONResponse(
                status_code=500,
                content={"error": {"message": "injected failure", "type": "fake_error"}},
            )

        body = await request.json()
        prompt = _extract_prompt(body)
        stream = bool(body.get("stream", False))
        n = max(1, body.get("n", 1))
        max_tokens = min(body.get("max_tokens", 16), 64)

        model = body.get("model", app.state.model_name)
        completion_id = f"chatcmpl-fake-{app.state.request_count:08d}"
        created = int(time.time())

        if not stream:
            choices = []
            for i in range(n):
                text = " ".join(_derive_text(f"{prompt}-{i}", max_tokens))
                choices.append(
                    {
                        "index": i,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                )
            return JSONResponse(
                content={
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": choices,
                    "usage": {
                        "prompt_tokens": max(1, len(prompt.split())),
                        "completion_tokens": max_tokens * n,
                        "total_tokens": max(1, len(prompt.split())) + max_tokens * n,
                    },
                }
            )

        # streaming：逐 chunk SSE
        return StreamingResponse(
            _stream_chunks(prompt, model, completion_id, created, n, max_tokens, app),
            media_type="text/event-stream",
        )

    async def _stream_chunks(
        prompt: str,
        model: str,
        cid: str,
        created: int,
        n: int,
        max_tokens: int,
        app: FastAPI,
    ) -> AsyncIterator[bytes]:
        # TTFT
        if app.state.ttft_seconds > 0:
            await asyncio.sleep(app.state.ttft_seconds)

        for i in range(n):
            # 首个 chunk：role
            yield _sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": i, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
            )
            # content chunks
            tokens = _derive_text(f"{prompt}-{i}", max_tokens)
            for tok in tokens:
                if app.state.tpot_seconds > 0:
                    await asyncio.sleep(app.state.tpot_seconds)
                yield _sse(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": i, "delta": {"content": f"{tok} "}, "finish_reason": None}
                        ],
                    }
                )
            # 终止 chunk
            yield _sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": i, "delta": {}, "finish_reason": "stop"}],
                }
            )
        yield b"data: [DONE]\n\n"

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        body = await request.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)
        text = " ".join(_derive_text(str(prompt), 16))
        return JSONResponse(
            content={
                "id": "cmpl-fake",
                "object": "text_completion",
                "created": int(time.time()),
                "model": body.get("model", app.state.model_name),
                "choices": [{"text": text, "finish_reason": "stop", "index": 0}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 16, "total_tokens": 20},
            }
        )

    @app.post("/sleep")
    async def sleep() -> dict[str, Any]:
        app.state.sleep_call_count += 1
        app.state.is_sleeping = True
        return {"status": "ok", "level": 1}

    @app.post("/wake_up")
    async def wake_up() -> dict[str, Any]:
        app.state.wake_call_count += 1
        app.state.is_sleeping = False
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        # 模拟 vLLM 的 /metrics 输出（带 model_name label，无 instance label，SPEC §7.3）
        text = (
            "# HELP vllm:request_success Total successful requests.\n"
            "# TYPE vllm:request_success counter\n"
            f'vllm:request_success{{model_name="{app.state.model_name}"}} {app.state.request_count}\n'
            "# HELP vllm:time_to_first_token_seconds TTFT\n"
            "# TYPE vllm:time_to_first_token_seconds histogram\n"
            f'vllm:time_to_first_token_seconds_bucket{{model_name="{app.state.model_name}",le="0.05"}} 1\n'
            f'vllm:time_to_first_token_seconds_bucket{{model_name="{app.state.model_name}",le="+Inf"}} 1\n'
            f'vllm:time_to_first_token_seconds_count{{model_name="{app.state.model_name}"}} 1\n'
            f'vllm:time_to_first_token_seconds_sum{{model_name="{app.state.model_name}"}} 0.02\n'
        )
        return PlainTextResponse(content=text, media_type="text/plain; version=0.0.4")

    return app


def _extract_prompt(body: dict[str, Any]) -> str:
    """从 chat 请求体提取 prompt 字符串（messages 拼接）。"""
    messages = body.get("messages", [])
    if isinstance(messages, list):
        return " ".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in messages)
    return str(messages)


def _sse(obj: dict[str, Any]) -> bytes:
    """把 dict 序列化为 SSE data 行。"""
    import json

    return f"data: {json.dumps(obj)}\n\n".encode()
