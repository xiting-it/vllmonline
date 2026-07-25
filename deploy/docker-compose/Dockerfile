# vLLMonline 容器镜像
# 多阶段构建：builder 装依赖（uv），runtime 用精简 Python 镜像。

# ─── builder ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# 装 uv（uv 是单二进制，不需要 Python 环境污染）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 先只拷依赖清单，利用 docker 层缓存
COPY pyproject.toml uv.lock ./

# 装依赖到虚拟环境（--frozen 严格按 lock 文件，不解析）
RUN uv sync --frozen --no-install-project --no-dev

# 再拷源码
COPY vllmonline/ ./vllmonline/
COPY README.md ./

# 装项目本身
RUN uv sync --frozen --no-dev

# ─── runtime ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# 装运行时系统依赖（curl 用于 healthcheck）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 拷整个虚拟环境
COPY --from=builder /app /app

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV VLLMONLINE_ENVIRONMENT=production

EXPOSE 8080

# 直接用 uvicorn 启动（容器内单进程；多 worker 由 docker scale 或 k8s 管）
CMD ["uvicorn", "vllmonline.server:app", "--host", "0.0.0.0", "--port", "8080"]

# Healthcheck：检查 /healthz
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:8080/healthz || exit 1
