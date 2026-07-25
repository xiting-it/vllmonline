# 开发指南

> 本文档记录开发环境的所有依赖、工具链用法、踩坑点、贡献流程。

## 环境要求

| 工具 | 版本要求 | 用途 | 安装方式 |
|------|---------|------|---------|
| **Python** | 3.12.x（严格 ≥3.12,<3.13） | 运行时 | uv 自动下载 |
| **[uv](https://docs.astral.sh/uv/)** | ≥ 0.11 | 包管理 + 虚拟环境 | `curl -LsSf https://astral.sh/install.sh \| sh` |
| **Docker** | ≥ 24.0 | 跑集成测试 + 实机部署 | 系统包管理器 |
| **ROCm 驱动** | 7.2.1+ | MI300X GPU（仅实机） | [AMD 官方文档](https://rocm.docs.amd.com/) |
| **git** | ≥ 2.30 | 版本控制 | 系统自带 |

**无 GPU 也能完整开发**：所有单测在 CPU 上跑通，fake vLLM 模拟全部 vLLM 行为。

## 首次设置

```bash
git clone <repo-url> && cd vllmonline
make install              # uv sync（装全部依赖到 .venv/）
make precommit-install    # 装 git pre-commit hooks
```

## 常用命令

所有命令封装在 `Makefile`，统一入口：

```bash
make help              # 列出所有命令

# 环境
make install           # 装依赖（dev + test 组）
make upgrade           # 升级所有依赖
make requirements-txt  # 导出扁平 requirements.txt（给不用 uv 的环境）
make clean             # 清缓存

# 测试
make test              # 跑单测（不含 integration，~10s）
make test-all          # 跑全部（含 integration，需要 Docker）
make test-integration  # 仅集成测试（testcontainers PG）
make test-cov          # 强制覆盖率检查（≥85%）
make test-verbose      # 详细模式

# 代码质量
make lint              # ruff check
make format            # ruff format（自动改）
make typecheck         # mypy strict
make check             # 一键：lint + format-check + mypy

# 运行
make run               # 本地起服务（热重载）
make compose-up-dev    # Docker 全套（fake vLLM，无 GPU）
make compose-up        # Docker 全套（MI300X 实机）

# 数据库
make db-migrate        # 应用 alembic migration
make db-revision m="描述"  # 生成新 migration
make db-reset          # 重置数据库（开发用）
```

## 依赖说明

依赖分三层（PEP 621 + PEP 735，由 uv 管理）：

| 层 | 位置 | 用途 |
|---|---|---|
| 运行时 | `pyproject.toml` `[project.dependencies]` | 生产部署必需 |
| 可选 extras | `[project.optional-dependencies]` | 用户可选装（如 `[gpu]`） |
| 开发期 | `[dependency-groups]` `dev` / `test` | 仅开发，不进 wheel |

扁平 `requirements.txt`（含 hash，给不用 uv 的环境/CI reproducible build）：

```bash
make requirements-txt   # 重新生成
```

关键依赖版本（2026-07 调研）：

- **fastapi** 0.140 + **uvicorn** 0.51 + **httpx** 0.28
- **sqlalchemy** 2.0.51 + **asyncpg** 0.31 + **alembic** 1.16
- **redis** 8.0（注意：6.x 是另一商业包，开源版已跳到 8.x）
- **scipy** 1.18 + **numpy** 2.5
- **prometheus-client** 0.26 + **structlog** 25.x
- **pytest** 9.1 + **pytest-asyncio** 0.26 + **ruff** 0.16 + **mypy** 1.18

## 测试策略

| 类型 | 标记 | DB | GPU | vLLM | 耗时 |
|---|---|---|---|---|---|
| 单测 | 无 | SQLite in-mem | 无 | fake (ASGITransport) | ~10s |
| API 测试 | 无 | SQLite in-mem | 无 | fake | ~3s |
| 集成测试 | `@pytest.mark.integration` | testcontainers PG | 无 | fake | ~30s |

- **fake vLLM**（`tests/fake_vllm.py`）：完整模拟 OpenAI 兼容 API + `/sleep` `/wake_up` `/metrics`，SSE streaming 真实分块
- **SQLite StaticPool**：in-memory 模式必须共享单连接，否则每个连接看到不同 DB
- **asgi-lifespan**：驱动 FastAPI lifespan（httpx ASGITransport 默认不发 lifespan 事件）
- **全局单例清理**：`conftest.py` 的 autouse fixture 在每个测试前清理全局 registry/engine/routing_manager

覆盖率门槛：整体 ≥85%，核心模块（gpu_memory/lifecycle/statistics/strategy/metrics）≥90-95%。

## 项目结构

```
vllmonline/
├── vllmonline/                # 主包
│   ├── server.py              # FastAPI 入口 + 代理
│   ├── config.py              # pydantic-settings 配置
│   ├── gpu_info.py            # ROCm/NVIDIA GPU 抽象
│   ├── metrics.py             # Prometheus 指标
│   ├── scheduler/             # GPU 显存 + 状态机 + 热切换
│   ├── router/                # 代理路由 + drain + middleware
│   ├── canary/                # 灰度策略 + 回滚 + metrics 采集
│   ├── eval/                  # 统计 + LLM-Judge + 报告
│   ├── vllm/                  # vLLM client + adapter + metrics relabel
│   ├── db/                    # SQLAlchemy models + session
│   └── api/                   # REST schemas + routes
├── alembic/                   # 数据库 migration
├── deploy/docker-compose/     # 部署配置（含 Grafana/Prometheus）
├── tests/                     # 测试（含 fake vLLM）
├── scripts/                   # 压测脚本
├── pyproject.toml             # 依赖 + 工具配置
├── Makefile                   # 命令封装
├── SPEC.md / PLAN.md          # 规约 + 计划
└── requirements.txt           # 扁平依赖（导出生成）
```

## 贡献流程

1. 起分支：`git checkout -b feat/xxx`
2. 写代码 + 测试（保持覆盖率 ≥85%）
3. `make check` 全绿（ruff + mypy strict）
4. `make test` 全过
5. commit（遵循 [Conventional Commits](https://www.conventionalcommits.org/)）
6. PR

### Commit 约定

```
<type>(<scope>): <subject>

<body>
```

type：`feat`（新功能）/ `fix`（修复）/ `docs` / `refactor` / `test` / `chore`
scope：phase 或模块名（如 `P1` / `gpu_memory` / `api`）

## 已知问题 / 踩坑

### MI300X 特定

- **`cuda_context_overhead_gb`**：MI300X 上实际 4-5GB（HBM stack + ROCm runtime），SPEC 默认 2.0 适用于 NVIDIA。通过 `VLLMONLINE_GPU__CUDA_CONTEXT_OVERHEAD_GB=4.0` 覆盖。
- **vLLM 镜像**：用 `rocm/vllm:ubuntu22.04-rocm7.2.1-py312-torch2.9.1-1.36.3`，**不是** NVIDIA 的 `vllm/vllm-openai`。
- **设备挂载**：`--device /dev/kfd --device /dev/dri --group-add video`。
- **gfx942 原生支持**：MI300X 不需要 `HSA_OVERRIDE_GFX_VERSION`（仅消费级 RDNA3 卡需要）。
- **PyTorch TunableOp**：镜像默认开启，建议加 `PYTORCH_TUNABLEOP_MATMUL_ADD_BMM=1`（AMD 官方推荐）。

### 开发环境

- **httpx ASGITransport 不发 lifespan**：用 `asgi-lifespan` 的 `LifespanManager` 驱动。
- **SQLite in-memory 跨连接隔离**：必须 `poolclass=StaticPool` + `connect_args={"check_same_thread": False}`。
- **testcontainers 4.x**：用 `testcontainers.community.postgres`（老路径 `testcontainers.postgres` 抛 DeprecationWarning）。
- **FastAPI/Starlette 1.x 路由**：`APIRouter(prefix=...)` 在 `include_router` 时生效，`app.router.routes` 列表可能不实时反映（用 HTTP 请求验证）。

### SPEC 与实现的差异

- **`min_sample_size`**：SPEC §6.4 示例 d=0.5→64, d=0.2→394（用 Z≈1.96/0.84 近似）；实现用 scipy 精确分位数得 63/393。差异在 Z 值精度，数学上实现更准确。
- **显存精度**：SPEC §3.4 说"保留 1 位小数"，实现保留完整精度（避免累积误差），1 位小数仅用于显示。
