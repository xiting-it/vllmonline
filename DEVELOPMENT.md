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

### 实测发现的 bug

#### Bug：rollback 时 v2 不在内存 registry，导致状态不同步

**现象**

MI300X pod 上实测回滚时间线时，rollback API 返回 200，但随后 `/api/models` 仍显示 v2 为 ACTIVE（预期应为 SLEEPING）。DB 与内存状态不一致。

**根因**

进程重启后内存 `ModelRegistry` 为空，但 DB 仍有数据。`RollbackExecutor` 走 `if v2_id in self._registry` 分支时跳过了 drain+sleep，导致 DB 状态没更新。

更深层问题：`/api/models` 读 DB（返回 ACTIVE），`/api/canary/start` 读内存 registry（返回 404）——两个 API 对同一 model 表现不一致，排查时极易误判。

**修复**

- 无论内存 registry 是否有 v2，强制把 DB 状态标成 SLEEPING
- 新增单测 `test_execute_rollback_v2_not_in_registry_force_db_sleeping` 复现此场景，防止回归

**根本解法（待做）**

vllmonline 启动时从 DB 重建内存 registry，保证两者一致。

**教训**

内存状态 + DB 状态的双写场景，必须有"启动时从 DB 恢复内存"的逻辑，否则进程重启后两者必然漂移。临时修复只能补一处漏，治本得把重建逻辑补上。

### A/B 评测的样本量陷阱

**陷阱：小样本下 t-test 可能不显著，即使效应量大**

**实测数据**

| 样本量 | p-value | Cohen's d | t-test 结论 |
|--------|---------|-----------|-------------|
| n=5 | 0.1361 | -1.12（大效应） | 不显著 |
| n=49 | 0.0001 | -0.83（大效应） | 极显著 |

**原因**

t-test 的显著性同时依赖效应量与样本量。SPEC §6.4 的 `min_sample_size(d=0.8) ≈ 26` 就是这个门槛。n=5 虽然效应量更大（d=-1.12），但因样本太少，t-test 仍判"证据不足"。

**教训**

灰度策略中 `should_advance` 的"最小样本量"检查（条件 1）是必要的，不能为了快而跳过。即使肉眼可见 v2 更差，样本不够时 t-test 仍会说"证据不足"——这正是该检查存在的意义。

## MI300X 单 Pod 部署实测（2026-07-26）

在阿里云 PAI-DSW pod（K8s，无 Docker daemon，直连 MI300X）实测验证：

### 环境

| 项 | 值 |
|---|---|
| GPU | MI300X 192GB HBM3（gfx942） |
| ROCm | 7.2.1 |
| vLLM | 0.20.1+rocm721（pod 预装） |
| PyTorch | 2.10.0 ROCm |
| Python | 3.12（系统） |
| 部署形态 | 单 pod 多进程（vLLM v1 + vLLM v2 + vllmonline），SQLite 文件 DB |

### 关键决策与踩坑

1. **模型下载**：HF 直连被墙，`hf-mirror.com` 也偶发 403。最终用 **ModelScope**（`snapshot_download`）下到 `/mnt/workspace/modelscope/`，vLLM 用本地路径启动。
2. **`uv run uvicorn` 找不到模块**：DSW 的 `uv run` 默认用系统 Python，但 `uvicorn` CLI 不把 CWD 加 sys.path。解法：`PYTHONPATH=. python -m uvicorn vllmonline.server:app`。
3. **SQLite in-memory 在生产重启丢数据**：改用文件 `sqlite+aiosqlite:////mnt/workspace/vllmonline/vllmonline.db`，pod 重启模型注册不丢。
4. **同一 GPU 跑两个 vLLM**：v1 用 `--gpu-memory-utilization 0.30`，v2 用 `0.20`。实测占用 97GB / 192GB，留 95GB 余量。
5. **curl 中文 reason 报 URL 格式错**：`/api/canary/{id}/rollback?reason=中文` 解析失败，改用 `--data-urlencode` 或英文 reason。

### 实测结果

- ✅ v1 (Qwen2.5-7B) + v2 (Qwen2.5-1.5B) 同 GPU 共存，显存 97GB / 192GB
- ✅ 灰度 10% 分流：20 请求 → v1:17 / v2:3（统计误差内）
- ✅ 推进 30%：v2 占比上升至 ~27%
- ✅ 手动回滚：v2 自动 SLEEPING，后续 10 请求 100% 走 v1
- ✅ per-version metrics：`model_version="v1"/"v2"` label 正确
- ✅ 状态机保护：ACTIVE→LOADING 非法转移返回 409

### 验证脚本与量化结果

三个实测脚本（均在 `scripts/`）：

| 脚本 | 用途 | 配置 |
|------|------|------|
| `run_ab_eval.py` | A/B 评测 | 49 prompt × 三维度评分 × t-test |
| `bench_hotswap.py` | 热切换 source rate 压测 | 20 并发 × 30s × 第 15s 回滚 |
| `rollback_timeline.py` | 回滚时间线探测 | 50ms 轮询状态变化 |

量化结果：

- **A/B 评测**：p=0.0001, Cohen's d=-0.83, recommendation=rollback
- **热切换 source rate**：4640 请求 / 154 QPS / 100% 成功率 / 零丢失
- **回滚时间线**：API 响应 210ms，流量切回 321ms

### 启停脚本

```bash
bash scripts/start_all.sh   # 启 3 个 tmux session（vllm-v1 / vllm-v2 / vllmonline）
bash scripts/stop_all.sh    # 顺序停止 + 显示 GPU 显存释放
```
