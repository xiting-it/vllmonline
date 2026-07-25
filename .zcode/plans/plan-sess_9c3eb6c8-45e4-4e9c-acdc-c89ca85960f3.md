# vLLMonline 实现计划（基于 SPEC.md + PLAN.md）

## 0. 关键决策（已与用户确认）

| 项 | 决策 |
|---|---|
| 目标硬件 | **AMD MI300X 192GB HBM3**（gfx942 / ROCm 7.2.1）|
| GPU 查询 | `rocm-smi --showmeminfo vram`（不是 nvidia-smi）|
| vLLM 镜像 | `rocm/vllm:ubuntu22.04-rocm7.2.1-py312-torch2.9.1-1.36.3` |
| Docker device | `--device /dev/kfd --device /dev/dri --group-add video --cap-add=SYS_PTRACE` |
| LoRA 运行时 | `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` |
| 工具链 | uv + PEP 621 + PEP 735 dependency-groups |
| 测试 | SQLite 快测（默认）+ testcontainers PG（集成测试，`-m integration`）|
| 统计 | scipy.stats.ttest_ind(equal_var=False) + 手写 Cohen's d / min_sample_size |
| fake vLLM | FastAPI 假 app + httpx ASGITransport |

## 1. 对 SPEC 的关键调整（MI300X 必须做）

SPEC 通篇假设 NVIDIA/CUDA，需做以下抽象层调整（不破坏接口契约）：

- **CUDA_CONTEXT_OVERHEAD_GB**: SPEC 默认 2.0。MI300X 上实际更大（HBM stack + ROCm runtime），改为**配置项** `config.cuda_context_overhead_gb`，默认值保留 2.0，但在 `deploy/` 的 MI300X 配置里给出推荐值 4.0-5.0GB，并写入 README 说明。
- **GPU 信息查询**: 新增 `vllm/gpu_info.py`，抽象 `GpuInfoProvider` 接口，两个实现：`RocmSmiProvider`（解析 rocm-smi 输出）、`NvidiaSmiProvider`（nvidia-smi），通过 config 切换。
- **vLLM 镜像**: docker-compose 的 `vllm` 服务用 `rocm/vllm:ubuntu22.04-rocm7.2.1-py312-torch2.9.1-1.36.3`，挂载 ROCm device。
- **注释/文档**: 所有"CUDA"字样在代码注释里改为"GPU device"，避免误导。

## 2. 目录结构（严格按 SPEC §2.2，新增少量文件）

```
vllmonline/
├── vllmonline/
│   ├── __init__.py
│   ├── server.py              # FastAPI 主入口
│   ├── config.py              # pydantic-settings，YAML + env
│   ├── gpu_info.py            # 【新增】GPU 信息抽象（ROCm/NVIDIA）
│   ├── router/                # proxy.py, middleware.py, drain.py
│   ├── scheduler/             # engine.py, gpu_memory.py, lifecycle.py, types.py
│   ├── canary/                # strategy.py, metrics_collector.py, rollback.py
│   ├── eval/                  # judge.py, statistics.py, reporter.py
│   ├── vllm/                  # client.py, metrics.py, adapter.py
│   ├── db/                    # models.py, session.py
│   └── api/                   # schemas.py, routes.py
├── alembic/                   # 【P5.2】migrations
├── deploy/docker-compose/
│   ├── docker-compose.yml
│   ├── docker-compose.override.dev.yml  # 无 GPU 开发环境：用 fake vLLM
│   ├── prometheus.yml
│   ├── alembic.ini
│   └── grafana-dashboards/vllmonline.json
├── tests/
│   ├── conftest.py            # fake vLLM app + ASGITransport + DB fixtures
│   ├── fake_vllm.py           # 假 vLLM server 实现（SSE/metrics/sleep/wake）
│   ├── test_gpu_memory.py
│   ├── test_lifecycle.py
│   ├── test_strategy.py
│   ├── test_statistics.py
│   ├── test_metrics.py        # /metrics 解析 + relabel
│   ├── test_proxy.py          # 路由 + 分流 + header 注入
│   ├── test_engine.py         # 热切换编排
│   ├── test_api_models.py
│   ├── test_api_canary.py
│   ├── test_api_eval.py
│   └── test_integration.py    # -m integration，用 testcontainers PG
├── scripts/
│   └── locustfile.py          # P6.3 压测
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── DEVELOPMENT.md             # 【新增】开发环境搭建/工具链说明
├── SPEC.md, PLAN.md
└── requirements.txt           # 【用户要求】显式记录所有外部工具
```

## 3. 各 Phase 实现要点

### Phase 0 — 项目骨架
- `pyproject.toml`（PEP 621+735，uv）：runtime deps + `[gpu]` extra + `[dev]`/`[test]` groups。
- `requirements.txt`：从 uv 导出的扁平依赖清单（用户明确要求）。
- `config.py`：pydantic-settings，所有可配置项有默认值；YAML + env var 双源。
- `server.py`：FastAPI app，注册 `/v1/chat/completions`（透传）、`/healthz`、`/readyz`、`/metrics`。
- `Makefile`：`make install/test/lint/check/compose-up/compose-down`。
- docker-compose：vllmonline + vLLM(ROCm) + PG(pgvector) + Redis + Prometheus + Grafana。
- **`docker-compose.override.dev.yml`**：无 GPU 时把 vLLM 换成假 server 容器（基于 vllmonline 自身镜像跑一个 fake 模式），方便 macOS 开发。
- 验收：`docker compose up` + `curl /v1/chat/completions` 通；`/healthz` ok；Prometheus 抓到 `/metrics`。

### Phase 1 — GPU 显存 + 状态机（最硬核）
- `scheduler/types.py`：`ModelState`/`LoadStrategy` 枚举；`ModelMemoryProfile`/`GPUState`/`LoadDecision` dataclass（`LoadDecision.detail: str`）。
- `scheduler/gpu_memory.py`：
  - dtype→bytes 表、量化 overhead 表**硬编码**（严格按 SPEC §3.2）。
  - `calculate_weight_memory(7.0, "fp16")` → 14.0；`(70.0, "int4", "gptq")` → 36.75。
  - `estimate_kv_cache_from_weight(w, util)` = `w * 0.25 * util`。
  - `can_load()` 决策树 4 分支（DIRECT/SLEEP_OLD/UNLOAD_OLD/INSUFFICIENT），每个返回人类可读 detail。
- `scheduler/lifecycle.py`：
  - TRANSITIONS 表硬编码（SPEC §4.2）。
  - `Model` 持 `asyncio.Lock`，`transition()` 加锁 + 校验 + 持久化回调。
  - `ModelRegistry`：register/get/list_serving/list_on_gpu/get_active_version。
  - `IllegalTransitionError`。
- 测试：穷举 dtype×量化、can_load 4 分支 + 边界（0B、刚好够、刚好不够、超大）；状态机全部合法/非法转移；并发竞态测试。

### Phase 2 — 热切换
- `vllm/client.py`：httpx.AsyncClient 封装 chat_completion/stream_chat/health/list_models，tenacity 重试。
- `vllm/adapter.py`：sleep/wake/load_lora/unload_lora + warmup workaround（SPEC §7.2）。
- `router/drain.py`：start_drain/wait_drain(timeout)/force_drain。
- `scheduler/engine.py`：`hot_swap(old, new)` 编排 5 步（can_load→load→drain→sleep/unload→更新路由）。
- `db/models.py` + `db/session.py`：SQLAlchemy 2.0 async，按 SPEC §9.1 建表。
- `api/schemas.py` + `routes.py`：模型管理 CRUD。
- 集成测试：注册→加载→ACTIVE→服务→热切换全程零请求丢失（持续压请求 + success rate=100%）。

### Phase 3 — 流量路由 + 灰度
- `router/proxy.py`：加权随机选版本、注入 `x-model-version` header、httpx streaming 透传。
- `router/middleware.py`：per-request metrics（TTFT/TPOT/throughput/error/in-flight），写 Prometheus 带 model_version label。
- `canary/strategy.py`：`GrayStrategy` 状态机（INIT→STAGE_10→STAGE_30→STAGE_100→COMPLETED），`should_advance()` 返回 `(bool, 具体 reason)`。推进条件按 SPEC §5.5（最小样本 + p<0.05 + 5 个性能阈值）。
- `canary/metrics_collector.py`：从 Prometheus 拉 per-version，算 P50/P99/mean/error_rate。
- `vllm/metrics.py`：解析 vLLM Prometheus text format，追加 `model_version` label（SPEC §7.3）。
- 灰度 API：start/status/advance/rollback/metrics。

### Phase 4 — A/B 评测
- `eval/statistics.py`：`welch_ttest()`（scipy + 手算交叉验证）、`min_sample_size(d, power, alpha)`（SPEC §6.4 公式）。验证 d=0.5→64, d=0.2→394。
- `eval/judge.py`：`LLMJudge`，按 SPEC §6.2 prompt 模板，解析三维度 JSON 评分。
- `eval/reporter.py`：Markdown 报告（样本量/均分/p/效应量/建议）。
- 评测 API：compare/report。

### Phase 5 — 自动回滚 + 集成
- `canary/rollback.py`：每 10s 巡检，任一指标超阈值→回滚（drain v2→100% v1→记事件）。
- Alembic migration（autogenerate + 手工校 JSONB 默认值）。
- `test_integration.py`（`-m integration`，testcontainers PG）：端到端灰度推进/回滚、热切换零丢失、A/B 全流程。

### Phase 6 — 文档 + 发布
- `README.md`：5 分钟 quickstart（含 MI300X docker 启动）、架构图、API 文档目录。
- `DEVELOPMENT.md`：环境搭建（uv 用法、make 命令、fake vLLM 模式）、依赖清单说明。
- `requirements.txt`：扁平依赖（用户要求显式记录）。
- Grafana dashboard JSON（v1 vs v2 对比）。
- `scripts/locustfile.py` 压测脚本 + 报告模板。
- ruff + mypy 全绿，整体覆盖率 ≥85%。

## 4. 测试矩阵

| 测试 | 标记 | DB | GPU | fake vLLM | 目的 |
|---|---|---|---|---|---|
| 单测（gpu_memory/lifecycle/statistics/strategy/metrics） | 无 | 无 | 无 | 无 | 纯函数 + 状态机 |
| API/路由单测 | 无 | SQLite in-mem | 无 | ASGITransport fake | 端点契约 |
| 集成测试 | `@pytest.mark.integration` | testcontainers PG | 无 | ASGITransport fake | 端到端流程 |

无 GPU 也能全部跑通（fake vLLM 完整模拟 /sleep /wake /metrics /chat streaming）。真实 MI300X 上只需 `docker compose up` 即可演示。

## 5. 执行方式

按 Phase 顺序串行做，每个 Phase 完成后：
1. 跑 `make test` 确认单测全绿；
2. 跑 `make lint check`（ruff + mypy）；
3. 向你报告该 Phase 的验收清单达成情况（如实标注通过/未通过/部分）；
4. git commit（每个 Phase 一个 commit，信息含 Phase 编号和验收点）。

预计工作量较大，会分多轮交付，每轮专注 1-2 个 Phase 并同步进度。P0+P1（骨架+显存/状态机）是第一轮，因为这是整个项目的地基。

## 6. 我会主动做的，不需要你操心

- git init + .gitignore + pre-commit 配置
- uv venv + 依赖安装 + uv.lock
- README/DEVELOPMENT 里记录**每一个**外部工具（uv、docker、ROCm 驱动、testcontainers 等）和版本
- 所有"未确认/踩坑/已知问题"在代码注释和 DEVELOPMENT.md 里显式标注

---

请确认。确认后我从 Phase 0 开始落地。