# vLLMonline — 项目开发计划

> 面向 Agent 团队的可执行开发计划。每个 Phase 有明确任务、输入依赖和验收标准。
> 总周期：6 周。Phase 之间串行，Phase 内部可按模块并行。

---

## 里程碑

| Phase | 周次 | 交付物 | 状态 |
|-------|------|--------|------|
| P0 项目骨架 | W1 | Docker Compose + FastAPI 代理跑通 | ⬜ |
| P1 GPU 显存计算 | W1-W2 | 显存计算模块 + 状态机 + 单测 ≥ 95% | ⬜ |
| P2 模型热切换 | W2-W3 | 完整热切换流程，零请求丢失 | ⬜ |
| P3 流量路由与灰度 | W3-W4 | 按比例分流 + 阶梯放量 + per-version metrics | ⬜ |
| P4 A/B 评测 | W4-W5 | LLM-Judge 对比 + t-test + 报告 | ⬜ |
| P5 自动回滚 + 完善 | W5 | 劣化回滚 + 数据库 + 集成测试 | ⬜ |
| P6 文档与发布 | W5-W6 | README + Grafana + 压测报告 | ⬜ |

---

## Phase 0：项目骨架（W1）

### 目标

Docker Compose 一键启动 vLLM + vllmonline，FastAPI 代理转发到 vLLM backend 跑通。

### 任务清单

- [ ] **P0.1** 初始化项目：`pyproject.toml`、`Makefile`、`.gitignore`、pre-commit
- [ ] **P0.2** Docker Compose：vllmonline + vLLM + PostgreSQL + Redis + Prometheus + Grafana
  - vLLM 用 `vllm/vllm-openai:latest` 镜像，加载一个测试模型（如 Qwen2.5-0.5B）
  - PostgreSQL 用 `pgvector/pgvector:pg16`
  - Prometheus 配置抓取 vllmonline 的 `/metrics`
- [ ] **P0.3** `config.py`：配置管理（YAML + env vars），所有组件有默认值
- [ ] **P0.4** `server.py`：FastAPI 主入口，最简代理——收到请求后转发到 vLLM backend，返回 response
  - `POST /v1/chat/completions` → 透传到 vLLM
  - `GET /healthz` → `{"status": "ok"}`
  - 用 `httpx.AsyncClient` 转发，streaming 透传
- [ ] **P0.5** `conftest.py`：pytest fixtures（mock vLLM server、test client）
- [ ] **P0.6** 验证：`curl localhost:8080/v1/chat/completions` 返回 vLLM 的模型回答

### 验收标准

- [ ] `docker compose up` 启动全部服务
- [ ] `POST /v1/chat/completions` 能正常返回模型回答
- [ ] streaming 请求逐 chunk 透传
- [ ] `/healthz` 返回 ok
- [ ] Prometheus 能抓取到 `/metrics`

---

## Phase 1：GPU 显存计算 + 状态机（W1-W2）

### 依赖

Phase 0 骨架可用。不需要实际 GPU——显存计算模块完全纯函数，可无 GPU 单测。

### 任务清单

- [ ] **P1.1** `scheduler/types.py`：枚举（`ModelState`, `LoadStrategy`）+ 数据类（`ModelMemoryProfile`, `GPUState`, `LoadDecision`）
- [ ] **P1.2** `scheduler/gpu_memory.py`：
  - `calculate_weight_memory(params_billion, dtype, quantization)` → GB
  - `estimate_kv_cache_from_weight(weight_gb, utilization=0.90)` → GB
  - `can_load(gpu_state, new_model_gb, kv_cache_budget_gb)` → `LoadDecision`
  - `build_model_profile(...)` → `ModelMemoryProfile`
- [ ] **P1.3** `scheduler/lifecycle.py`：
  - `Model` 数据类（id, state, endpoint, weight_gb…）
  - `Model.transition(to)` 方法（加锁 + 合法性检查 + 持久化到 DB）
  - `ModelRegistry`（register/get/list_serving/list_on_gpu）
- [ ] **P1.4** `tests/test_gpu_memory.py`：
  - 测试各种 dtype/量化的 weight 计算
  - 测试 can_load 所有 4 个分支（DIRECT/SLEEP_OLD/UNLOAD_OLD/INSUFFICIENT）
  - 边界条件：刚好够、刚好不够、空 GPU、模型超大
- [ ] **P1.5** `tests/test_lifecycle.py`：
  - 所有合法转移通过
  - 所有非法转移被 `IllegalTransitionError` 拦截
  - 并发安全：两个协程同时 transition 不会出现竞态

### 验收标准

- [ ] `calculate_weight_memory(7.0, "fp16")` → 14.0
- [ ] `calculate_weight_memory(70.0, "int4", "gptq")` → 36.75
- [ ] 80GB GPU，free=10GB → 加载 14GB 模型 → `SLEEP_OLD`（如果旧模型 KV 够大）
- [ ] ACTIVE→SLEEPING 合法，ACTIVE→IDLE 非法（抛异常）
- [ ] 单测覆盖率 ≥ 95%（gpu_memory）+ ≥ 90%（lifecycle）

---

## Phase 2：模型热切换（W2-W3）

### 依赖

Phase 1 完成（显存计算 + 状态机）。

### 任务清单

- [ ] **P2.1** `vllm/client.py`：封装 vLLM API
  - `chat_completion` / `stream_chat` / `health` / `list_models`
  - 超时 + 重试（tenacity）
- [ ] **P2.2** `vllm/adapter.py`：
  - `sleep` / `wake` / `load_lora` / `unload_lora`
  - sleep/wake 后的 CUDA graph workaround（发 warmup 请求）
- [ ] **P2.3** `router/drain.py`：
  - `start_drain(model)`：停止向该模型发新请求
  - `wait_drain(model, timeout=30)`：等待 pending_requests 降为 0
  - `force_drain(model)`：超时后强制 cancel
- [ ] **P2.4** `scheduler/engine.py`：
  - `hot_swap(old_model_id, new_model_id)`：编排完整热切换流程
  - 流程：①can_load 校验 → ②load 新模型 → ③drain 旧模型 → ④sleep/unload 旧模型 → ⑤更新路由表
- [ ] **P2.5** `api/schemas.py`：Pydantic models（ModelRegisterRequest, ModelResponse, CanaryStartRequest…）
- [ ] **P2.6** `api/routes.py`：模型管理 API
  - `POST /api/models/register` → `GET /api/models/{id}` → `POST /api/models/{id}/load` → `POST /api/models/{id}/unload`
- [ ] **P2.7** `db/models.py` + `db/session.py`：SQLAlchemy models + async session
- [ ] **P2.8** `tests/test_integration.py`：端到端热切换测试

### 验收标准

- [ ] 注册两个模型 → 加载 v1 → v1 ACTIVE → 可以服务请求
- [ ] 加载 v2 → 旧模型 drain → v2 ACTIVE → 全程零请求丢失（持续发请求，success rate=100%）
- [ ] GPU 显存不足时热切换被拒绝，返回友好错误
- [ ] 状态变更持久化到 PostgreSQL

---

## Phase 3：流量路由与灰度（W3-W4）

### 依赖

Phase 2 完成（热切换可用）。

### 任务清单

- [ ] **P3.1** `router/proxy.py`：
  - 加权随机选择目标版本
  - 注入 `x-model-version` header
  - 转发到对应 vLLM endpoint
- [ ] **P3.2** `router/middleware.py`：per-request metrics 采集
  - 记录 start_time、first_token_time、end_time
  - 计算 TTFT/TPOT → 写入 Prometheus（带 model_version label）
- [ ] **P3.3** `canary/strategy.py`：
  - `GrayStrategy` 状态机（INIT→STAGE_10%→STAGE_30%→STAGE_100%→COMPLETED）
  - `should_advance(stage, metrics_v1, metrics_v2)` → `(bool, reason)`
  - 推进条件：最小样本量 + 统计显著 + 指标未劣化
- [ ] **P3.4** `canary/metrics_collector.py`：
  - 从 Prometheus 拉取 per-version metrics
  - 计算 P50/P99/mean/error_rate
- [ ] **P3.5** `vllm/metrics.py`：
  - 解析 vLLM `/metrics`（Prometheus text format）
  - 按 model_version relabel
- [ ] **P3.6** `api/routes.py`：灰度管理 API
  - `POST /api/canary/start` → `GET /api/canary/{id}/status` → `POST /api/canary/{id}/advance` → `POST /api/canary/{id}/rollback`
- [ ] **P3.7** `tests/test_strategy.py`：灰度策略单测

### 验收标准

- [ ] 两个 ACTIVE 模型，路由按配置比例分流（误差 < 5%）
- [ ] `/metrics` 端点输出的指标正确带 `model_version` label
- [ ] 模拟 v2 指标优秀 → 自动从 10% 推进到 30%
- [ ] 模拟 v2 指标劣化 → 灰度暂停（不推进）
- [ ] Grafana 仪表盘可同时查看 v1 vs v2 的 TTFT/TPOT/吞吐对比曲线

---

## Phase 4：A/B 评测（W4-W5）

### 依赖

Phase 3 完成（灰度路由可用，有 per-version metrics）。

### 任务清单

- [ ] **P4.1** `eval/statistics.py`：
  - `welch_ttest(scores_v1, scores_v2)` → `TTestResult`
  - `min_sample_size(effect_size, power=0.8, alpha=0.05)` → int
  - 用已知数据集验证正确性
- [ ] **P4.2** `eval/judge.py`：
  - `LLMJudge`：发送同一 prompt 到 v1 和 v2，收集两个回答
  - `compare(response_v1, response_v2, user_prompt)` → `ComparisonResult`
  - 用 SPEC.md §6.2 的 prompt 模板
  - 三维度独立评分（accuracy/completeness/safety）
- [ ] **P4.3** `eval/reporter.py`：
  - 生成 Markdown 格式评测报告
  - 含：样本量、各维度均分、p 值、效应量、建议
- [ ] **P4.4** `api/routes.py`：评测 API
  - `POST /api/eval/compare` → `GET /api/eval/{id}/report`
- [ ] **P4.5** `tests/test_statistics.py`：统计模块单测 ≥ 95%

### 验收标准

- [ ] Welch's t-test 用已知数据集验证（如 scipy.stats.ttest_ind 交叉验证）
- [ ] LLM-as-Judge 能正确解析 JSON 格式的评分结果
- [ ] 评测报告含三维度分数 + p 值 + 效应量 + 建议（advance/hold/rollback）
- [ ] 最小样本量计算：d=0.5 → 64, d=0.2 → 394

---

## Phase 5：自动回滚 + 集成完善（W5）

### 依赖

Phase 4 完成（评测可用）。

### 任务清单

- [ ] **P5.1** `canary/rollback.py`：
  - 每 10s 检查当前 metrics
  - 任一指标劣化超阈值 → 自动执行回滚
  - 回滚流程：新模型 drain → 流量 100% 切回旧模型 → 记录事件
- [ ] **P5.2** 数据库 migration（Alembic）
- [ ] **P5.3** `tests/test_integration.py`：全部集成测试
  - 端到端灰度：启动→10%→30%→回滚
  - 端到端热切换：零请求丢失
  - 端到端 A/B：采样→双发→Judge→报告
- [ ] **P5.4** Grafana 仪表盘：
  - 全局概览：在线模型数、GPU 显存使用率
  - 灰度详情：v1 vs v2 的 TTFT/TPOT/throughput/error_rate 对比
  - A/B 评测算分趋势

### 验收标准

- [ ] 模拟 v2 劣化（P99 +50%）→ 30s 内自动回滚
- [ ] 回滚后流量 100% 恢复到 v1
- [ ] 回滚事件记录到数据库（含 metrics_snapshot）
- [ ] 集成测试全部通过

---

## Phase 6：文档与发布（W5-W6）

### 任务清单

- [ ] **P6.1** README.md：5 分钟快速开始 + 架构图 + API 文档链接
- [ ] **P6.2** Grafana 仪表盘 JSON 文件（`deploy/docker-compose/grafana-dashboards/vllmonline.json`）
- [ ] **P6.3** Locust 压测脚本（模拟多模型并发请求）
- [ ] **P6.4** 压测报告：单 GPU 上的最大 QPS、热切换期间的成功率
- [ ] **P6.5** 清理代码注释、统一日志格式、确保 ruff/mypy 通过

### 验收标准

- [ ] `docker compose up` 后按 README 走完全流程
- [ ] Grafana 仪表盘可展示所有核心指标
- [ ] ruff + mypy 无报错
- [ ] 单测 + 集成测覆盖率 ≥ 85%

---

## 测试覆盖率总目标

| 模块 | 目标 |
|------|------|
| `scheduler/gpu_memory.py` | ≥ 95% |
| `scheduler/lifecycle.py` | ≥ 90% |
| `eval/statistics.py` | ≥ 95% |
| `canary/strategy.py` | ≥ 90% |
| `vllm/metrics.py` | ≥ 85% |
| 项目整体 | ≥ 85% |

---

## 通用验收标准（每个 PR）

- [ ] 代码通过 ruff + mypy
- [ ] 新增函数有 docstring
- [ ] 涉及外部调用的有超时 + 重试
- [ ] 关键路径有日志
- [ ] 单测覆盖新增代码

---

_本计划为活文档。每个 Phase 完成后打勾，遇到阻塞标注原因。_
