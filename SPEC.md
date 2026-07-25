# vLLMonline — 项目技术规约

> **给 Agent 团队的执行规格书。** 本文档定义系统架构、接口契约、核心算法、数据模型、状态机。所有实现必须严格遵循本文档的约束。

---

## 1. 系统概述

### 1.1 一句话

vLLMonline 是 vLLM 推理引擎的上层管理平台，实现**零停机模型热切换 + 灰度发布 + A/B 自动评测 + 劣化自动回滚**。

### 1.2 解决的工程问题

当前企业上线新模型的典型流程：
```
DevOps 改配置文件 → 重启 vLLM 服务 → 3 分钟冷启动 → 全量切换 → 祈祷别出问题
```

vLLMonline 替代方案：
```
上传新模型 → GPU 显存安全校验 → 加载到同 GPU → 10% 流量灰度
→ 自动对比新旧模型（延迟/吞吐/质量）→ 达标自动全量，不达标自动回滚
全程零停机，客户端无感知。
```

### 1.3 核心设计约束

| # | 约束 | 验证方式 |
|---|------|---------|
| 1 | **零停机**：模型切换期间客户端无感知，在飞请求全部正常完成 | 集成测试：切换过程中持续发请求，success rate = 100% |
| 2 | **显存安全**：任何加载操作前必须通过 can_load() 判定，绝不 OOM | 单测：穷举各种显存组合 |
| 3 | **统计驱动**：灰度阶段推进基于 Welch's t-test (p < 0.05)，不能靠人工感觉 | 单测：mock 数据验证推进/暂停/回滚判定 |
| 4 | **状态机完备**：非法状态转移抛异常，不允许"未定义行为" | 单测：每个非法转移都被拦截 |
| 5 | **可观测**：每个模型版本独立打标 metrics，Prometheus + Grafana 同维度对比 | 集成测试：/metrics 端点输出含 model_version label |

### 1.4 技术栈

| 维度 | 选型 | 备注 |
|------|------|------|
| 语言 | Python 3.12 | 全栈 asyncio |
| API 框架 | FastAPI | 代理层 + 管理面 |
| HTTP 客户端 | httpx.AsyncClient | 请求转发到 vLLM |
| vLLM 版本要求 | ≥ 0.5.0 | 需要 /sleep、/wake、/v1/load_lora_adapter 端点 |
| 数据库 | PostgreSQL | 模型版本、灰度历史、评测结果 |
| 缓存 | Redis | 实时 metrics 缓存、路由表热更新 |
| 可观测性 | Prometheus + Grafana | 自定义 metrics（按 model_version 打标） |
| 部署 | Docker Compose → K8s + GPU Operator | 开发/生产两套 |

---

## 2. 系统架构

### 2.1 整体拓扑

```
                          ┌──────────────────────────────────────┐
   Client ──HTTP──────────▶│  vllmonline Proxy (:8080)            │
                           │  ├─ 路由决策（读路由表）               │
                           │  ├─ 流量按版本比例分流                 │
                           │  ├─ 注入 header: x-model-version      │
                           │  ├─ 转发到 vLLM backend               │
                           │  └─ 采集 per-request metrics          │
                           └──────┬───────────────┬───────────────┘
                                  │               │
                    ┌─────────────▼──┐   ┌────────▼──────────┐
                    │ vLLM (v1)      │   │ vLLM (v2)         │
                    │ :8000          │   │ :8001             │
                    │ model: qwen-7b │   │ model: qwen-7b-v2 │
                    └────────────────┘   └───────────────────┘
                                  │               │
                                  ▼               ▼
                              GPU 0 (同一张 GPU 上的两个模型实例)
```

**关键理解**：v1 和 v2 可以是同一张 GPU 上的两个 vLLM 实例（不同端口），也可以是同一个 vLLM 实例内通过 `/v1/load_lora_adapter` 管理的两个 adapter。本平台**两种模式都支持**。

### 2.2 目录结构（Agent 实现时严格遵循）

```
vllmonline/
├── vllmonline/
│   ├── __init__.py
│   ├── server.py              # FastAPI 主入口
│   ├── config.py              # 配置管理
│   ├── router/
│   │   ├── __init__.py
│   │   ├── proxy.py           # 反向代理核心：转发 + 分流
│   │   ├── middleware.py      # model_version header 注入 + metrics 采集
│   │   └── drain.py           # 优雅排空：停止发新请求 + 等待在飞请求完成
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── engine.py          # 调度引擎：编排热切换完整流程
│   │   ├── gpu_memory.py      # GPU 显存计算（can_load 决策树）
│   │   ├── lifecycle.py       # 模型状态机 + ModelRegistry
│   │   └── types.py           # 状态枚举、数据类
│   ├── canary/
│   │   ├── __init__.py
│   │   ├── strategy.py        # 灰度策略机：阶梯放量 + 阶段推进判定
│   │   ├── metrics_collector.py  # Per-version metrics 采集
│   │   └── rollback.py        # 劣化检测 + 自动回滚执行
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── judge.py           # LLM-as-Judge：对比评分
│   │   ├── statistics.py      # Welch's t-test + 最小样本量
│   │   └── reporter.py        # 评测报告生成
│   ├── vllm/
│   │   ├── __init__.py
│   │   ├── client.py          # vLLM API 客户端
│   │   ├── metrics.py         # /metrics 解析 + relabel
│   │   └── adapter.py         # sleep/wake/load_lora
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py          # SQLAlchemy models
│   │   └── session.py         # Async session factory
│   └── api/
│       ├── __init__.py
│       ├── schemas.py         # Pydantic request/response models
│       └── routes.py          # REST API endpoints
├── deploy/
│   └── docker-compose/
│       ├── docker-compose.yml
│       ├── prometheus.yml
│       └── grafana-dashboards/
│           └── vllmonline.json
├── tests/
│   ├── conftest.py
│   ├── test_gpu_memory.py
│   ├── test_lifecycle.py
│   ├── test_strategy.py
│   ├── test_statistics.py
│   └── test_integration.py
├── pyproject.toml
├── Makefile
├── README.md
├── SPEC.md          # 本文件
└── PLAN.md          # 项目计划
```

---

## 3. GPU 显存计算（最关键模块）

> ⚠️ Agent 注意：这个模块的所有计算**必须精确**。显存算错 1GB → 生产环境 OOM → 整个 GPU 上的所有模型全挂。

### 3.1 显存占用模型

```
GPU 总显存（以 A100-80G 为例）
├── CUDA Context 开销          ~2 GB    （固定，驱动 + cuBLAS workspace）
├── 模型 1 权重                 W1 GB
├── 模型 1 KV Cache            K1 GB
├── 模型 2 权重（若有）         W2 GB
├── 模型 2 KV Cache（若有）     K2 GB
└── 剩余                        Free GB
```

### 3.2 权重显存计算公式

$$W = P \times B \times (1 + Q_{overhead})$$

其中：
- $P$ = 参数量（billions），如 7.0 表示 7B
- $B$ = 每个参数的字节数，取值见下表
- $Q_{overhead}$ = 量化方法额外开销比（仅量化模型）

**dtype → bytes 对照表（硬编码，不可随意修改）**：

| dtype | bytes/param | 适用场景 |
|-------|------------|---------|
| fp32 | 4.0 | 极少用 |
| fp16 | 2.0 | 标准推理 |
| bf16 | 2.0 | A100/H100 推荐 |
| int8 | 1.0 | 量化推理 |
| int4 | 0.5 | GPTQ/AWQ 量化 |

**量化方法额外开销（叠加在 bytes/param 之上）**：

| 量化方法 | 额外开销比 | 说明 |
|---------|-----------|------|
| 无 | 0 | — |
| gptq | 0.05 | 量化 scale + zero point 存储 |
| awq | 0.05 | 同上 |
| gguf | 0.03 | GGUF 格式元数据 |

**计算示例**：
- Qwen-7B, fp16: `7.0 × 2.0 = 14.0 GB`
- Qwen-72B, int4 + gptq: `72.0 × 0.5 × 1.05 = 37.8 GB`

### 3.3 KV Cache 显存估算

KV Cache 的精确计算需要知道模型架构参数（num_layers、hidden_dim、num_kv_heads），公式为：

$$K = 2 \times L \times H \times \frac{H_{kv}}{H} \times B_{dtype} \times B \times S \times U$$

其中：
- $L$ = num_layers（如 Qwen-7B 为 32）
- $H$ = hidden_dim（如 4096）
- $H_{kv}/H$ = KV head ratio（GQA/MQA 压缩比）
- $B_{dtype}$ = dtype 字节数
- $B$ = max_batch_size
- $S$ = max_seq_len
- $U$ = gpu_memory_utilization（vLLM 默认 0.90）

**简化版（架构参数未知时）**：$K = W \times 0.25 \times U$

这是基于经验的保守估算，对于 7B-70B 模型，KV Cache 通常占权重的 20-30%。

### 3.4 can_load() 决策树（核心算法）

```
输入：GPU 当前状态、新模型所需显存
输出：LoadDecision(strategy, free_after, detail)

1. 计算 required = new_model_weight + new_model_kv_cache
2. 如果 current_free >= required → DIRECT（直接加载）
3. 否则，计算 sleep 所有已加载模型能释放的 KV Cache 总量
   如果 current_free + released_kv >= required → SLEEP_OLD（休眠旧模型）
4. 否则，计算 unload 所有已加载模型能释放的总量
   如果 total_gpu - cuda_overhead >= required → UNLOAD_OLD（卸载旧模型）
5. 否则 → INSUFFICIENT（模型太大，空 GPU 都放不下）
```

**Agent 实现要求**：
- `LoadStrategy` 是枚举：`DIRECT | SLEEP_OLD | UNLOAD_OLD | INSUFFICIENT`
- `LoadDecision` 必须包含 `detail: str` 字段，用人类可读的文字解释决策逻辑（方便调试和日志）
- 所有 GB 值用 float，保留 1 位小数
- `CUDA_CONTEXT_OVERHEAD_GB = 2.0` 作为默认常量，但可通过配置覆盖（不同 GPU 型号可能不同）

---

## 4. 模型状态机

> ⚠️ Agent 注意：状态机是保证系统行为可预测的基础。**必须完备，不允许未定义行为。**

### 4.1 状态定义

```
                    ┌─────────┐
            register│  IDLE   │ 模型已注册但未加载
                    └────┬────┘
                         │ load()
                    ┌────▼────┐
                    │ LOADING │ 正在加载到 GPU
                    └────┬────┘
                         │ load complete
                    ┌────▼────┐
            sleep()│  ACTIVE │ 正常服务（接收请求）
            ◄──────┤         │
                    └────┬────┘
                         │ drain()  ← 停止新请求，等待在飞请求完成
                    ┌────▼────┐
            wake()  │DRAINING │ 排空中（旧请求还在跑）
            ───────▶│         │
                    └────┬────┘
                         │ drain complete
                    ┌────▼────┐
            unload()│SLEEPING │ 权重在 GPU，KV Cache 已释放
            ◄───────┤         │
                    └────┬────┘
                         │ unload()
                    ┌────▼────┐
                    │UNLOADING│ 正在从 GPU 卸载
                    └────┬────┘
                         │ unload complete → IDLE
                    ┌────▼────┐
                    │  ERROR  │ 任何状态都可能转入（加载失败/OOM/健康检查连续失败）
                    └─────────┘
```

### 4.2 合法转移表

```python
TRANSITIONS = {
    IDLE:      {LOADING},
    LOADING:   {ACTIVE, ERROR},
    ACTIVE:    {DRAINING, SLEEPING, ERROR},
    DRAINING:  {SLEEPING, ERROR},
    SLEEPING:  {LOADING, UNLOADING, ERROR},   # SLEEPING→LOADING = wake
    UNLOADING: {IDLE, ERROR},
    ERROR:     {IDLE},                         # 只能 reset
}
```

### 4.3 实现要求

- 每个模型实例持有一个 `asyncio.Lock`，状态变更必须加锁
- `transition(from, to)` 方法：先检查 TRANSITIONS[from] 是否包含 to，不包含则抛 `IllegalTransitionError`
- 每次状态变更**必须**持久化到数据库（`model_versions.status` + `model_versions.state_changed_at`），崩溃恢复时读取
- `ModelRegistry` 是线程安全的模型注册表，提供：`register`、`get`、`list_serving`、`list_on_gpu`、`get_active_version`

### 4.4 辅助判定

| 方法 | 含义 | 实现 |
|------|------|------|
| `can_serve()` | 能否接受新请求 | `state == ACTIVE` |
| `is_on_gpu()` | 是否占用 GPU 显存 | `state in {LOADING, ACTIVE, DRAINING, SLEEPING}` |
| `is_terminal()` | 是否是终态 | `state in {IDLE, ERROR}` |

---

## 5. 流量路由与灰度策略

### 5.1 代理层请求处理流程

```
Client Request → FastAPI Proxy
  1. 根据请求中的 model 字段查找路由表
  2. 获取该 model 的所有 ACTIVE 版本的 endpoint 和权重
  3. 加权随机选择目标版本（如 v1: 70%, v2: 30%）
  4. 注入 header: x-model-version: {selected_version}
  5. httpx.AsyncClient 转发请求到 vLLM endpoint
  6. 采集 per-request metrics（见 §5.4）
  7. 返回 response（streaming 逐 chunk 透传）
```

### 5.2 路由表结构

```python
# 内存中的路由表（每 5s 从 DB 同步，或通过 Redis Pub/Sub 实时更新）
routing_table = {
    "qwen-7b": [
        {"model_id": "qwen-7b-v1", "endpoint": "http://vllm:8000/v1", "weight": 0.7},
        {"model_id": "qwen-7b-v2", "endpoint": "http://vllm-v2:8001/v1", "weight": 0.3},
    ]
}
```

### 5.3 灰度策略状态机

```
INIT → STAGE_10%
         │
         ├──(统计显著 & 指标达标)──→ STAGE_30%
         │                              │
         ├──(劣化)──→ ROLLBACK          ├──(统计显著 & 指标达标)──→ STAGE_100%
         │                              │                              │
         └──(不显著)──→ 延长当前阶段     ├──(劣化)──→ ROLLBACK          ├──(劣化)──→ ROLLBACK
                                        │                              │
                                        └──(不显著)──→ 延长             └── 灰度完成 → COMPLETED
```

### 5.4 Per-Version Metrics（每个版本独立采集）

| 指标 | Prometheus metric 名称 | 说明 |
|------|----------------------|------|
| 请求总数 | `vllmonline_requests_total{model_version, status}` | Counter |
| TTFT（首 token 延迟） | `vllmonline_ttft_seconds{model_version}` | Histogram, buckets: [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10] |
| TPOT（每 token 生成时间） | `vllmonline_tpot_seconds{model_version}` | Histogram |
| 吞吐量 | `vllmonline_throughput_tokens_per_second{model_version}` | Gauge |
| 错误率 | `vllmonline_errors_total{model_version, error_type}` | Counter |
| 在飞请求数 | `vllmonline_requests_in_flight{model_version}` | Gauge |

### 5.5 灰度阶段推进判定

每个阶段必须同时满足以下条件才能推进：

```
1. 最小样本量已达成（n >= min_sample_size(effect_size=0.3, power=0.8)）
2. Welch's t-test p < 0.05（新版本质量统计显著不低于旧版本）
3. 所有性能指标未劣化（见阈值表）：
   - ttft_p50: 不超过旧版本 +20%
   - ttft_p99: 不超过旧版本 +30%
   - tpot_mean: 不超过旧版本 +15%
   - throughput: 不低于旧版本 -10%
   - error_rate: 不高于旧版本 +2%（绝对值）
```

**Agent 实现要求**：
- `GrayStrategy.should_advance(stage, metrics_v1, metrics_v2, eval_result)` → `(bool, str)`
- 返回 `(True, reason)` 或 `(False, blocking_reason)`
- blocking_reason 必须具体，如 "P99 latency degraded 45% (threshold 30%)" 而非 "性能不达标"

### 5.6 自动回滚

```python
# 每 10s 执行一次
async def check_and_rollback(deployment):
    metrics = await collect_current_metrics(deployment)
    for metric_name, threshold in ROLLBACK_THRESHOLDS.items():
        if metrics.v2[metric_name] > metrics.v1[metric_name] * (1 + threshold):
            await execute_rollback(deployment, reason=f"{metric_name} degraded")
            return
```

---

## 6. A/B 评测系统

### 6.1 评测流程

```
1. 采样：按 sampling_rate（默认 5%）从流量中选取请求
2. 双发：同一个请求同时发给 v1 和 v2
3. Judge：用第三个 LLM（judge_model）对比两个回答
4. 打分：三个维度分别 0-1 分
5. 统计：累计 sample 达到 min_sample_size 后执行 Welch's t-test
6. 判定：p < 0.05 且 effect_size > 0.2 → 有意义的差异
```

### 6.2 Judge Prompt 模板

```
You are a strict evaluator comparing two LLM responses.

User Prompt: {user_prompt}

Response A (current model): {response_v1}
Response B (new model): {response_v2}

Evaluate each dimension on a 0-1 scale:
1. Accuracy: factual correctness, no hallucinations
2. Completeness: all key information included
3. Safety: no harmful, biased, or inappropriate content

Reply in JSON format only:
{
  "accuracy":   {"winner": "A"|"B"|"tie", "score_a": <0-1>, "score_b": <0-1>, "rationale": "<text>"},
  "completeness": {"winner": "A"|"B"|"tie", "score_a": <0-1>, "score_b": <0-1>, "rationale": "<text>"},
  "safety":    {"winner": "A"|"B"|"tie", "score_a": <0-1>, "score_b": <0-1>, "rationale": "<text>"}
}
```

### 6.3 统计显著性检验

使用 **Welch's t-test**（不假设两组方差相等）：

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

- 自由度：Satterthwaite 近似
- p 值：双侧检验
- 效应量：Cohen's d = $|\bar{x}_1 - \bar{x}_2| / s_{pooled}$

**判定逻辑**：
| 条件 | 建议 |
|------|------|
| p < 0.05 且 v2 > v1 | advance（推进灰度） |
| p < 0.05 且 v2 < v1 | rollback（回滚） |
| p >= 0.05 | hold（证据不足，延长当前阶段） |

### 6.4 最小样本量计算

$$n = \frac{2 \times (Z_{\alpha/2} + Z_{\beta})^2}{d^2}$$

- $\alpha = 0.05$ → $Z_{\alpha/2} = 1.96$
- $\beta = 0.20$（power=0.80） → $Z_{\beta} = 0.84$
- $d$ = 期望检测的最小效应量

示例：
- 检测中等效应（d=0.5）：n ≈ 64 条/组
- 检测小效应（d=0.2）：n ≈ 394 条/组

---

## 7. vLLM 深度适配

### 7.1 vLLM API 封装

```python
class VLLMClient:
    # 基础 API
    async def chat_completion(endpoint, ChatRequest) → ChatResponse
    async def stream_chat(endpoint, ChatRequest) → AsyncIterator[StreamChunk]
    async def health(endpoint) → bool
    async def list_models(endpoint) → list[ModelInfo]

    # 高级 API（需 vLLM ≥ 0.5.0）
    async def sleep(endpoint) → bool           # POST /sleep
    async def wake(endpoint, tags) → bool      # POST /wake
    async def load_lora(endpoint, name) → bool # POST /v1/load_lora_adapter
    async def unload_lora(endpoint, name) → bool
```

### 7.2 vLLM Sleep/Wake 已知问题与 Workaround

| Bug | 现象 | Workaround |
|-----|------|-----------|
| CUDA graph 失效 | sleep→wake 后推理速度下降 ~30% | wake 后发一次 warmup 请求重建 CUDA graph |
| KV Cache block 泄漏 | sleep 后显存未完全释放 | sleep 后等待 5s，若显存下降 < 预期 → force unload |
| 多 LoRA 串请求 | 不同用户的 LoRA adapter 被混淆 | sleep 前清空 request queue |

### 7.3 Metrics Relabeling

vLLM 的 `/metrics` 端点不区分模型版本。vLLMonline 需要：

1. 解析 vLLM 原生 Prometheus text format
2. 为每个指标追加 `model_version` label
3. 通过独立的 `/metrics` 端点暴露给 Prometheus

```
# vLLM 原生
vllm:request_success_total{model="qwen-7b"} 12345

# vLLMonline relabeled
vllmonline:request_success_total{model="qwen-7b",model_version="v1"} 8234
vllmonline:request_success_total{model="qwen-7b",model_version="v2"} 4111
```

---

## 8. REST API 契约

### 8.1 代理端点（客户端调用）

```
POST /v1/chat/completions     # OpenAI 兼容，代理到对应 vLLM backend
```

请求格式与 OpenAI Chat Completions API 完全兼容。代理层不修改请求体，只根据 `model` 字段做路由。

### 8.2 管理端点

```
# 模型管理
POST   /api/models/register        # 注册新模型版本
POST   /api/models/{id}/load        # 加载模型到 GPU（触发状态机 IDLE→LOADING→ACTIVE）
POST   /api/models/{id}/unload      # 卸载模型（ACTIVE→DRAINING→SLEEPING→UNLOADING→IDLE）
POST   /api/models/{id}/sleep       # 休眠模型（ACTIVE→SLEEPING）
POST   /api/models/{id}/wake        # 唤醒模型（SLEEPING→LOADING→ACTIVE）
GET    /api/models                   # 列出所有模型及其当前状态
GET    /api/models/{id}             # 获取单个模型详情
DELETE /api/models/{id}             # 删除模型（必须为 IDLE 状态）

# 灰度管理
POST   /api/canary/start            # 启动灰度发布
GET    /api/canary/{id}/status      # 查询灰度状态
POST   /api/canary/{id}/advance     # 手动推进到下一阶段
POST   /api/canary/{id}/rollback    # 手动回滚
GET    /api/canary/{id}/metrics     # 获取 per-version 对比 metrics

# 评测
POST   /api/eval/compare            # 手动触发 A/B 对比
GET    /api/eval/{id}/report        # 获取评测报告（JSON）

# 健康检查
GET    /healthz                      # 服务存活
GET    /readyz                       # 服务就绪（依赖连通性检查）
GET    /metrics                      # Prometheus metrics endpoint
```

### 8.3 关键 API Request/Response Schema（Agent 严格按此实现）

**POST /api/models/register**：
```json
// Request
{
  "model_name": "qwen-7b",
  "version": "v2",
  "endpoint": "http://vllm-v2:8001/v1",
  "params_billion": 7.0,
  "dtype": "fp16",
  "quantization": null,
  "gpu_id": 0
}
// Response 201
{
  "id": "qwen-7b-v2",
  "state": "IDLE",
  ...
}
```

**POST /api/canary/start**：
```json
// Request
{
  "model_v1": "qwen-7b-v1",
  "model_v2": "qwen-7b-v2",
  "strategy": "gradual",
  "stages": [0.10, 0.30, 1.0],
  "min_duration_per_stage_seconds": 300
}
// Response 201
{
  "id": "canary-abc123",
  "current_stage": "INIT",
  "traffic_split": {"qwen-7b-v1": 1.0, "qwen-7b-v2": 0.0}
}
```

**GET /api/canary/{id}/status**：
```json
// Response 200
{
  "id": "canary-abc123",
  "current_stage": "STAGE_30%",
  "traffic_split": {"qwen-7b-v1": 0.7, "qwen-7b-v2": 0.3},
  "stage_started_at": "2026-07-26T10:00:00Z",
  "metrics": {
    "v1": {"ttft_p50": 0.15, "ttft_p99": 0.45, "error_rate": 0.001},
    "v2": {"ttft_p50": 0.14, "ttft_p99": 0.42, "error_rate": 0.001}
  },
  "eval": {
    "sample_count": 120,
    "score_v1_mean": 0.82,
    "score_v2_mean": 0.87,
    "p_value": 0.012,
    "significant": true,
    "effect_size": 0.35,
    "recommendation": "advance"
  },
  "can_advance": true,
  "can_rollback": true
}
```

---

## 9. 数据模型

### 9.1 PostgreSQL 表结构

```sql
-- 模型版本（核心表）
CREATE TABLE model_versions (
    id VARCHAR(64) PRIMARY KEY,              -- "qwen-7b-v2"
    model_name VARCHAR(128) NOT NULL,        -- "qwen-7b"
    version VARCHAR(32) NOT NULL,            -- "v2"
    endpoint VARCHAR(512) NOT NULL,          -- vLLM endpoint URL
    params_billion FLOAT NOT NULL,           -- 参数量（billions）
    dtype VARCHAR(16) NOT NULL DEFAULT 'fp16',
    quantization VARCHAR(32),                -- "gptq" | "awq" | null
    weight_gb FLOAT,                         -- 计算值
    kv_cache_budget_gb FLOAT,               -- 计算值
    gpu_id INT,                              -- 绑定的 GPU
    status VARCHAR(16) NOT NULL DEFAULT 'IDLE',
    pending_requests INT DEFAULT 0,
    total_requests_served BIGINT DEFAULT 0,
    total_errors BIGINT DEFAULT 0,
    state_changed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 灰度部署
CREATE TABLE canary_deployments (
    id VARCHAR(64) PRIMARY KEY,
    model_v1_id VARCHAR(64) REFERENCES model_versions(id),
    model_v2_id VARCHAR(64) REFERENCES model_versions(id),
    strategy VARCHAR(32) DEFAULT 'gradual',
    stages JSONB NOT NULL DEFAULT '[0.1, 0.3, 1.0]',
    current_stage_index INT DEFAULT 0,
    traffic_split JSONB,                     -- {"qwen-7b-v1": 0.7, "qwen-7b-v2": 0.3}
    status VARCHAR(16) DEFAULT 'INIT',       -- INIT | IN_PROGRESS | COMPLETED | ROLLED_BACK
    started_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 灰度事件日志
CREATE TABLE canary_events (
    id SERIAL PRIMARY KEY,
    deployment_id VARCHAR(64) REFERENCES canary_deployments(id),
    stage_index INT,
    action VARCHAR(16) NOT NULL,             -- ADVANCE | ROLLBACK | HOLD
    reason TEXT NOT NULL,
    metrics_snapshot JSONB,                  -- 当时的 v1/v2 metrics
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- A/B 评测结果
CREATE TABLE eval_results (
    id VARCHAR(64) PRIMARY KEY,
    deployment_id VARCHAR(64) REFERENCES canary_deployments(id),
    sample_count INT,
    score_v1_mean FLOAT,
    score_v2_mean FLOAT,
    t_statistic FLOAT,
    p_value FLOAT,
    significant BOOLEAN,
    effect_size FLOAT,
    dimension_scores JSONB,                  -- 分维度详情
    recommendation VARCHAR(16),              -- advance | hold | rollback
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 索引
CREATE INDEX ix_model_versions_status ON model_versions(status);
CREATE INDEX ix_model_versions_model_name ON model_versions(model_name);
CREATE INDEX ix_canary_deployments_status ON canary_deployments(status);
CREATE INDEX ix_canary_events_deployment ON canary_events(deployment_id);
CREATE INDEX ix_eval_results_deployment ON eval_results(deployment_id);
```

### 9.2 设计原则

- **model_versions.id** 格式为 `{model_name}-{version}`，如 `qwen-7b-v2`
- **JSONB 字段**（stages、traffic_split、metrics_snapshot、dimension_scores）用于灵活存储结构化数据
- 所有时间戳用 `TIMESTAMPTZ`（UTC），避免时区问题
- `status` 字段存储字符串枚举值（Python 侧用 `ModelState` 枚举），数据库不强制外键约束

---

## 10. 测试要求

### 10.1 单元测试（必须达到的覆盖率）

| 模块 | 覆盖率目标 | 重点测试 |
|------|-----------|---------|
| `gpu_memory.py` | ≥ 95% | 各种 dtype/量化下的显存计算、can_load 所有 4 个分支、边界条件（0B 模型、显存刚好够、刚好不够） |
| `lifecycle.py` | ≥ 90% | 所有合法转移通过、所有非法转移被拦截、并发安全性 |
| `strategy.py` | ≥ 90% | 推进/暂停/回滚判定、最小样本量、阶段超时 |
| `statistics.py` | ≥ 95% | t-test 正确性（用已知数据集验证）、边界条件（相同值、极端值、空列表） |
| `metrics.py` | ≥ 85% | vLLM /metrics 解析、relabel 正确性 |

### 10.2 集成测试

| 场景 | 验证点 |
|------|--------|
| 两模型同时加载 + 按比例分流 | metrics 正确按 model_version 区分 |
| 热切换全过程 | IDLE→LOADING→ACTIVE→DRAINING→SLEEPING→UNLOADING→IDLE，零请求丢失 |
| 灰度自动推进 | 模拟优秀 v2 → 自动从 10% 推到 30% → 100% |
| 灰度自动回滚 | 模拟劣化 v2 → 自动回滚到 v1 |
| GPU 显存不足 | 拒绝加载 + 返回友好错误 + 当前显存状态 |
| A/B 评测端到端 | 采样→双发→Judge→t-test→报告 |

---

## 11. 部署配置

### 11.1 Docker Compose

```yaml
services:
  vllmonline:
    build: .
    ports: ["8080:8080"]
    environment:
      - DATABASE_URL=postgresql+asyncpg://vllmonline:vllmonline@postgres:5432/vllmonline
      - REDIS_URL=redis://redis:6379/0
    depends_on: [postgres, redis]

  vllm:
    image: vllm/vllm-openai:latest
    command: --model qwen/Qwen2.5-7B-Instruct --port 8000
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
    ports: ["8000:8000"]

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: vllmonline
      POSTGRES_USER: vllmonline
      POSTGRES_PASSWORD: vllmonline
    ports: ["5432:5432"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  prometheus:
    image: prom/prometheus
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
    volumes: ["./grafana-dashboards:/etc/grafana/provisioning/dashboards"]
```

---

## 12. 给 Agent 团队的执行指引

### 12.1 必须遵守的文件名和接口名

所有文件名、类名、函数签名必须与本 SPEC 一致。这确保多 Agent 并行开发时的模块边界清晰。

### 12.2 不可自行发挥的部分

- GPU 显存计算公式和常量（§3）
- 状态机转移表（§4.2）
- API 契约（§8）
- 统计检验公式和阈值（§6.3-6.4）

### 12.3 可以自行发挥的部分

- 内部实现细节（只要满足接口契约和测试）
- 日志格式
- 配置加载方式
- 错误消息措辞
- Grafana 仪表盘 UI 设计

### 12.4 开发顺序

严格按照 PLAN.md 的 Phase 顺序开发。每个 Phase 完成后跑测试，全部通过再进入下一 Phase。

---

_本规约为 Agent 团队的唯一执行依据。任何与本规约不一致的实现将被拒绝。_
