# vLLMonline

> vLLM 推理引擎上层管理平台：**零停机模型热切换 + 灰度发布 + A/B 自动评测 + 劣化自动回滚**。

[![tests](https://img.shields.io/badge/tests-407%20passed-brightgreen)](#测试)
[![coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)](#测试)
[![python](https://img.shields.io/badge/python-3.12-blue)](#环境要求)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

---

## 这是什么

当前企业上线新模型的典型流程：改配置文件 → 重启 vLLM → 3 分钟冷启动 → 全量切换 → 祈祷别出问题。

**vLLMonline 替代方案**：

```
上传新模型 → GPU 显存安全校验 → 加载到同 GPU → 10% 流量灰度
→ 自动对比新旧模型（延迟/吞吐/质量）→ 达标自动全量，不达标自动回滚
全程零停机，客户端无感知。
```

### 核心特性

| 特性 | 说明 |
|------|------|
| 🔥 **零停机热切换** | 同 GPU 上多模型共存，drain + sleep/unload 实现请求零丢失 |
| 📊 **统计驱动灰度** | Welch's t-test (p<0.05) 决定推进，不靠人工感觉 |
| 🤖 **LLM-as-Judge** | 三维度（准确/完整/安全）自动评测对比新旧模型 |
| ⚡ **劣化自动回滚** | 任一性能指标超阈值，10 秒内自动切回旧版本 |
| 🎯 **MI300X 原生** | AMD ROCm 适配（rocm-smi 显存查询），192GB HBM3 |
| 📈 **可观测** | per-version Prometheus metrics，Grafana 同维度对比 |

---

## 5 分钟快速开始

### 前置要求

| 工具 | 版本 | 用途 |
|------|------|------|
| [uv](https://docs.astral.sh/uv/) | ≥ 0.11 | Python 包管理 |
| Docker | ≥ 24.0 | 跑 docker-compose（实机部署） |
| ROCm 驱动 | 7.2.1+ | MI300X GPU 访问（仅实机需要） |

### 开发环境（无需 GPU）

```bash
# 1. 克隆 + 装依赖
git clone <repo-url> && cd vllmonline
make install          # uv sync

# 2. 跑测试验证
make test             # 407 个单测全过
make check            # ruff + mypy

# 3. 启动开发服务（fake vLLM，CPU 运行）
make compose-up-dev
curl http://localhost:8080/healthz   # {"status":"ok"}
```

### MI300X 生产部署

```bash
# 1. 一键启动全套（vLLMonline + ROCm vLLM + PG + Redis + Prometheus + Grafana）
make compose-up

# 2. 注册模型 + 加载
curl -X POST http://localhost:8080/api/models/register \
  -H "Content-Type: application/json" \
  -d '{"model_name":"qwen-7b","version":"v1","endpoint":"http://vllm-v1:8000/v1","params_billion":7.0,"dtype":"fp16"}'

curl -X POST http://localhost:8080/api/models/qwen-7b-v1/load

# 3. 代理请求（OpenAI 兼容）
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-7b","messages":[{"role":"user","content":"hello"}]}'

# 4. 灰度发布新版本
curl -X POST http://localhost:8080/api/canary/start \
  -H "Content-Type: application/json" \
  -d '{"model_v1":"qwen-7b-v1","model_v2":"qwen-7b-v2","stages":[0.1,0.3,1.0]}'

# 5. 查看状态 + Grafana 仪表盘
open http://localhost:3000   # admin/admin
```

完整 API 文档：启动后访问 `http://localhost:8080/docs`。

### MI300X 单 Pod 部署（✅ 已实测跑通）

适用于 K8s/PAI/DSW 等"GPU pod 内直连、无 Docker daemon"的场景。
vLLM 与 vllmonline 作为同 pod 内的两个进程跑，DB 用 SQLite 文件。

**前置**：pod 内已装 vLLM（`pip show vllm` 能看到 `0.20.1+rocm721`），`/dev/kfd` + `/dev/dri/` 可访问，`rocm-smi` 能看到 GPU。

**一键启动**（启 3 个 tmux session：vLLM v1 + vLLM v2 + vllmonline）：

```bash
bash scripts/start_all.sh
```

脚本会：
1. 用 tmux 后台起 vLLM v1（Qwen2.5-7B，端口 8000，30% 显存）
2. 用 tmux 后台起 vLLM v2（Qwen2.5-1.5B，端口 8001，20% 显存）
3. 等 vLLM 健康检查通过后起 vllmonline（端口 8080，SQLite 文件 DB）

**一键停止**：

```bash
bash scripts/stop_all.sh
```

**手动注册模型**（启动后跑一次，配置存在 SQLite，重启 vllmonline 不丢）：

```bash
# 注册并加载 v1（baseline）
curl -X POST http://localhost:8080/api/models/register \
  -H "Content-Type: application/json" \
  -d '{"model_name":"qwen-7b","version":"v1","endpoint":"http://localhost:8000","params_billion":7.0,"dtype":"fp16"}'
curl -X POST http://localhost:8080/api/models/qwen-7b-v1/load

# 注册并加载 v2（灰度候选）
curl -X POST http://localhost:8080/api/models/register \
  -H "Content-Type: application/json" \
  -d '{"model_name":"qwen-7b","version":"v2","endpoint":"http://localhost:8001","params_billion":1.5,"dtype":"fp16"}'
curl -X POST http://localhost:8080/api/models/qwen-7b-v2/load

# 启动灰度（v2 占 10%）
curl -X POST http://localhost:8080/api/canary/start \
  -H "Content-Type: application/json" \
  -d '{"model_v1":"qwen-7b-v1","model_v2":"qwen-7b-v2","stages":[0.1,0.3,1.0]}'
```

**查看日志**：

```bash
tmux attach -t vllm-v1       # v1 vLLM 日志（Ctrl+B D 退出）
tmux attach -t vllm-v2       # v2 vLLM 日志
tmux attach -t vllmonline    # vllmonline 日志
```

**实测验证结果**（MI300X 192GB，2026-07-26）：

| 场景 | 结果 |
|------|------|
| v1 (7B) + v2 (1.5B) 同 GPU 共存 | ✅ 显存 97GB / 192GB（50%） |
| 10% 灰度分流 20 请求 | ✅ v1:17 / v2:3（实测 85%/15%） |
| 推进到 30% 灰度 | ✅ v2 流量占比上升 |
| 手动回滚 | ✅ v2 自动 SLEEPING，后续流量 100% 回 v1 |
| per-version metrics | ✅ `model_version="v1"/"v2"` label 正确区分 |
| 状态机保护 | ✅ ACTIVE→LOADING 非法转移被拦截 |

---

## 架构

```
                           ┌──────────────────────────────────────┐
   Client ──HTTP──────────▶│  vllmonline Proxy (:8080)            │
                           │  ├─ 路由决策（加权随机选版本）            │
                           │  ├─ 注入 header: x-model-version      │
                           │  ├─ 转发到 vLLM backend               │
                           │  └─ 采集 per-request metrics          │
                           └──────┬───────────────┬───────────────┘
                                  │               │
                    ┌─────────────▼──┐   ┌────────▼──────────┐
                    │ vLLM (v1)      │   │ vLLM (v2)         │
                    │ rocm/vllm      │   │ rocm/vllm         │
                    │ :8000          │   │ :8001             │
                    └────────────────┘   └───────────────────┘
                                  │               │
                                  ▼               ▼
                              MI300X 192GB HBM3（同一 GPU 上两个实例）
```

### 核心子系统

| 子系统 | 模块 | 说明 |
|--------|------|------|
| GPU 显存管理 | `scheduler/gpu_memory.py` | 权重/KV cache 计算 + `can_load()` 决策树 |
| 模型状态机 | `scheduler/lifecycle.py` | 7 状态完备状态机，非法转移抛异常 |
| 热切换编排 | `scheduler/engine.py` | `hot_swap()` 编排零停机切换 |
| 流量路由 | `router/proxy.py` | 加权随机分流，支持灰度配比 |
| 灰度策略 | `canary/strategy.py` | 阶梯放量 + 统计推进判定 |
| A/B 评测 | `eval/statistics.py` + `eval/judge.py` | Welch t-test + LLM-as-Judge |
| 自动回滚 | `canary/rollback.py` | 后台巡检，劣化自动切回 |

完整架构与接口契约见 [SPEC.md](SPEC.md)，开发计划见 [PLAN.md](PLAN.md)。

---

## API 速览

### 代理端点（客户端调用）

```
POST /v1/chat/completions     # OpenAI 兼容，自动路由到对应版本
```

### 管理端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/models/register` | 注册新模型版本 |
| POST | `/api/models/{id}/load` | 加载模型到 GPU |
| GET | `/api/models` | 列出所有模型 |
| POST | `/api/canary/start` | 启动灰度发布 |
| GET | `/api/canary/{id}/status` | 查询灰度状态 |
| POST | `/api/canary/{id}/advance` | 手动推进阶段 |
| POST | `/api/canary/{id}/rollback` | 手动回滚 |
| POST | `/api/eval/compare` | 触发 A/B 对比评测 |
| GET | `/api/eval/{id}/report` | 获取评测报告 |
| GET | `/metrics` | Prometheus 抓取端点 |

完整 schema 见 `/docs`（启动后）或 [SPEC.md §8](SPEC.md)。

---

## 测试

```bash
make test              # 单测（407 个，无外部依赖，~10s）
make test-integration  # 集成测试（需要 Docker，testcontainers PG）
make test-cov          # 强制覆盖率检查（≥85%）
```

| 模块 | 覆盖率 | SPEC 要求 |
|------|--------|-----------|
| `scheduler/gpu_memory.py` | 100% | ≥95% ✅ |
| `scheduler/lifecycle.py` | 99% | ≥90% ✅ |
| `eval/statistics.py` | 100% | ≥95% ✅ |
| `canary/strategy.py` | 91% | ≥90% ✅ |
| `vllm/metrics.py` | 94% | ≥85% ✅ |
| **整体** | **85%** | ≥85% ✅ |

---

## 验证脚本

真实部署上跑的三个验证脚本（都在 `scripts/`）。详细输出样例见 [BLOG.md](BLOG.md)。

| 脚本 | 用途 |
|------|------|
| `run_ab_eval.py` | A/B 评测：对比两个 vLLM 实例的模型，v1 当 judge 三维度评分（accuracy/completeness/safety） |
| `bench_hotswap.py` | 热切换期间 source rate 压测：20 并发 30 秒，第 15 秒触发回滚，验证零请求丢失 |
| `rollback_timeline.py` | 回滚时间线记录：50ms 高频轮询 v2 状态变化，输出毫秒级时间线 |

**A/B 评测**（输出 t / p / Cohen's d / recommendation）：

```bash
PYTHONPATH=. python scripts/run_ab_eval.py
# 报告保存到 scripts/reports/ab_eval_<timestamp>.md
```

**热切换期间 source rate 压测**（输出成功率 / QPS / P50 / P99 + 回滚前后对比）：

```bash
PYTHONPATH=. python scripts/bench_hotswap.py <deployment_id>
```

**回滚时间线记录**（输出：发起 → API 返回 → v2 SLEEPING → 流量切回）：

```bash
PYTHONPATH=. python scripts/rollback_timeline.py <deployment_id>
```

---

## 实测数据

MI300X 192GB 单 Pod 部署上的实测结果（2026-07-26）：

| 实验 | 关键数据 | 结论 |
|------|----------|------|
| A/B 评测 | n=49, p=0.0001, d=-0.83 | ❌ v2 显著劣于 v1，建议 rollback |
| 回滚时间线 | API 210ms 返回，321ms 流量全切 | ✅ 零停机回滚 |
| source rate | 4640 请求 / 154 QPS / 100% 成功 | ✅ 热切换期间零请求丢失 |
| 灰度分流 | 10% 配比实测 17:3（20 请求） | ✅ 加权路由正确 |
| 状态机保护 | ACTIVE→LOADING 返回 409 | ✅ 非法转移被拦截 |

完整复现步骤与原始日志见 [BLOG.md](BLOG.md)。

---

## 配置

配置来源（优先级高到低）：环境变量 > YAML 文件 > 默认值。

```bash
# 环境变量（前缀 VLLMONLINE_，嵌套用 __）
export VLLMONLINE_DATABASE__URL=postgresql+asyncpg://...
export VLLMONLINE_GPU__CUDA_CONTEXT_OVERHEAD_GB=4.0   # MI300X 推荐
export VLLMONLINE_GPU__BACKEND=rocm

# 或写 config.yaml
cat > config.yaml <<EOF
gpu:
  backend: rocm
  cuda_context_overhead_gb: 4.0
database:
  url: postgresql+asyncpg://vllmonline:vllmonline@postgres:5432/vllmonline
EOF
export VLLMONLINE_CONFIG=config.yaml
```

完整配置项见 [`vllmonline/config.py`](vllmonline/config.py)。

---

## 开发

详见 [DEVELOPMENT.md](DEVELOPMENT.md)。常用命令：

```bash
make install       # 装依赖
make test          # 跑测试
make check         # ruff + mypy
make run           # 本地起服务
make compose-up    # Docker 全套
```

---

## 技术栈

- **Python 3.12** + asyncio 全栈
- **FastAPI** + httpx（代理 + 管理面）
- **SQLAlchemy 2.0** async + PostgreSQL（asyncpg）/ SQLite（aiosqlite）
- **scipy** + numpy（统计检验）
- **prometheus-client** + Grafana（可观测）
- **uv**（包管理）+ ruff + mypy（代码质量）
- **AMD ROCm** vLLM（MI300X 192GB）

---

## License

Apache-2.0
