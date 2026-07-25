# 开发指南

> 本文档记录开发环境的所有依赖、工具链用法、踩坑点。
> 完整内容在 Phase 6 补全，此处先记录关键信息。

## 环境要求

| 工具 | 版本要求 | 用途 | 安装方式 |
|------|---------|------|---------|
| **Python** | 3.12.x（严格 ≥3.12,<3.13） | 运行时 | uv 会自动下载 |
| **uv** | ≥ 0.11 | 包管理 + 虚拟环境 | `curl -LsSf https://astral.sh/install.sh \| sh` |
| **Docker** | ≥ 24.0 | 跑集成测试 + 部署（仅 MI300X 实机需要） | 系统包管理器 |
| **ROCm 驱动** | 7.2.1+ | GPU 访问（仅 MI300X 宿主机） | AMD 官方文档 |
| **git** | ≥ 2.30 | 版本控制 | 系统自带 |

**无 GPU 也能完整开发**：所有单测在 CPU 上跑通，fake vLLM server 模拟全部 vLLM 行为。

## 常用命令

```bash
make install       # 装依赖（uv sync）
make test          # 跑单测（不含 integration）
make test-all      # 跑全部测试（需要 Docker）
make check         # ruff + mypy 一键检查
make run           # 本地起服务（热重载）
make compose-up-dev  # Docker 起全套（fake vLLM，无需 GPU）
make compose-up    # Docker 起全套（MI300X 实机）
```

## 依赖说明

- **生产依赖**：见 `pyproject.toml` 的 `[project.dependencies]`
- **开发依赖**：见 `[dependency-groups]` 的 `dev` 和 `test` 组
- **扁平 requirements.txt**：`make requirements-txt` 导出（给不用 uv 的环境用）

所有版本范围基于 2026-07 的 PyPI 调研，详见各依赖注释。

## 测试策略

- **单测**（默认）：纯 Python，用 SQLite in-memory + fake vLLM（ASGITransport），零网络零 GPU
- **集成测试**（`-m integration`）：用 testcontainers 起真实 PostgreSQL 容器
- **覆盖率门槛**：整体 ≥85%，核心模块（gpu_memory/lifecycle/statistics）≥90-95%

## 已知问题 / 踩坑

详见代码内注释（搜 `NOTE`、`FIXME`、`SPEC`）。
