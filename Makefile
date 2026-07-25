# vLLMonline Makefile
# 所有常用命令封装，开发期统一入口。详见 DEVELOPMENT.md。
#
# 使用：make <target>
# 常用：make install | make test | make check | make compose-up

PYTHON := uv run python
PYTEST := uv run pytest
RUFF   := uv run ruff
MYPY   := uv run mypy

.DEFAULT_GOAL := help

.PHONY: help
help:  ## 显示所有可用命令
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ─── 环境 ────────────────────────────────────────────────────────────────────

.PHONY: install
install:  ## 安装依赖（uv sync，含 dev+test 组）
	uv sync

.PHONY: install-prod
install-prod:  ## 仅装生产依赖（不含 dev/test）
	uv sync --no-default-groups

.PHONY: upgrade
upgrade:  ## 升级所有依赖到最新（更新 uv.lock）
	uv lock --upgrade

.PHONY: lock
lock:  ## 重新生成 uv.lock（不改版本，只刷新）
	uv lock

.PHONY: requirements-txt
requirements-txt:  ## 导出扁平 requirements.txt（给不用 uv 的环境用）
	uv export --format requirements-txt --no-dev --output-file requirements.txt
	@echo "已生成 requirements.txt（生产依赖）"

.PHONY: clean
clean:  ## 清理缓存和构建产物
	rm -rf .venv .ruff_cache .mypy_cache .pytest_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ─── 测试 ────────────────────────────────────────────────────────────────────

.PHONY: test
test:  ## 跑全部单测（默认 SQLite，不含 integration）
	$(PYTEST) -m "not integration"

.PHONY: test-all
test-all:  ## 跑全部测试（含 integration，需要 Docker）
	$(PYTEST)

.PHONY: test-integration
test-integration:  ## 仅跑集成测试（需要 Docker 启动 testcontainers）
	$(PYTEST) -m "integration"

.PHONY: test-cov
test-cov:  ## 跑测试并强制检查覆盖率（失败则退出码非 0）
	$(PYTEST) --cov-fail-under=85 -m "not integration"

.PHONY: test-verbose
test-verbose:  ## 详细模式跑测试
	$(PYTEST) -v -s -m "not integration"

# ─── 代码质量 ────────────────────────────────────────────────────────────────

.PHONY: lint
lint:  ## Ruff lint
	$(RUFF) check .

.PHONY: format
format:  ## Ruff format（自动格式化）
	$(RUFF) format .

.PHONY: format-check
format-check:  ## 检查格式是否正确（CI 用，不修改文件）
	$(RUFF) format --check .

.PHONY: typecheck
typecheck:  ## Mypy 类型检查
	$(MYPY) vllmonline

.PHONY: check
check: lint format-check typecheck  ## 一键检查：lint + format + mypy

# ─── Docker Compose ──────────────────────────────────────────────────────────

COMPOSE_DIR := deploy/docker-compose
COMPOSE := docker compose --project-directory . -f $(COMPOSE_DIR)/docker-compose.yml

.PHONY: compose-up
compose-up:  ## 启动全套服务（MI300X 实机：vllmonline + vLLM + PG + Redis + Prom + Grafana）
	$(COMPOSE) up -d

.PHONY: compose-up-dev
compose-up-dev:  ## 启动开发环境（无 GPU：vLLM 被 fake server 替代）
	$(COMPOSE) -f $(COMPOSE_DIR)/docker-compose.yml -f $(COMPOSE_DIR)/docker-compose.override.dev.yml up -d

.PHONY: compose-down
compose-down:  ## 停止全部服务
	$(COMPOSE) down

.PHONY: compose-logs
compose-logs:  ## 跟踪全部服务日志
	$(COMPOSE) logs -f

.PHONY: compose-ps
compose-ps:  ## 查看服务状态
	$(COMPOSE) ps

# ─── 数据库 migration ────────────────────────────────────────────────────────

.PHONY: db-migrate
db-migrate:  ## 应用所有 migration 到目标数据库
	$(PYTHON) -m alembic upgrade head

.PHONY: db-revision
db-revision:  ## 自动生成 migration（用法：make db-revision m="add xxx table"）
	@test -n "$(m)" || (echo "用法: make db-revision m=\"描述信息\"" && exit 1)
	$(PYTHON) -m alembic revision --autogenerate -m "$(m)"

.PHONY: db-reset
db-reset:  ## 重置数据库（开发用，慎用）
	$(PYTHON) -m alembic downgrade base
	$(PYTHON) -m alembic upgrade head

# ─── 运行 ────────────────────────────────────────────────────────────────────

.PHONY: run
run:  ## 本地启动 vllmonline 服务（开发模式，热重载）
	$(PYTHON) -m vllmonline

.PHONY: run-prod
run-prod:  ## 生产模式启动
	$(PYTHON) -m uvicorn vllmonline.server:app --host 0.0.0.0 --port 8080 --workers 4

# ─── Git hooks ───────────────────────────────────────────────────────────────

.PHONY: precommit-install
precommit-install:  ## 安装 pre-commit git hooks
	uv run pre-commit install

.PHONY: precommit-run
precommit-run:  ## 手动跑所有 pre-commit hooks
	uv run pre-commit run --all-files
