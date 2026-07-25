# vLLMonline

> vLLM 推理引擎上层管理平台：**零停机模型热切换 + 灰度发布 + A/B 自动评测 + 劣化自动回滚**。

> ⚠️ 文档占位——完整内容在 Phase 6（P6.1）补全。当前阶段请参考 `SPEC.md` 与 `PLAN.md`。

## 快速开始

```bash
# 开发环境（无需 GPU）
make install        # uv sync 装依赖
make test           # 跑单测

# 生产部署（MI300X）
cd deploy/docker-compose
docker compose up -d
```

详见 `DEVELOPMENT.md`（P6 补全）。
