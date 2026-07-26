#!/bin/bash
# vLLMonline + vLLM 一键启动脚本（MI300X 单 pod 部署）
#
# 用法：bash scripts/start_all.sh
#
# 启动三个服务（各自 tmux session，断开 SSH 不挂）：
#   vllm-v1     Qwen2.5-7B-Instruct  端口 8000（baseline）
#   vllm-v2     Qwen2.5-1.5B-Instruct 端口 8001（灰度候选）
#   vllmonline  代理 + 管理面          端口 8080
#
# 数据库：SQLite 文件（持久化到 /mnt/workspace/vllmonline/vllmonline.db）
# GPU：MI300X 192GB（v1 占 30% util ≈ 58GB，v2 占 20% util ≈ 38GB）

set -e

PROJECT_DIR="/mnt/workspace/vllmonline"
MODELSCOPE_DIR="/mnt/workspace/modelscope"
V1_MODEL="${V1_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
V2_MODEL="${V2_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
VLLM_V1_UTIL="${VLLM_V1_UTIL:-0.30}"
VLLM_V2_UTIL="${VLLM_V2_UTIL:-0.20}"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== vLLMonline 一键启动 ===${NC}"
echo ""

# 1. 启动 vLLM v1
echo -e "${YELLOW}[1/3] 启动 vLLM v1（${V1_MODEL}，端口 8000）...${NC}"
tmux kill-session -t vllm-v1 2>/dev/null || true
tmux new-session -d -s vllm-v1 \
  "vllm serve ${MODELSCOPE_DIR}/${V1_MODEL} \
    --served-model-name qwen-7b \
    --port 8000 --host 0.0.0.0 \
    --gpu-memory-utilization ${VLLM_V1_UTIL} \
    --max-model-len 8192 2>&1 | tee /tmp/vllm-v1.log"

# 2. 启动 vLLM v2
echo -e "${YELLOW}[2/3] 启动 vLLM v2（${V2_MODEL}，端口 8001）...${NC}"
tmux kill-session -t vllm-v2 2>/dev/null || true
tmux new-session -d -s vllm-v2 \
  "vllm serve ${MODELSCOPE_DIR}/${V2_MODEL} \
    --served-model-name qwen-7b \
    --port 8001 --host 0.0.0.0 \
    --gpu-memory-utilization ${VLLM_V2_UTIL} \
    --max-model-len 8192 2>&1 | tee /tmp/vllm-v2.log"

# 3. 等 vLLM 起来
echo -e "${YELLOW}等待 vLLM 加载模型（约 60-90 秒）...${NC}"
for port in 8000 8001; do
  for i in $(seq 1 60); do
    if curl -s -m 1 "http://localhost:${port}/health" >/dev/null 2>&1; then
      echo "  端口 ${port} 就绪"
      break
    fi
    sleep 2
  done
done

# 4. 启动 vllmonline
echo -e "${YELLOW}[3/3] 启动 vllmonline（端口 8080）...${NC}"
tmux kill-session -t vllmonline 2>/dev/null || true
tmux new-session -d -s vllmonline \
  "cd ${PROJECT_DIR} && \
   export VLLM_BACKEND_URL=http://localhost:8000 && \
   export VLLMONLINE_DATABASE__URL='sqlite+aiosqlite:////mnt/workspace/vllmonline/vllmonline.db' && \
   export VLLMONLINE_GPU__BACKEND=rocm && \
   export VLLMONLINE_ENVIRONMENT=production && \
   PYTHONPATH=. python -m uvicorn vllmonline.server:app --host 0.0.0.0 --port 8080 2>&1 | tee /tmp/vllmonline.log"

# 等 vllmonline 起来
for i in $(seq 1 15); do
  if curl -s -m 1 "http://localhost:8080/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo ""
echo -e "${GREEN}=== 全部启动完成 ===${NC}"
echo ""
echo "服务地址："
echo "  vLLM v1:        http://localhost:8000"
echo "  vLLM v2:        http://localhost:8001"
echo "  vllmonline API: http://localhost:8080"
echo "  API 文档:       http://localhost:8080/docs"
echo ""
echo "查看日志："
echo "  tmux attach -t vllm-v1       # v1 vLLM 日志"
echo "  tmux attach -t vllm-v2       # v2 vLLM 日志"
echo "  tmux attach -t vllmonline    # vllmonline 日志"
echo "  （Ctrl+B 然后 D 退出 tmux，进程继续跑）"
echo ""
echo -e "${YELLOW}注意：vllmonline 启动后需要手动注册模型版本：${NC}"
echo "  curl -X POST http://localhost:8080/api/models/register \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model_name\":\"qwen-7b\",\"version\":\"v1\",\"endpoint\":\"http://localhost:8000\",\"params_billion\":7.0,\"dtype\":\"fp16\"}'"
echo "  curl -X POST http://localhost:8080/api/models/qwen-7b-v1/load"
