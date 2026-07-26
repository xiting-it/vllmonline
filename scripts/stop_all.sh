#!/bin/bash
# vLLMonline + vLLM 一键停止脚本
#
# 用法：bash scripts/stop_all.sh
#
# 停止顺序（先停依赖方）：
#   1. vllmonline（避免向 vLLM 发新请求）
#   2. vLLM v2
#   3. vLLM v1
#   4. 杀残留进程
#   5. 显示 GPU 显存释放情况

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== vLLMonline 一键停止 ===${NC}"
echo ""

# 1. 停 vllmonline
echo "[1/4] 停止 vllmonline..."
tmux kill-session -t vllmonline 2>/dev/null && echo "  已停" || echo "  未运行"

# 2. 停 vLLM v2
echo "[2/4] 停止 vLLM v2..."
tmux kill-session -t vllm-v2 2>/dev/null && echo "  已停" || echo "  未运行"

# 3. 停 vLLM v1
echo "[3/4] 停止 vLLM v1..."
tmux kill-session -t vllm-v1 2>/dev/null && echo "  已停" || echo "  未运行"

# 4. 杀残留
echo "[4/4] 清理残留进程..."
pkill -9 -f "vllm serve" 2>/dev/null && echo "  已清 vllm 残留" || echo "  无残留"
pkill -9 -f "vllmonline.server" 2>/dev/null && echo "  已清 vllmonline 残留" || echo "  无残留"

sleep 2

echo ""
echo -e "${GREEN}=== 停止完成 ===${NC}"
echo ""
echo "残留进程检查（应为空）："
ps aux | grep -iE "vllm|vllmonline" | grep -v grep || echo "  无残留"
echo ""
echo "GPU 显存状态："
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showmeminfo vram
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv
else
  echo "  无 GPU 工具"
fi
