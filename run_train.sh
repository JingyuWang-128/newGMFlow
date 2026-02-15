#!/bin/bash
# GenMamba-Flow 训练：单脚本执行全部阶段，自动使用当前机器所有 GPU（DDP）
set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CONFIG="${1:-configs/default.yaml}"
OVERRIDE="${2:-configs/train.yaml}"
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
if [ "$NUM_GPUS" -lt 1 ]; then
  NUM_GPUS=1
fi

STAGE="${STAGE:-all}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

echo "Config: $CONFIG, Override: $OVERRIDE, GPUs: $NUM_GPUS, Stage: $STAGE"
if [ "$NUM_GPUS" -eq 1 ]; then
  python train.py --config "$CONFIG" --override "$OVERRIDE" --stage "$STAGE"
else
  torchrun --nproc_per_node="$NUM_GPUS" train.py --config "$CONFIG" --override "$OVERRIDE" --stage "$STAGE"
fi
