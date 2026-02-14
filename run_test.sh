#!/bin/bash
# GenMamba-Flow 测试：单脚本执行全部评估与可视化，自动使用多卡（DataParallel）
set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CONFIG="${1:-configs/default.yaml}"
CKPT="${2:-outputs/final.pt}"
OUT="${3:-outputs/eval}"
NUM_BATCHES="${4:-20}"

echo "Config: $CONFIG, Checkpoint: $CKPT, Output: $OUT, Num batches: $NUM_BATCHES"
python test.py --config "$CONFIG" --checkpoint "$CKPT" --output_dir "$OUT" --num_batches "$NUM_BATCHES" --multi_gpu
