#!/bin/bash
# GenMamba-Flow 测试与评估入口
set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CONFIG="${1:-configs/default.yaml}"
CKPT="${2:-outputs/final.pt}"
OUT="${3:-outputs/eval}"
DEVICE="${4:-cuda}"

echo "Config: $CONFIG, Checkpoint: $CKPT, Output: $OUT"
python test.py --config "$CONFIG" --checkpoint "$CKPT" --output_dir "$OUT" --device "$DEVICE" --num_batches 20

# 鲁棒性曲线（可选）
# python test.py --config "$CONFIG" --checkpoint "$CKPT" --output_dir "$OUT" --device "$DEVICE" --robustness_only
