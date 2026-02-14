#!/bin/bash
# GenMamba-Flow 训练入口
# 使用默认配置
set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CONFIG="${1:-configs/default.yaml}"
OVERRIDE="${2:-configs/train.yaml}"
DEVICE="${3:-cuda}"

echo "Config: $CONFIG, Override: $OVERRIDE, Device: $DEVICE"
python train.py --config "$CONFIG" --override "$OVERRIDE" --device "$DEVICE"
