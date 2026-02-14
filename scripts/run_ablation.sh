#!/bin/bash
# 消融实验：可通过覆盖配置关闭三流/干扰引导等（需在 config 中增加对应开关后再用）
# 示例：训练 baseline（无干扰引导）时可将 loss.lambda_robust 设为 0
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 完整模型
echo "Training full model..."
python train.py --config configs/default.yaml --override configs/train.yaml --device "${1:-cuda}"

# 无鲁棒引导：先复制配置并设 lambda_robust=0，再训练
# cp configs/train.yaml configs/train_no_robust.yaml
# 编辑 configs/train_no_robust.yaml 添加 loss.lambda_robust: 0
# python train.py --config configs/default.yaml --override configs/train_no_robust.yaml --device cuda
