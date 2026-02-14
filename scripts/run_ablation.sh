#!/bin/bash
# 消融实验：对多组 checkpoint 运行评估并生成对比图与表格
# 使用方式：
#   1) 先分别训练各变体，将 final.pt 放到 outputs/ablation/<变体名>/ 下
#   2) 执行: bash scripts/run_ablation.sh
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

OUT="${1:-outputs/ablation}"
NUM_BATCHES="${2:-15}"
echo "Output: $OUT, Num batches: $NUM_BATCHES"
python scripts/run_ablation.py --mode eval --output_base "$OUT" --num_batches "$NUM_BATCHES" \
  --variants full no_align no_robust \
  --checkpoint_dir "$OUT"
