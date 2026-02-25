# GenMamba-Flow 使用说明：训练、测试、消融

## 环境与数据

```bash
cd /home/wangjingyu/newGMFlow
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

将数据放到配置中的路径（默认见 `configs/default.yaml`）：

- **覆盖图目录**：`data.cover_roots`（如 `DIV2K`、`COCO`）
- **秘密图目录**：`data.secret_roots`（如 `Paris_StreetView`、`CelebA-HQ`）

**自动划分**：代码会从上述目录递归收集所有图片（`.jpg/.png` 等），再按 `data.split_ratios`（默认 `[0.8, 0.1, 0.1]`）和 `data.split_seed`（默认 42）划分为 **训练集 / 验证集 / 测试集**，保证可复现。训练只用训练集，测试只用测试集。启动训练或测试时会在终端打印各集合数量，便于核对。

---

## 一、训练

**支持分阶段训练**：阶段1 只训 RQ-VAE，阶段2 只训生成器+解码器（需先有阶段1 的 `rq_vae.pt`）；也可一次跑两阶段（默认）。

### 分阶段训练（推荐按顺序执行）

| 阶段 | 内容 | 输出 |
|------|------|------|
| **阶段 1** | 仅 RQ-VAE 预训练（重建 + commitment） | `outputs/rq_vae.pt` |
| **阶段 2** | 仅生成器 + 解码器（RQ-VAE 冻结） | `outputs/checkpoint.pt`、`outputs/final.pt` |

**只跑阶段 1：**

```bash
# 脚本（自动多卡）
STAGE=1 bash run_train.sh

# 或直接
python train.py --config configs/default.yaml --override configs/train.yaml --stage 1
```

**只跑阶段 2**（必须先有 `outputs/rq_vae.pt`，即先跑完阶段 1）：

```bash
STAGE=2 bash run_train.sh

# 或
python train.py --config configs/default.yaml --override configs/train.yaml --stage 2
```

**两阶段一起跑**（默认：没有 `rq_vae.pt` 时先阶段1，再阶段2）：

```bash
bash run_train.sh
# 或
python train.py --config configs/default.yaml --override configs/train.yaml
# 等价于 --stage all
```

### 方式 1：用脚本（自动多卡）

```bash
# 默认两阶段
bash run_train.sh
bash run_train.sh configs/default.yaml configs/train.yaml

# 分阶段
STAGE=1 bash run_train.sh
STAGE=2 bash run_train.sh
```

### 方式 2：直接调 Python

```bash
# 单卡
python train.py --config configs/default.yaml --override configs/train.yaml --stage 1
python train.py --config configs/default.yaml --override configs/train.yaml --stage 2

# 多卡
torchrun --nproc_per_node=4 train.py --config configs/default.yaml --override configs/train.yaml --stage 1
torchrun --nproc_per_node=4 train.py --config configs/default.yaml --override configs/train.yaml --stage 2
```

### 输出与可视化

- 权重：`outputs/rq_vae.pt`（阶段1）、`outputs/checkpoint.pt`（定期）、`outputs/final.pt`（最终）
- 训练可视化（在 `outputs/vis/`）：
  - 损失曲线：`loss_curves.png`、`loss_curves_final.png`
  - 损失历史：`loss_history.json`
  - 隐写对比：`stego_step*.png`（cover / stego / secret / recovered）

**显存不足（OOM）**：Stage2 的 DiS（18 层 Mamba、256×256）较吃显存。若报 `CUDA out of memory`：
- 将 `data.batch_size` 调小（如改为 `2` 或 `1`），或
- 在 override 中减小 `generator.model_channels`（如 256）、`generator.num_layers`（如 12），或
- 减小 `decoder.hidden_dim`（默认 384；512 时 decoder 已开梯度检查点，仍 OOM 可改回 384），或
- 设置环境变量：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片。  
默认 `batch_size: 1`、`decoder.hidden_dim: 384`，约 48GB 单卡可跑；24GB 卡建议 `batch_size: 1` 且 generator/decoder 用更小维度。

---

## 二、测试

**一条命令完成**：主指标评估 + 鲁棒性曲线 + 所有对比图；**多卡时自动用 DataParallel**。

### 方式 1：推荐用脚本

```bash
# 默认：config=default.yaml, checkpoint=outputs/final.pt, 结果目录=outputs/eval, 20 个 batch
bash run_test.sh

# 自定义
bash run_test.sh configs/default.yaml outputs/final.pt outputs/eval 20
```

### 方式 2：直接调 Python

```bash
# 全量评估（指标 + 鲁棒性 + 对比图），多卡
python test.py --config configs/default.yaml --checkpoint outputs/final.pt \
  --output_dir outputs/eval --num_batches 20 --multi_gpu

# 只跑鲁棒性曲线
python test.py --checkpoint outputs/final.pt --output_dir outputs/eval --robustness_only --multi_gpu
```

### 输出内容（均在 `--output_dir` 下）

| 文件 | 说明 |
|------|------|
| `metrics.txt` / `metrics.json` | Bit Accuracy、Recovery PSNR、Recovery SSIM |
| `metrics_bars.png` | 上述指标柱状图 |
| `robustness_jpeg_curve.png`、`robustness_jpeg.json` | 不同 JPEG 质量下的 Bit Accuracy |
| `compare_batch*.png` | cover / stego / secret / recovered 四宫格 |
| `recovery_batch*.png` | 秘密图 vs 恢复图 |
| `depth_recovery_batch*.png` | 按 RQ 深度 1～4 的层级恢复 |
| `summary.json` | 指标与鲁棒性汇总 |

---

## 三、消融实验

消融需要 **多组 checkpoint**（完整模型 + 去掉某一损失的变体），再统一评估并画对比图。

### 步骤概览

1. **训练完整模型**（若还没有）
2. **训练消融变体**（no_align、no_robust 等）
3. **把各变体的 `final.pt` 放到约定目录**
4. **运行消融脚本**：对上述 checkpoint 做评估并生成对比图与表格

### 3.1 训练各变体并整理 checkpoint

**完整模型（full）：**

```bash
bash run_train.sh configs/default.yaml configs/train.yaml
mkdir -p outputs/ablation/full
cp outputs/final.pt outputs/ablation/full/
```

**去掉 rSMI 对齐（no_align）：**

```bash
bash run_train.sh configs/default.yaml configs/ablation_no_align.yaml
mkdir -p outputs/ablation/no_align
cp outputs/final.pt outputs/ablation/no_align/
```

**去掉干扰流形引导（no_robust）：**

```bash
bash run_train.sh configs/default.yaml configs/ablation_no_robust.yaml
mkdir -p outputs/ablation/no_robust
cp outputs/final.pt outputs/ablation/no_robust/
```

这样会得到：

- `outputs/ablation/full/final.pt`
- `outputs/ablation/no_align/final.pt`
- `outputs/ablation/no_robust/final.pt`

### 3.2 运行消融评估与对比图

```bash
# 对 outputs/ablation 下 full、no_align、no_robust 做评估，并画对比图
bash scripts/run_ablation.sh outputs/ablation 15
```

参数含义：`outputs/ablation` 为结果与 checkpoint 所在根目录，`15` 为每个变体评估的 batch 数。

### 3.3 一键“训练+评估”消融（可选）

若希望脚本**按顺序训练各变体再评估**（会多次覆盖 `outputs/`，耗时较长）：

```bash
python scripts/run_ablation.py --mode full --output_base outputs/ablation \
  --config configs/default.yaml --variants full no_align no_robust --num_batches 15
```

会依次：训练 full → 复制到 `outputs/ablation/full/`；训练 no_align → 复制到 `outputs/ablation/no_align/`；训练 no_robust → 复制到 `outputs/ablation/no_robust/`；再对三个变体分别跑测试并生成对比。

### 消融输出（在 `outputs/ablation/`）

| 文件 | 说明 |
|------|------|
| `ablation_comparison.png` | 多方法 × 多指标（BA、PSNR、SSIM）柱状图 |
| `ablation_results.json` | 各变体数值结果表 |
| `full/`、`no_align/`、`no_robust/` | 各变体单独评估的 metrics、对比图等 |

---

## 命令速查

| 目的 | 命令 |
|------|------|
| 训练（多卡自动） | `bash run_train.sh` |
| 测试（多卡自动） | `bash run_test.sh` |
| 消融（先准备好各变体 final.pt） | `bash scripts/run_ablation.sh outputs/ablation 15` |

若你希望把「训练 / 测试 / 消融」写进一个总脚本（例如 `run_all.sh`）里按顺序执行，也可以再说，我可以按你当前目录结构给一版脚本内容。
