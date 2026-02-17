# GenMamba-Flow: Robust Generative Steganography

基于解耦 Mamba 流与干扰流形引导的鲁棒**无载体**生成式隐写术。生成器采用 **DiS 类架构**（Patch + **并行**三流 Mamba 序列），支持 **多卡训练/测试**，**单脚本** 完成训练与测试，并带 **完整可视化**（损失曲线、隐写对比、指标图、鲁棒性曲线、消融对比）。训练与测试均为**无载体**：仅用秘密图生成隐写图，无需独立载体图。

## 创新点

1. **内生鲁棒性**：干扰流形引导的生成与传输；**生成即防御**在 **ODE 采样器** 层面实现——默认使用 **Predictor-Corrector** 纠错采样（Euler + Langevin 式 Corrector），而非在直线插值轨迹上做手脚。
2. **高保真性**：语义-结构-纹理 **真并行** 三流 Mamba 解耦 + **正交损失**（Orthogonal Loss）约束，使结构流与纹理流互不干扰（正交），隐写仅改局部纹理，保留语义与结构。
3. **高还原度**：RQ-VAE 残差量化作为特征瓶颈 + **连续域解码损失**（L1/平滑 L1 + 可选 FFT 频域），避免纯离散索引分类带来的级联崩溃；可选语义辅助硬负样本对比（hDCE）进一步强化解码。

## 环境

```bash
pip install -r requirements.txt
```

可选（Linux + CUDA）：`pip install mamba-ssm` 以使用官方 Mamba 加速。

## 数据

将数据放入以下占位目录（或修改 `configs/default.yaml`）：

- 覆盖/条件图像：`data/placeholder/DIV2K`, `data/placeholder/COCO`
- 秘密图像：`data/placeholder/Paris_StreetView`, `data/placeholder/CelebA-HQ`

支持任意图像目录结构（递归收集 `.jpg/.png` 等）。本实现为**无载体隐写**：训练 Stage2 与测试均仅使用秘密图像，生成隐写图不依赖独立载体图。

## 生成器架构：DiS + 并行三流 Mamba

生成器为 **类 DiS**（Diffusion with State Space）：Patch 嵌入 + 时间/文本嵌入 + 一串 **并行三流 Mamba 块** + 线性输出为速度场，无 U-Net 多尺度。每个块内 **语义 / 结构 / 纹理** 三路 **并行** 独立计算（`h_sem = SSM_sem(x, c_txt)`、`h_struc = SSM_struc(x)`、`h_tex = SSM_tex(x) ⊕ M(f_sec)`），再拼接后经线性融合与残差得到下一层输入；结构流与纹理流用于 **正交损失**，确保二者表示互不干扰。配置见 `configs/default.yaml` 中 `generator`：`patch_size`、`num_layers`、`img_size` 等。

## 训练（单脚本、多卡 DDP）

**一条命令跑完全部阶段**（RQ-VAE 预训练 + 生成器与解码器联合训练），并自动使用 **当前机器所有 GPU**（DDP）：

```bash
bash run_train.sh [config] [override_config]
# 示例
bash run_train.sh configs/default.yaml configs/train.yaml
```

或直接（单卡）：

```bash
python train.py --config configs/default.yaml --override configs/train.yaml
```

多卡时脚本内部使用 `torchrun --nproc_per_node=<GPU数> train.py`，无需手动指定设备。

**训练阶段可视化**（自动写入 `outputs/vis/`）：

- **损失曲线**：`loss_curves.png` / `loss_curves_final.png`（loss_flow, loss_align 正交, loss_robust 连续解码, loss_dec）
- **损失历史**：`loss_history.json` 供后续重绘
- **隐写对比图**：每隔 `project.vis_every` 步保存 cover / stego / secret / recovered 四宫格 `stego_step*.png`

配置项：`project.vis_every`、`project.log_every`、`project.save_every`。

**解码与采样**：解码器同时输出离散索引 logits（推理/可选 hDCE）与 **连续特征**（各深度 `(B, C, H', W')`），训练时以 **ContinuousRobustDecodingLoss**（连续 L1 + 频域）为主，量化作为瓶颈。采样时 `sample()` 默认委托 **Predictor-Corrector**（`sample_pc`），可通过 `num_steps`、`corrector_steps`、`snr` 调节。

## 测试与评估（单脚本、多卡）

**一条命令跑完所有评估与可视化**（主指标 + 鲁棒性曲线 + 对比图）：

```bash
bash run_test.sh [config] [checkpoint] [output_dir] [num_batches]
# 示例
bash run_test.sh configs/default.yaml outputs/final.pt outputs/eval 20
```

脚本会使用 **所有可用 GPU**（DataParallel）加速推理。仅做鲁棒性曲线时：

```bash
python test.py --checkpoint outputs/final.pt --output_dir outputs/eval --robustness_only --multi_gpu
```

**测试输出**（均在 `output_dir` 下）：

- **指标**：`metrics.txt`、`metrics.json`（Bit Accuracy, Recovery PSNR, Recovery SSIM）
- **指标柱状图**：`metrics_bars.png`
- **鲁棒性曲线**：`robustness_jpeg_curve.png`、`robustness_jpeg.json`
- **隐写对比**：`compare_batch*.png`、`recovery_batch*.png`、`depth_recovery_batch*.png`（层级恢复）
- **汇总**：`summary.json`

## 消融实验与对比可视化

对多组变体（如 full / no_align / no_robust）分别评估，并生成 **对比柱状图** 与 **结果表**：

```bash
bash scripts/run_ablation.sh [output_dir] [num_batches]
# 示例
bash scripts/run_ablation.sh outputs/ablation 15
```

要求各变体 checkpoint 已放在 `output_dir/<变体名>/final.pt`，例如：

- `outputs/ablation/full/final.pt`
- `outputs/ablation/no_align/final.pt`（可用 `configs/ablation_no_align.yaml` 训练得到）
- `outputs/ablation/no_robust/final.pt`（可用 `configs/ablation_no_robust.yaml` 训练得到）

生成：

- `outputs/ablation/ablation_comparison.png`：多方法 × 多指标柱状图
- `outputs/ablation/ablation_results.json`：数值结果表

也可直接运行：

```bash
python scripts/run_ablation.py --mode eval --output_base outputs/ablation --num_batches 15 \
  --variants full no_align no_robust --checkpoint_dir outputs/ablation
```

## 目录结构

```
GenMamba-Flow/
├── configs/             # 默认/训练/消融配置
├── data/                # 数据与 DataLoader（含 DDP 采样）
├── models/              # RQ-VAE, DiS, Rectified Flow, Decoder, Interference
├── losses/              # Orthogonal/FeatureDecorrelation, ContinuousRobustDecoding, hDCE
├── utils/               # 指标、可视化（vis_plots）、配置加载
├── scripts/             # 可视化脚本、消融入口
├── train.py             # 训练入口（DDP + 阶段内可视化）
├── test.py              # 测试入口（多卡 + 全量可视化）
├── run_train.sh         # 一键训练（自动多卡）
├── run_test.sh          # 一键测试（自动多卡）
└── requirements.txt
```

## 引用

若使用本代码，请引用论文：GenMamba-Flow: Robust Generative Steganography via Decoupled Mamba Streams and Interference Manifold Guidance.
