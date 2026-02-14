# GenMamba-Flow: Robust Generative Steganography

基于解耦 Mamba 流与干扰流形引导的鲁棒生成式隐写术。

## 创新点

1. **内生鲁棒性**：干扰流形引导的生成对抗传输，在 Rectified Flow 轨迹中引入对抗性梯度，生成即防御。
2. **高保真性**：语义-结构-纹理三流 Mamba 解耦 + rSMI 约束，隐写仅改局部纹理，保留语义与结构。
3. **高还原度**：RQ-VAE 残差离散化 + 语义辅助硬负样本对比解码，实现分级完美重建。

## 环境

```bash
pip install -r requirements.txt
```

可选（Linux + CUDA）：`pip install mamba-ssm` 以使用官方 Mamba 加速。

## 数据

将数据放入以下占位目录（或修改 `configs/default.yaml`）：

- 覆盖/条件图像：`data/placeholder/DIV2K`, `data/placeholder/COCO`
- 秘密图像：`data/placeholder/Paris_StreetView`, `data/placeholder/CelebA-HQ`

支持任意图像目录结构（递归收集 `.jpg/.png` 等）。

## 训练

```bash
bash run_train.sh [config] [override_config] [device]
# 示例
bash run_train.sh configs/default.yaml configs/train.yaml cuda
```

或直接：

```bash
python train.py --config configs/default.yaml --override configs/train.yaml --device cuda
```

训练会先预训练 RQ-VAE（若未提供 `outputs/rq_vae.pt`），再联合训练生成器与解码器。

## 测试与评估

```bash
bash run_test.sh [config] [checkpoint] [output_dir] [device]
python test.py --config configs/default.yaml --checkpoint outputs/final.pt --output_dir outputs/eval --num_batches 20
```

鲁棒性曲线（不同 JPEG 质量下 Bit Accuracy）：

```bash
python test.py --config configs/default.yaml --checkpoint outputs/final.pt --output_dir outputs/eval --robustness_only
```

## 目录结构

```
GenMamba-Flow/
├── configs/           # 配置
├── data/              # 数据与 DataLoader
│   └── placeholder/   # 数据集占位目录
├── models/            # RQ-VAE, Tri-Stream Mamba, Rectified Flow, Decoder, Interference
├── losses/            # rSMI, Robust Decoding, hDCE
├── utils/             # 指标、可视化、配置加载
├── train.py           # 训练入口
├── test.py            # 测试与评估
├── run_train.sh
├── run_test.sh
└── requirements.txt
```

## 引用

若使用本代码，请引用论文：GenMamba-Flow: Robust Generative Steganography via Decoupled Mamba Streams and Interference Manifold Guidance.
