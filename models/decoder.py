"""
鲁棒解码器 D_φ：从（可能受干扰的）隐写图预测 RQ-VAE 各层离散索引。
基于 Mamba，使用语义辅助对比检索（hDCE）训练。
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from .mamba_blocks import SelectiveSSM


class DecoderBackbone(nn.Module):
    """CNN + Mamba 骨干：将图像编码为序列特征。"""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 256, num_layers: int = 4, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.ssm_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                SelectiveSSM(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand),
            ) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.conv_in(x)
        _, _, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        for layer in self.ssm_layers:
            x = x + layer[1](layer[0](x))
        return x, h, w


class IndexHead(nn.Module):
    """为每一深度 d 预测 RQ-VAE 码本上的 logits，用于 CE 或对比检索。"""

    def __init__(self, hidden_dim: int, num_embeddings: int, num_depths: int):
        super().__init__()
        self.num_depths = num_depths
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_embeddings) for _ in range(num_depths)
        ])

    def forward(self, feat: torch.Tensor) -> List[torch.Tensor]:
        """feat: (B, L, C) -> list of (B, L, num_embeddings)."""
        return [self.heads[d](feat) for d in range(self.num_depths)]


class RobustDecoder(nn.Module):
    """
    解码器 D_φ：输入图像 x（可能已被干扰），输出各层索引的 logits。
    训练时用 CE 或 hDCE（硬负样本对比）损失。
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_rq_depths: int = 4,
        num_embeddings: int = 8192,
    ):
        super().__init__()
        self.backbone = DecoderBackbone(in_channels, hidden_dim, num_layers, d_state, d_conv, expand)
        self.head = IndexHead(hidden_dim, num_embeddings, num_rq_depths)
        self.num_rq_depths = num_rq_depths
        self.num_embeddings = num_embeddings

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: (B, 3, H, W) 隐写图或受干扰图
        Returns:
            logits_list: 每层 (B, H', W', K)，H'*W' 与 RQ-VAE 潜在网格一致（默认 H/16, W/16）
        """
        feat, h, w = self.backbone(x)
        # 对齐 RQ-VAE 潜在空间：通常 16x 下采样
        latent_h, latent_w = max(1, x.shape[2] // 16), max(1, x.shape[3] // 16)
        feat_2d = rearrange(feat, "b (h w) c -> b c h w", h=h, w=w)
        feat_2d = F.adaptive_avg_pool2d(feat_2d, (latent_h, latent_w))
        feat = rearrange(feat_2d, "b c h w -> b (h w) c")
        logits_list = self.head(feat)
        out = []
        for logits in logits_list:
            logits = rearrange(logits, "b (h w) k -> b h w k", h=latent_h, w=latent_w)
            out.append(logits)
        return out

    def predict_indices(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回每层 argmax 索引 (B, H', W')."""
        logits_list = self.forward(x)
        return [logits.argmax(dim=-1) for logits in logits_list]
