"""
鲁棒解码器 D_φ：从（可能受干扰的）隐写图预测 RQ-VAE 各层离散索引。
基于 Mamba，使用语义辅助对比检索（hDCE）训练。
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import math

from .mamba_blocks import MambaBlock


class DecoderBackbone(nn.Module):
    """CNN + Mamba 骨干：将图像编码为序列特征。"""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 256, num_layers: int = 4, d_state: int = 16, d_conv: int = 4, expand: int = 2, use_mamba_ssm: bool = True):
        super().__init__()
        # 三次 stride=2：256→128→64→32，序列长 32*32=1024，避免 64*64=4096 时 SSM 反向 48GB OOM
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.ssm_layers = nn.ModuleList([
            MambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand, use_mamba_ssm=use_mamba_ssm)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.conv_in(x)
        _, _, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        for layer in self.ssm_layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x, h, w


class IndexHead(nn.Module):
    """为每一深度 d 预测 RQ-VAE 码本上的 logits，用于推理/对比检索（可选）。"""

    def __init__(self, hidden_dim: int, num_embeddings: int, num_depths: int):
        super().__init__()
        self.num_depths = num_depths
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_embeddings) for _ in range(num_depths)
        ])

    def forward(self, feat: torch.Tensor) -> List[torch.Tensor]:
        """feat: (B, L, C) -> list of (B, L, num_embeddings)."""
        return [self.heads[d](feat) for d in range(self.num_depths)]


class ContinuousHead(nn.Module):
    """为每一深度输出连续特征 (B, latent_channels, h, w)，用于 L1/L2/感知等连续解码损失。"""

    def __init__(self, hidden_dim: int, latent_channels: int, num_depths: int):
        super().__init__()
        self.num_depths = num_depths
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, latent_channels) for _ in range(num_depths)
        ])

    def forward(self, feat: torch.Tensor, h: int, w: int) -> List[torch.Tensor]:
        """feat: (B, L, C) -> list of (B, latent_channels, h, w)."""
        out = []
        for d in range(self.num_depths):
            z = self.heads[d](feat)
            z = rearrange(z, "b (h w) c -> b c h w", h=h, w=w)
            out.append(z)
        return out


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
        latent_channels: int = 256,
        use_mamba_ssm: bool = True,
    ):
        super().__init__()
        self.backbone = DecoderBackbone(in_channels, hidden_dim, num_layers, d_state, d_conv, expand, use_mamba_ssm=use_mamba_ssm)
        self.head = IndexHead(hidden_dim, num_embeddings, num_rq_depths)
        self.continuous_head = ContinuousHead(hidden_dim, latent_channels, num_rq_depths)
        self.num_rq_depths = num_rq_depths
        self.num_embeddings = num_embeddings
        self.latent_channels = latent_channels
        self.feat_proj = nn.Linear(hidden_dim, latent_channels)

    def forward(
        self, x: torch.Tensor, return_feat: bool = False
    ) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]]:
        """
        x: (B, 3, H, W) 隐写图或受干扰图
        return_feat: 为 True 时额外返回 (B, L, latent_channels) 特征，用于 hDCE。
        Returns:
            (logits_list, continuous_list) 或 (logits_list, continuous_list, feat_256)
            连续列表用于 L1/L2/感知等解码损失，避免纯离散索引的级联崩溃。
        """
        feat, h, w = self.backbone(x)
        if feat.dim() == 4:
            # SelectiveSSM 4D 兼容时可能返回 (B,L,L,C)，压成 (B, L, C)
            # 必须用索引(比如0)而不是切片，才能真正消除多余的维度
            feat = feat[:, 0, : h * w, :].contiguous()
            
        # 强制 3D，防止 checkpoint/DDP 或缓存导致仍为 4D
        if feat.dim() != 3:
            # 如果此时还是非3D，尝试 squeeze
            feat = feat.squeeze()
            if feat.dim() == 4:
                feat = feat[:, 0, :, :].contiguous()
        latent_h, latent_w = max(1, x.shape[2] // 16), max(1, x.shape[3] // 16)
        feat_2d = rearrange(feat, "b (h w) c -> b c h w", h=h, w=w)
        feat_2d = F.adaptive_avg_pool2d(feat_2d, (latent_h, latent_w))
        feat = rearrange(feat_2d, "b c h w -> b (h w) c")
        logits_list = self.head(feat)
        out_logits = []
        for logits in logits_list:
            logits = rearrange(logits, "b (h w) k -> b h w k", h=latent_h, w=latent_w)
            out_logits.append(logits)
        continuous_list = self.continuous_head(feat, latent_h, latent_w)
        if return_feat:
            feat_256 = self.feat_proj(feat)
            return out_logits, continuous_list, feat_256
        return out_logits, continuous_list

    def predict_indices(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回每层 argmax 索引 (B, H', W')."""
        logits_list, _ = self.forward(x)
        return [logits.argmax(dim=-1) for logits in logits_list]

