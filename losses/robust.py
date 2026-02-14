"""
鲁棒解码损失 L_robust = E_Π Σ_d w_d * CE(D_φ(Π(x̂_0))_d, S_d)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RobustDecodingLoss(nn.Module):
    def __init__(self, depth_weights: List[float] = None):
        super().__init__()
        self.depth_weights = depth_weights or [1.0, 0.8, 0.6, 0.4]

    def forward(
        self,
        logits_list: List[torch.Tensor],
        indices_list: List[torch.Tensor],
        depth_weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        logits_list: 解码器输出，每层 (B, H', W', K)
        indices_list: 真实 RQ-VAE 索引，每层 (B, H', W') long
        """
        weights = depth_weights or self.depth_weights
        loss = 0.0
        for d, (logits, indices) in enumerate(zip(logits_list, indices_list)):
            w = weights[d] if d < len(weights) else 1.0
            logits_flat = rearrange(logits, "b h w k -> (b h w) k")
            indices_flat = rearrange(indices, "b h w -> (b h w)")
            loss = loss + w * F.cross_entropy(logits_flat, indices_flat, ignore_index=-1)
        return loss / max(len(logits_list), 1)
