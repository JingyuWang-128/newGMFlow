"""
rSMI 结构-纹理对齐损失
L_align = -log( exp(sim(h_struc, h_tex)/τ) / Σ_j exp(sim(h_struc, h_tex^{(j)})/τ) ) + λ_reg ||Cov(h_tex) - Cov(h_struc)||_F^2
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: (B, C) or (B, C, H, W) -> (B,) or (B, H, W)."""
    if a.dim() == 4:
        a = rearrange(a, "b c h w -> b (c h w)")
        b = rearrange(b, "b c h w -> b (c h w)")
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


class rSMIAlignmentLoss(nn.Module):
    """
    相对平方互信息对齐：InfoNCE 形式 + 协方差正则。
    对多尺度 h_struc, h_tex 列表取平均或最后一层。
    """

    def __init__(self, temperature: float = 0.07, lambda_reg: float = 0.01, use_last_only: bool = False):
        super().__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg
        self.use_last_only = use_last_only

    def forward(
        self,
        h_struc_list: List[torch.Tensor],
        h_tex_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        h_struc_list, h_tex_list: 每层 (B, C, H, W)
        取最后一层或所有层平均。
        """
        if self.use_last_only:
            h_struc_list = [h_struc_list[-1]]
            h_tex_list = [h_tex_list[-1]]
        total_loss = 0.0
        n = 0
        for h_struc, h_tex in zip(h_struc_list, h_tex_list):
            B, C, H, W = h_struc.shape
            h_s = rearrange(h_struc, "b c h w -> b (c h w)")
            h_t = rearrange(h_tex, "b c h w -> b (c h w)")
            h_s = F.normalize(h_s, dim=-1)
            h_t = F.normalize(h_t, dim=-1)
            logits = torch.mm(h_s, h_t.t()) / self.temperature
            labels = torch.arange(B, device=h_s.device)
            total_loss = total_loss + F.cross_entropy(logits, labels)
            if self.lambda_reg > 0 and B > 1:
                cov_s = (h_s - h_s.mean(0)).t().mm(h_s - h_s.mean(0)) / (B - 1)
                cov_t = (h_t - h_t.mean(0)).t().mm(h_t - h_t.mean(0)) / (B - 1)
                total_loss = total_loss + self.lambda_reg * F.mse_loss(cov_t, cov_s)
            n += 1
        return total_loss / max(n, 1)
