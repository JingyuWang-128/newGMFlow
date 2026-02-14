"""
hDCE: Hard Decoupled Contrastive Entropy
L_hDCE = -log( exp(q·k+/τ) / ( exp(q·k+/τ) + Σ_{n∈N_hard} exp(q·k_n-/τ) ) )
硬负样本：在 RQ-VAE 码本空间中与正确 Token 欧氏距离最近的其他码字。
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_hard_negatives(
    codebook: torch.Tensor,
    positive_idx: torch.Tensor,
    num_hard: int = 16,
) -> torch.Tensor:
    """
    codebook: (K, C)
    positive_idx: (B, L) 正确索引
    Returns: (B, L, num_hard) 每个位置的硬负样本索引
    """
    K, C = codebook.shape
    B, L = positive_idx.shape
    device = codebook.device
    pos_emb = codebook[positive_idx.clamp(0, K - 1)]
    dist = torch.cdist(pos_emb.view(B * L, C), codebook)
    _, topk = torch.topk(dist, min(num_hard + 1, K), dim=1, largest=False)
    hard = topk[:, 1 : num_hard + 1]
    return hard.view(B, L, -1)


class hDCELoss(nn.Module):
    """
    语义辅助硬负样本对比损失。
    q: 解码器特征 (B, L, C), k+: 正确码本向量, k_n-: 硬负样本码本向量。
    """

    def __init__(self, temperature: float = 0.07, num_hard: int = 16):
        super().__init__()
        self.temperature = temperature
        self.num_hard = num_hard

    def forward(
        self,
        decoder_feat: torch.Tensor,
        codebook: torch.Tensor,
        positive_indices: torch.Tensor,
        depth: int = 0,
    ) -> torch.Tensor:
        """
        decoder_feat: (B, L, C) 解码器某层特征（需与 codebook 同维或投影后同维）
        codebook: (K, C) 该层 RQ-VAE 码本
        positive_indices: (B, L) 正确 Token 索引
        """
        B, L, C = decoder_feat.shape
        K = codebook.shape[0]
        if C != codebook.shape[1]:
            return torch.tensor(0.0, device=decoder_feat.device)
        q = F.normalize(decoder_feat, dim=-1)
        k_pos = F.normalize(codebook[positive_indices.clamp(0, K - 1)], dim=-1)
        pos_logits = (q * k_pos).sum(-1) / self.temperature
        hard_idx = get_hard_negatives(codebook, positive_indices, self.num_hard)
        neg_embs = codebook[hard_idx.clamp(0, K - 1)]
        neg_logits = torch.einsum("blc,blnc->bln", q, F.normalize(neg_embs, dim=-1)) / self.temperature
        logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
        labels = torch.zeros(B, L, dtype=torch.long, device=q.device)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
