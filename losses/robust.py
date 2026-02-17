"""
鲁棒解码损失 L_robust = E_Π Σ_d w_d * CE(D_φ(Π(x̂_0))_d, S_d)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ContinuousRobustDecodingLoss(nn.Module):
    """
    鲁棒连续回归损失：替代极其脆弱的离散索引交叉熵损失。
    结合 L1 损失与 FFT 频域损失，对提取出来的连续特征进行多尺度重建。
    """
    def __init__(self, depth_weights: list = None, use_frequency: bool = True):
        super().__init__()
        self.depth_weights = depth_weights or [1.0, 0.8, 0.6, 0.4]
        self.use_frequency = use_frequency
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        preds_list: list,       # 解码器输出的连续特征 (B, C, H', W')
        targets_list: list,     # RQ-VAE 编码的真实连续特征 (B, C, H', W')
        depth_weights: list = None,
    ) -> torch.Tensor:
        
        weights = depth_weights or self.depth_weights
        total_loss = 0.0
        
        for d, (pred, target) in enumerate(zip(preds_list, targets_list)):
            w = weights[d] if d < len(weights) else 1.0
            # 空间尺寸不一致时（如 decoder 16x16 vs RQ-VAE 32x32）将 target 对齐到 pred
            if pred.shape[-2:] != target.shape[-2:]:
                target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            # 空间域连续回归：平滑 L1 缓解极端离群值的梯度爆炸
            layer_loss = F.smooth_l1_loss(pred, target, beta=0.1)
            
            # 频域回归：隐写任务极其依赖高频特征的完美对齐
            if self.use_frequency:
                pred_fft = torch.fft.fftn(pred, dim=(-2, -1))
                target_fft = torch.fft.fftn(target, dim=(-2, -1))
                # 惩罚振幅谱的差异
                fft_loss = self.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
                layer_loss = layer_loss + 0.1 * fft_loss
                
            total_loss = total_loss + w * layer_loss
            
        return total_loss / max(len(preds_list), 1)


# class RobustDecodingLoss(nn.Module):
#     def __init__(self, depth_weights: List[float] = None):
#         super().__init__()
#         self.depth_weights = depth_weights or [1.0, 0.8, 0.6, 0.4]

#     def forward(
#         self,
#         logits_list: List[torch.Tensor],
#         indices_list: List[torch.Tensor],
#         depth_weights: Optional[List[float]] = None,
#     ) -> torch.Tensor:
#         """
#         logits_list: 解码器输出，每层 (B, H', W', K)
#         indices_list: 真实 RQ-VAE 索引，每层 (B, H', W') long
#         """
#         weights = depth_weights or self.depth_weights
#         loss = 0.0
#         for d, (logits, indices) in enumerate(zip(logits_list, indices_list)):
#             w = weights[d] if d < len(weights) else 1.0
#             logits_flat = rearrange(logits, "b h w k -> (b h w) k")
#             indices_flat = rearrange(indices, "b h w -> (b h w)")
#             loss = loss + w * F.cross_entropy(logits_flat, indices_flat, ignore_index=-1)
#         return loss / max(len(logits_list), 1)
