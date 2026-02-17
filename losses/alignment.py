"""
正交/去相关损失：确保结构流与纹理流互不干扰（正交），替代已废弃的 rSMI 对齐。
- OrthogonalLoss: 最小化结构特征与纹理特征在通道维度的内积，使二者正交。
- FeatureDecorrelationLoss: 最小化交叉协方差，使通道间互不相关。
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class OrthogonalLoss(nn.Module):
    """
    正交损失：使 h_struc 与 h_tex 在通道维度上正交（内积为零），
    确保纹理信息与结构信息互不干扰，而非强制相似。
    """
    def __init__(self, use_last_only: bool = False, lambda_orth: float = 0.1):
        super().__init__()
        self.use_last_only = use_last_only
        self.lambda_orth = lambda_orth

    def forward(
        self,
        h_struc_list: List[torch.Tensor],
        h_tex_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """h_struc_list, h_tex_list: 每项 (B, C, H, W)。"""
        if self.use_last_only:
            h_struc_list = [h_struc_list[-1]]
            h_tex_list = [h_tex_list[-1]]
        total = 0.0
        n = 0
        for h_struc, h_tex in zip(h_struc_list, h_tex_list):
            # (B, C, H, W) -> 沿 C 做内积得 (B, H, W)，再平方取平均
            inner = (h_struc * h_tex).sum(dim=1)
            total = total + (inner ** 2).mean()
            n += 1
        return self.lambda_orth * (total / max(n, 1))


class FeatureDecorrelationLoss(nn.Module):
    """
    计算两个特征矩阵之间的交叉协方差惩罚。
    目标是促使 h_struc 和 h_tex 的特征通道互不相关 (协方差矩阵的非对角元素趋于0)。
    """
    def __init__(self, use_last_only: bool = False, lambda_decorr: float = 0.1):
        super().__init__()
        self.use_last_only = use_last_only
        self.lambda_decorr = lambda_decorr

    def forward(
        self,
        h_struc_list: List[torch.Tensor],
        h_tex_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        h_struc_list, h_tex_list: (B, C, H, W) 或 (B, C, L) 维度的特征列表
        """
        if self.use_last_only:
            h_struc_list = [h_struc_list[-1]]
            h_tex_list = [h_tex_list[-1]]
            
        total_loss = 0.0
        n = 0
        
        for h_struc, h_tex in zip(h_struc_list, h_tex_list):
            B, C, H, W = h_struc.shape
            
            # 将空间维度合并，转为 (B, C, H*W)
            h_s = rearrange(h_struc, "b c h w -> b c (h w)")
            h_t = rearrange(h_tex, "b c h w -> b c (h w)")
            
            # 1. 实例级均值中心化 (Instance-level Mean Centering)
            # 在空间维度上减去均值
            h_s_centered = h_s - h_s.mean(dim=2, keepdim=True)
            h_t_centered = h_t - h_t.mean(dim=2, keepdim=True)
            
            # 2. L2 归一化 (防止由于特征绝对数值过大导致的梯度爆炸)
            h_s_norm = F.normalize(h_s_centered, p=2, dim=2)
            h_t_norm = F.normalize(h_t_centered, p=2, dim=2)
            
            # 3. 计算交叉协方差矩阵 (B, C, C)
            # 理想情况下，我们希望不同的特征通道之间不要包含对方的信息
            cross_cov = torch.bmm(h_s_norm, h_t_norm.transpose(1, 2))
            
            # 4. 惩罚交叉协方差矩阵的 Frobenius 范数 (即所有元素的平方和)
            # 注意：因为它们是不同的特征，对角线元素也不应该高度相关，所以惩罚整个矩阵
            cov_loss = (cross_cov ** 2).mean()
            
            total_loss += cov_loss
            n += 1
            
        return self.lambda_decorr * (total_loss / max(n, 1))