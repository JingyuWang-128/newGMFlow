"""
三流解耦 Mamba 块 (Tri-Stream Block)
h_sem = SSM_sem(x; c_txt), h_struc = SSM_struc(x + h_sem), h_tex = SSM_tex(h_struc) ⊕ M(f_sec)
供 DiS 主干使用，用于 Rectified Flow 的速度场预测 v_θ。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import math
from .mamba_blocks import SelectiveSSM, HAS_MAMBA_SSM

if HAS_MAMBA_SSM:
    from mamba_ssm import Mamba as MambaSSM
else:
    MambaSSM = None


def _build_ssm(dim: int, d_state: int, d_conv: int, expand: int, use_mamba_ssm: bool) -> nn.Module:
    """DiS 的 S 可选用 Selective SSM（PyTorch）或官方 Mamba（mamba_ssm）。"""
    if use_mamba_ssm and MambaSSM is not None:
        return MambaSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
    return SelectiveSSM(dim, d_state=d_state, d_conv=d_conv, expand=expand)


def timestep_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class SecretModulation(nn.Module):
    """M(f_sec): 将秘密特征通过 Cross-Scan 与线性层注入纹理流。"""

    def __init__(self, secret_dim: int, tex_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(secret_dim, tex_dim),
            nn.SiLU(),
            nn.Linear(tex_dim, tex_dim),
        )

    def forward(self, f_sec: torch.Tensor, h_tex: torch.Tensor) -> torch.Tensor:
        """
        f_sec: (B, C_sec, H', W') 或 (B, H'*W', C_sec)
        h_tex: (B, C_tex, H, W)
        """
        if f_sec.dim() == 4:
            f_sec = rearrange(f_sec, "b c h w -> b (h w) c")
        B, L, C_sec = f_sec.shape
        f = self.proj(f_sec)
        _, C_tex, H, W = h_tex.shape
        if L != H * W:
            f = F.interpolate(
                rearrange(f, "b (h w) c -> b c h w", h=int(math.sqrt(L)), w=int(math.sqrt(L))),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            f = rearrange(f, "b c h w -> b (h w) c")
        return h_tex + rearrange(f, "b (h w) c -> b c h w", h=H, w=W)

class ParallelTriStreamBlock(nn.Module):
    """
    重构后的真并行三流 Mamba 块：
    h_sem = SSM_sem(x, c_txt)
    h_struc = SSM_struc(x)
    h_tex = SSM_tex(x) ⊕ M(f_sec)
    x_out = Linear(Concat(h_sem, h_struc, h_tex)) + x (残差连接)
    """
    def __init__(
        self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
        secret_dim: int = 256, text_dim: int = 768, freeze_semantic: bool = False,
        use_mamba_ssm: bool = True
    ):
        super().__init__()
        self.freeze_semantic = freeze_semantic
        
        # 语义分支
        self.sem_norm = nn.LayerNorm(dim)
        self.sem_ssm = _build_ssm(dim, d_state, d_conv, expand, use_mamba_ssm)
        self.sem_proj = nn.Linear(text_dim, dim)

        # 结构分支 (完全独立)
        self.struc_norm = nn.LayerNorm(dim)
        self.struc_ssm = _build_ssm(dim, d_state, d_conv, expand, use_mamba_ssm)

        # 纹理分支 (完全独立)
        self.tex_norm = nn.LayerNorm(dim)
        self.tex_ssm = _build_ssm(dim, d_state, d_conv, expand, use_mamba_ssm)
        self.secret_mod = SecretModulation(secret_dim, dim)
        
        # 特征融合层
        self.fusion = nn.Linear(dim * 3, dim)
        self.fusion_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, f_sec: torch.Tensor, c_txt: torch.Tensor = None):
        """
        返回: x_out (给下一层的融合特征), h_struc_2d, h_tex_2d (用于正交损失)
        """
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = rearrange(x, "b c h w -> b (h w) c")
        else:
            x_flat = x
            B, L, C = x_flat.shape
            H = W = int(math.sqrt(L))

        if c_txt is None:
            c_txt = torch.zeros(B, self.sem_proj.in_features, device=x.device, dtype=x_flat.dtype)
            
        c_emb = self.sem_proj(c_txt)
        
        # 1. 语义流
        h_sem = self.sem_ssm(self.sem_norm(x_flat + c_emb.unsqueeze(1)))
        if self.freeze_semantic:
            h_sem = h_sem.detach()
            
        # 2. 结构流 (并行)
        h_struc = self.struc_ssm(self.struc_norm(x_flat))
        
        # 3. 纹理流 (并行) + 秘密信息注入
        h_tex_pre = self.tex_ssm(self.tex_norm(x_flat))
        h_tex_2d = rearrange(h_tex_pre, "b (h w) c -> b c h w", h=H, w=W)
        h_tex_2d = self.secret_mod(f_sec, h_tex_2d)
        h_tex = rearrange(h_tex_2d, "b c h w -> b (h w) c")
        
        # 4. 融合与残差连接
        fused_features = torch.cat([h_sem, h_struc, h_tex], dim=-1)
        x_out = x_flat + self.fusion(self.fusion_norm(fused_features))
        
        h_struc_2d = rearrange(h_struc, "b (h w) c -> b c h w", h=H, w=W)
        
        if x.dim() == 4:
            x_out = rearrange(x_out, "b (h w) c -> b c h w", h=H, w=W)
            
        # 注意：DiS 主干网络需要更新接收逻辑为 x = block(x, f_sec)[0]
        return x_out, h_struc_2d, h_tex_2d


# 对外兼容：串行 TriStreamBlock 已废弃，统一使用并行块
TriStreamBlock = ParallelTriStreamBlock


# class TriStreamBlock(nn.Module):
#     """
#     单层三流块：
#     h_sem = SSM_sem(x, c_txt) [可选冻结]
#     h_struc = SSM_struc(x + h_sem)
#     h_tex = SSM_tex(h_struc) ⊕ M(f_sec)
#     """

#     def __init__(
#         self,
#         dim: int,
#         d_state: int = 16,
#         d_conv: int = 4,
#         expand: int = 2,
#         secret_dim: int = 256,
#         text_dim: int = 768,
#         freeze_semantic: bool = False,
#         use_mamba_ssm: bool = True,
#     ):
#         super().__init__()
#         self.freeze_semantic = freeze_semantic
#         self.sem_norm = nn.LayerNorm(dim)
#         self.sem_ssm = _build_ssm(dim, d_state, d_conv, expand, use_mamba_ssm)
#         self.sem_proj = nn.Linear(text_dim, dim)

#         self.struc_norm = nn.LayerNorm(dim)
#         self.struc_ssm = _build_ssm(dim, d_state, d_conv, expand, use_mamba_ssm)

#         self.tex_norm = nn.LayerNorm(dim)
#         self.tex_ssm = _build_ssm(dim, d_state, d_conv, expand, use_mamba_ssm)
#         self.secret_mod = SecretModulation(secret_dim, dim)

#     def forward(
#         self,
#         x: torch.Tensor,
#         f_sec: torch.Tensor,
#         c_txt: Optional[torch.Tensor] = None,
#     ) -> tuple:
#         """
#         x: (B, C, H, W) 或 (B, L, C)
#         f_sec: (B, C_sec, H', W') 或 (B, L', C_sec)
#         c_txt: (B, text_dim)
#         Returns:
#             h_struc, h_tex (用于 rSMI 对齐与输出)
#         """
#         if x.dim() == 4:
#             B, C, H, W = x.shape
#             x_flat = rearrange(x, "b c h w -> b (h w) c")
#         else:
#             x_flat = x
#             B, L, C = x_flat.shape
#             H = W = int(math.sqrt(L))

#         if c_txt is None:
#             c_txt = torch.zeros(B, self.sem_proj.in_features, device=x.device, dtype=x_flat.dtype)
#         c_emb = self.sem_proj(c_txt)
#         h_sem = self.sem_ssm(self.sem_norm(x_flat + c_emb.unsqueeze(1)))
#         if self.freeze_semantic:
#             h_sem = h_sem.detach()
#         h_struc = self.struc_ssm(self.struc_norm(x_flat + h_sem))
#         h_tex = self.tex_ssm(self.tex_norm(h_struc))
#         h_tex_2d = rearrange(h_tex, "b (h w) c -> b c h w", h=H, w=W)
#         h_tex_2d = self.secret_mod(f_sec, h_tex_2d)
#         h_tex = rearrange(h_tex_2d, "b c h w -> b (h w) c")
#         h_struc_2d = rearrange(h_struc, "b (h w) c -> b c h w", h=H, w=W)
#         return h_struc_2d, h_tex_2d
