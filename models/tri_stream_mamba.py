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
from .mamba_blocks import SelectiveSSM


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


class TriStreamBlock(nn.Module):
    """
    单层三流块：
    h_sem = SSM_sem(x, c_txt) [可选冻结]
    h_struc = SSM_struc(x + h_sem)
    h_tex = SSM_tex(h_struc) ⊕ M(f_sec)
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        secret_dim: int = 256,
        text_dim: int = 768,
        freeze_semantic: bool = False,
    ):
        super().__init__()
        self.freeze_semantic = freeze_semantic
        self.sem_norm = nn.LayerNorm(dim)
        self.sem_ssm = SelectiveSSM(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.sem_proj = nn.Linear(text_dim, dim)

        self.struc_norm = nn.LayerNorm(dim)
        self.struc_ssm = SelectiveSSM(dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.tex_norm = nn.LayerNorm(dim)
        self.tex_ssm = SelectiveSSM(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.secret_mod = SecretModulation(secret_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        f_sec: torch.Tensor,
        c_txt: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        x: (B, C, H, W) 或 (B, L, C)
        f_sec: (B, C_sec, H', W') 或 (B, L', C_sec)
        c_txt: (B, text_dim)
        Returns:
            h_struc, h_tex (用于 rSMI 对齐与输出)
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
        h_sem = self.sem_ssm(self.sem_norm(x_flat + c_emb.unsqueeze(1)))
        if self.freeze_semantic:
            h_sem = h_sem.detach()
        h_struc = self.struc_ssm(self.struc_norm(x_flat + h_sem))
        h_tex = self.tex_ssm(self.tex_norm(h_struc))
        h_tex_2d = rearrange(h_tex, "b (h w) c -> b c h w", h=H, w=W)
        h_tex_2d = self.secret_mod(f_sec, h_tex_2d)
        h_tex = rearrange(h_tex_2d, "b c h w -> b (h w) c")
        h_struc_2d = rearrange(h_struc, "b (h w) c -> b c h w", h=H, w=W)
        return h_struc_2d, h_tex_2d
