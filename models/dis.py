"""
类 DiS 架构 (Diffusion with State Space)
Patch 嵌入 + 时间/条件嵌入 + 三流 Mamba 块序列 + 线性输出，无 U-Net 多尺度。
forward(x, t, f_sec, c_txt) -> (v, all_struc, all_tex)
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .mamba_blocks import SelectiveSSM
from .tri_stream_mamba import TriStreamBlock, timestep_embed, SecretModulation


class PatchEmbed(nn.Module):
    """将图像切为 patch 并线性嵌入。(B,3,H,W) -> (B, N, D), N=(H/p)*(W/p)."""

    def __init__(self, img_size: int = 256, patch_size: int = 4, in_ch: int = 3, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, D, H/p, W/p) -> (B, N, D)
        x = self.proj(x)
        return rearrange(x, "b c h w -> b (h w) c")


class FinalLayer(nn.Module):
    """将序列投影回 patch 像素并 unpatch。(B, N, D) -> (B, 3, H, W)."""

    def __init__(self, embed_dim: int, patch_size: int, out_ch: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * out_ch)
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # x: (B, N, D)
        x = self.linear(x)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=self.patch_size, p2=self.patch_size, c=self.out_ch)
        return x


class TriStreamDiS(nn.Module):
    """
    类 DiS 主干：PatchEmbed -> [t_embed + c_embed 加在首 token 或广播] -> N x TriStreamBlock -> FinalLayer.
    输出 v (B,3,H,W) 与 all_struc, all_tex 用于 rSMI。
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_channels: int = 3,
        model_channels: int = 256,
        out_channels: int = 3,
        num_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        text_embed_dim: int = 768,
        secret_embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.h = self.w = img_size // patch_size
        self.model_channels = model_channels

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, model_channels)
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels),
        )
        self.time_embed_input_dim = model_channels
        self.text_proj = nn.Linear(text_embed_dim, model_channels)
        self.blocks = nn.ModuleList([
            TriStreamBlock(model_channels, d_state, d_conv, expand, secret_embed_dim, text_embed_dim)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(model_channels)
        self.final_layer = FinalLayer(model_channels, patch_size, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        f_sec: torch.Tensor,
        c_txt: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        x: (B, 3, H, W)
        t: (B,)
        f_sec: (B, secret_embed_dim, H', W')
        c_txt: (B, text_embed_dim)
        Returns:
            v: (B, 3, H, W)
            all_struc: list of (B, C, h, w)
            all_tex: list of (B, C, h, w)
        """
        B = x.shape[0]
        # Patch embed
        x_seq = self.patch_embed(x)
        # Time embed: broadcast to all tokens
        t_emb = timestep_embed(t, self.time_embed_input_dim)
        t_emb = self.time_embed(t_emb)
        x_seq = x_seq + t_emb.unsqueeze(1)
        # Text: broadcast
        if c_txt is not None:
            c_emb = self.text_proj(c_txt)
            x_seq = x_seq + c_emb.unsqueeze(1)
        # f_sec: 插值到 patch 网格 (h, w) 以便与 block 内空间一致
        if f_sec.shape[2] != self.h or f_sec.shape[3] != self.w:
            f_sec = F.interpolate(f_sec, size=(self.h, self.w), mode="bilinear", align_corners=False)

        all_struc, all_tex = [], []
        for blk in self.blocks:
            # TriStreamBlock 接受 (B,L,C) 或 (B,C,H,W)；这里传入序列，内部会做 2D 变换做 mod
            h_struc_2d, h_tex_2d = blk(x_seq, f_sec, c_txt)
            all_struc.append(h_struc_2d)
            all_tex.append(h_tex_2d)
            x_seq = rearrange(h_tex_2d, "b c h w -> b (h w) c")
        x_seq = self.norm_out(x_seq)
        v = self.final_layer(x_seq, self.h, self.w)
        return v, all_struc, all_tex
