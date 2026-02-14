"""
三流解耦 Mamba U-Net (Tri-Stream Mamba)
h_sem = SSM_sem(x_t; c_txt), h_struc = SSM_struc(x_t + h_sem), h_tex = SSM_tex(h_struc) ⊕ M(f_sec)
用于 Rectified Flow 的速度场预测网络 v_θ。
"""

from typing import Optional, List

import torch
import torch.nn as nn
from einops import rearrange

import math
from .mamba_blocks import Mamba2DBlock, CrossScan, SelectiveSSM
from .rq_vae import RQVAE


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
            c_txt = torch.zeros(B, 1, device=x.device)
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


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int, d_state: int, d_conv: int, expand: int, secret_dim: int, text_dim: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([
            TriStreamBlock(out_ch, d_state, d_conv, expand, secret_dim, text_dim) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, f_sec: torch.Tensor, c_txt: Optional[torch.Tensor]) -> tuple:
        x = self.down(x)
        struc_list, tex_list = [], []
        for blk in self.blocks:
            h_struc, h_tex = blk(x, f_sec, c_txt)
            x = h_tex
            struc_list.append(h_struc)
            tex_list.append(h_tex)
        return x, struc_list, tex_list


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int, d_state: int, d_conv: int, expand: int, secret_dim: int, text_dim: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.blocks = nn.ModuleList([
            TriStreamBlock(out_ch, d_state, d_conv, expand, secret_dim, text_dim) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, f_sec: torch.Tensor, c_txt: Optional[torch.Tensor]) -> tuple:
        x = self.up(x)
        struc_list, tex_list = [], []
        for blk in self.blocks:
            h_struc, h_tex = blk(x, f_sec, c_txt)
            x = h_tex
            struc_list.append(h_struc)
            tex_list.append(h_tex)
        return x, struc_list, tex_list


class TriStreamMambaUNet(nn.Module):
    """
    U-Net 骨架 + 多尺度三流 Mamba，输出速度 v 与各层 h_struc, h_tex 用于 rSMI 损失。
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 256,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        channel_mult: List[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        text_embed_dim: int = 768,
        secret_embed_dim: int = 256,
    ):
        super().__init__()
        channel_mult = channel_mult or [1, 2, 3, 4]
        self.channels = [model_channels * m for m in channel_mult]
        self.num_res_blocks = num_res_blocks
        self.time_embed_dim = model_channels * 4
        self.time_embed_input_dim = model_channels
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_input_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            self.down_blocks.append(
                DownBlock(ch, out_ch, num_res_blocks, d_state, d_conv, expand, secret_embed_dim, text_embed_dim)
            )
            ch = out_ch
        self.mid_block = TriStreamBlock(ch, d_state, d_conv, expand, secret_embed_dim, text_embed_dim)
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            self.up_blocks.append(
                UpBlock(ch, out_ch, num_res_blocks, d_state, d_conv, expand, secret_embed_dim, text_embed_dim)
            )
            ch = out_ch
        self.norm_out = nn.GroupNorm(8, model_channels)
        self.conv_out = nn.Conv2d(model_channels, out_channels, 3, padding=1)
        self.f_sec_proj = nn.Conv2d(secret_embed_dim, model_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        f_sec: torch.Tensor,
        c_txt: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        x: (B, 3, H, W) 噪声/插值状态
        t: (B,) 时间步
        f_sec: (B, secret_embed_dim, H', W') 秘密特征
        c_txt: (B, text_embed_dim)
        Returns:
            v: (B, 3, H, W) 预测速度
            all_struc: 各层 h_struc 列表（用于 rSMI）
            all_tex: 各层 h_tex 列表
        """
        B = x.shape[0]
        t_emb = timestep_embed(t, self.time_embed_input_dim)
        t_emb = self.time_embed(t_emb)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        x = self.conv_in(x) + t_emb
        f_sec = self.f_sec_proj(f_sec)
        all_struc, all_tex = [], []
        skips = []
        for down in self.down_blocks:
            x, struc_list, tex_list = down(x, f_sec, c_txt)
            all_struc.extend(struc_list)
            all_tex.extend(tex_list)
            skips.append(x)
        h_struc_mid, h_tex_mid = self.mid_block(x, f_sec, c_txt)
        all_struc.append(h_struc_mid)
        all_tex.append(h_tex_mid)
        x = h_tex_mid
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = x + skip
            x, struc_list, tex_list = up(x, f_sec, c_txt)
            all_struc.extend(struc_list)
            all_tex.extend(tex_list)
        x = self.norm_out(x)
        x = F.silu(x)
        v = self.conv_out(x)
        return v, all_struc, all_tex
