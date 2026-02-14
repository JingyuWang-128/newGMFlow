"""
Mamba 风格选择性状态空间模块 (Selective SSM)
纯 PyTorch 实现，支持 2D 特征经 Cross-Scan 转为 1D 序列后处理。
可选：若安装 mamba-ssm 则使用官方 Mamba 加速。
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

try:
    from mamba_ssm import Mamba as MambaSSM
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False


class SelectiveSSM(nn.Module):
    """
    简化选择性 SSM：y = SSM(x, Δ, A, B, C)。
    输入 x (B, L, D)，输出 (B, L, D)。
    使用离散化 A, B 与选择性 Δ, B, C 的近似实现（非 CUDA 级优化）。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        self.expand = expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)

        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> (d n)", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def _selective_scan_approx(self, u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """简化扫描：h_t = (1 - delta*A) * h_{t-1} + delta * B * u_t, y_t = C * h_t + D * u_t."""
        B_batch, L, D_in = u.shape
        h = torch.zeros(B_batch, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        outs = []
        for t in range(L):
            ut = u[:, t, :]
            dt = delta[:, t, :]
            bt = B[:, t, :]
            ct = C[:, t, :]
            h = h * (1 - dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0).to(u.dtype))
            h = h + dt.unsqueeze(-1) * bt.unsqueeze(-1) * ut.unsqueeze(-1)
            yt = (h * ct.unsqueeze(-2)).sum(-1) + self.D.to(u.dtype) * ut
            outs.append(yt)
        return torch.stack(outs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        x_dbl = self.x_proj(x)
        delta, B_sel, C_sel = x_dbl.split([self.d_inner, self.d_state, self.d_inner], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.view(self.d_inner, self.d_state))
        h = self._selective_scan_approx(x, delta, A, B_sel, C_sel)
        return self.out_proj(h * F.silu(z))


class MambaBlock(nn.Module):
    """单路 Mamba 块：LayerNorm + SSM + 残差。"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, use_mamba_ssm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.use_mamba_ssm = use_mamba_ssm and HAS_MAMBA_SSM
        if self.use_mamba_ssm:
            self.ssm = MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.ssm = SelectiveSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


def _build_2d_scans(h: int, w: int, device: torch.device) -> List[torch.Tensor]:
    """生成 2D 多方向扫描顺序（用于 Cross-Scan）。返回多个 (H*W,) 索引。"""
    indices = torch.arange(h * w, device=device).view(h, w)
    scans = []
    scans.append(indices.flatten())
    scans.append(indices.flip(0).flatten())
    scans.append(indices.flip(1).flatten())
    scans.append(indices.flip(0).flip(1).flatten())
    return scans


class CrossScan(nn.Module):
    """
    将 2D 特征 (B, C, H, W) 按多方向扫描为序列，经 mamba_fn 后合并回 (B, C, H, W)。
    用于秘密序列注入时重排空间顺序。
    """

    def __init__(self, merge: bool = True):
        super().__init__()
        self.merge = merge

    @staticmethod
    def flatten_2d_to_seq(x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C)."""
        return rearrange(x, "b c h w -> b (h w) c")

    @staticmethod
    def seq_to_2d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(B, L, C) -> (B, C, H, W)."""
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

    def apply_scan_merge(self, x: torch.Tensor, mamba_fn: nn.Module) -> torch.Tensor:
        """x: (B, C, H, W). 多方向扫描后经 mamba_fn 合并。"""
        B, C, H, W = x.shape
        device = x.device
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        scans = _build_2d_scans(H, W, device)
        out_list = []
        for scan in scans:
            x_scan = x_flat[:, scan, :]
            y_scan = mamba_fn(x_scan)
            inv_scan = torch.zeros_like(scan)
            inv_scan[scan] = torch.arange(len(scan), device=device)
            y_scan = y_scan[:, inv_scan, :]
            out_list.append(y_scan)
        out = torch.stack(out_list, dim=0).mean(0)
        return rearrange(out, "b (h w) c -> b c h w", h=H, w=W)


class Mamba2DBlock(nn.Module):
    """2D 特征上的 Mamba 块：Cross-Scan + Mamba + 残差。"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scan = CrossScan(merge=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        x_flat = x_flat + self.ssm(self.norm(x_flat))
        return rearrange(x_flat, "b (h w) c -> b c h w", h=H, w=W)
