from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .stego_mamba_block import LatentStegoDiSBlock


def haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable 2D Haar DWT: returns (LL, LH, HL, HH)."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got shape {tuple(x.shape)}.")
    if x.size(-2) % 2 != 0 or x.size(-1) % 2 != 0:
        raise ValueError(f"H and W must be even for Haar DWT, got {tuple(x.shape[-2:])}.")

    x00 = x[:, :, 0::2, 0::2]  # [B, C, H/2, W/2]
    x01 = x[:, :, 0::2, 1::2]  # [B, C, H/2, W/2]
    x10 = x[:, :, 1::2, 0::2]  # [B, C, H/2, W/2]
    x11 = x[:, :, 1::2, 1::2]  # [B, C, H/2, W/2]

    ll = (x00 + x01 + x10 + x11) * 0.5  # [B, C, H/2, W/2]
    lh = (x00 - x01 + x10 - x11) * 0.5  # [B, C, H/2, W/2]
    hl = (x00 + x01 - x10 - x11) * 0.5  # [B, C, H/2, W/2]
    hh = (x00 - x01 - x10 + x11) * 0.5  # [B, C, H/2, W/2]
    return ll, lh, hl, hh


def haar_idwt2d(
    ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
) -> torch.Tensor:
    """Differentiable 2D Haar IDWT from (LL, LH, HL, HH)."""
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError("LL/LH/HL/HH must share the same shape.")
    if ll.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got shape {tuple(ll.shape)}.")

    x00 = (ll + lh + hl + hh) * 0.5  # [B, C, H/2, W/2]
    x01 = (ll - lh + hl - hh) * 0.5  # [B, C, H/2, W/2]
    x10 = (ll + lh - hl - hh) * 0.5  # [B, C, H/2, W/2]
    x11 = (ll - lh - hl + hh) * 0.5  # [B, C, H/2, W/2]

    bsz, ch, h, w = ll.shape
    out = ll.new_zeros((bsz, ch, h * 2, w * 2))  # [B, C, H, W]
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11
    return out


class TriStreamLatentDiS(nn.Module):
    """
    Dual-track latent DiS backbone:
      - semantic stream for LL
      - texture stream for concat(LH, HL, HH)
      - long skip connection is handled ONLY at macro loop level
    """

    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 512,
        depth: int = 12,
        text_dim: Optional[int] = None,
        secret_dim: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if depth % 2 != 0:
            raise ValueError(f"depth must be even for dual long-skip design, got {depth}.")

        self.in_channels = in_channels
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.half_depth = depth // 2

        # Input projections after DWT split.
        self.sem_in_proj = nn.Linear(in_channels, dim)
        self.tex_in_proj = nn.Linear(in_channels * 3, dim)
        self.secret_map_proj = nn.Linear(in_channels * 4, secret_dim)

        self.blocks = nn.ModuleList(
            [
                LatentStegoDiSBlock(
                    dim=dim,
                    text_dim=text_dim,
                    secret_dim=secret_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # Deep-stage long skip fusion: Concat + Linear for each stream independently.
        self.skip_fuse_sem = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(self.half_depth)])
        self.skip_fuse_tex = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(self.half_depth)])

        self.sem_out_norm = nn.LayerNorm(dim)
        self.tex_out_norm = nn.LayerNorm(dim)
        self.sem_out_proj = nn.Linear(dim, in_channels)
        self.tex_out_proj = nn.Linear(dim, in_channels * 3)

    @staticmethod
    def _to_seq(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        bsz, ch, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(bsz, h * w, ch).contiguous()  # [B, C, H, W] -> [B, HW, C]
        return seq, h, w

    @staticmethod
    def _from_seq(seq: torch.Tensor, h: int, w: int) -> torch.Tensor:
        bsz, length, ch = seq.shape
        if length != h * w:
            raise ValueError(f"Invalid seq length {length}, expected {h*w} for ({h}, {w}).")
        return seq.reshape(bsz, h, w, ch).permute(0, 3, 1, 2).contiguous()  # [B, HW, C] -> [B, C, H, W]

    def _run_block(
        self,
        block: LatentStegoDiSBlock,
        h_sem: torch.Tensor,
        h_tex: torch.Tensor,
        text_cond: Optional[torch.Tensor],
        secret_seq: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_checkpoint and self.training:
            def custom_forward(sem: torch.Tensor, tex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return block(sem, tex, text_cond=text_cond, secret_seq=secret_seq)

            return checkpoint(custom_forward, h_sem, h_tex, use_reentrant=False)

        return block(h_sem, h_tex, text_cond=text_cond, secret_seq=secret_seq)

    def _prepare_secret_seq(
        self, secret_cond: Optional[torch.Tensor], target_len: int
    ) -> Optional[torch.Tensor]:
        if secret_cond is None:
            return None

        if secret_cond.ndim == 4:
            s_ll, s_lh, s_hl, s_hh = haar_dwt2d(secret_cond)  # [B, C, H, W] -> 4 * [B, C, H/2, W/2]
            s_cat = torch.cat([s_ll, s_lh, s_hl, s_hh], dim=1)  # [B, 4C, H/2, W/2]
            s_seq, _, _ = self._to_seq(s_cat)  # [B, H/2*W/2, 4C]
            return self.secret_map_proj(s_seq)  # [B, H/2*W/2, secret_dim]

        if secret_cond.ndim == 3:
            if secret_cond.size(1) != target_len:
                raise ValueError(
                    f"secret sequence length mismatch: got {secret_cond.size(1)}, expected {target_len}."
                )
            return secret_cond

        if secret_cond.ndim == 2:
            return secret_cond

        raise ValueError(
            f"Unsupported secret_cond shape {tuple(secret_cond.shape)}. "
            "Expected [B,C,H,W], [B,L,D], or [B,D]."
        )

    def forward(
        self,
        z_noisy: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        secret_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1) Latent DWT decomposition and sequence conversion.
        z_sem, z_lh, z_hl, z_hh = haar_dwt2d(z_noisy)  # [B, C, H, W] -> each [B, C, H/2, W/2]
        z_tex = torch.cat([z_lh, z_hl, z_hh], dim=1)  # [B, 3C, H/2, W/2]

        h_sem, h2, w2 = self._to_seq(z_sem)  # [B, H/2*W/2, C]
        h_tex, _, _ = self._to_seq(z_tex)  # [B, H/2*W/2, 3C]

        h_sem = self.sem_in_proj(h_sem)  # [B, H/2*W/2, dim]
        h_tex = self.tex_in_proj(h_tex)  # [B, H/2*W/2, dim]
        secret_seq = self._prepare_secret_seq(secret_cond, target_len=h_sem.size(1))  # [B, H/2*W/2, secret_dim]

        # 2) Macro dual-track long skip connection.
        skip_sem: list[torch.Tensor] = []
        skip_tex: list[torch.Tensor] = []

        for i, block in enumerate(self.blocks):
            if i < self.half_depth:
                h_sem, h_tex = self._run_block(block, h_sem, h_tex, text_cond, secret_seq)  # both [B, HW/4, dim]
                skip_sem.append(h_sem)
                skip_tex.append(h_tex)
                continue

            skip_idx = i - self.half_depth
            h_sem = self.skip_fuse_sem[skip_idx](torch.cat([skip_sem.pop(), h_sem], dim=-1))  # [B, HW/4, 2dim] -> [B, HW/4, dim]
            h_tex = self.skip_fuse_tex[skip_idx](torch.cat([skip_tex.pop(), h_tex], dim=-1))  # [B, HW/4, 2dim] -> [B, HW/4, dim]
            h_sem, h_tex = self._run_block(block, h_sem, h_tex, text_cond, secret_seq)  # both [B, HW/4, dim]

        # 3) Sequence -> 2D and IDWT reconstruction to full latent velocity field.
        ll_seq = self.sem_out_proj(self.sem_out_norm(h_sem))  # [B, HW/4, C]
        tex_seq = self.tex_out_proj(self.tex_out_norm(h_tex))  # [B, HW/4, 3C]

        ll_map = self._from_seq(ll_seq, h2, w2)  # [B, C, H/2, W/2]
        tex_map = self._from_seq(tex_seq, h2, w2)  # [B, 3C, H/2, W/2]
        lh_map, hl_map, hh_map = torch.chunk(tex_map, 3, dim=1)  # each [B, C, H/2, W/2]

        z_velocity = haar_idwt2d(ll_map, lh_map, hl_map, hh_map)  # [B, C, H, W]
        return z_velocity
