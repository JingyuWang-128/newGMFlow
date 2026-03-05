from __future__ import annotations

import torch
import torch.nn as nn

from .dis import haar_dwt2d

try:
    from mamba_ssm import Mamba as MambaSSM
except Exception:  # pragma: no cover
    MambaSSM = None


class BiMambaLayer(nn.Module):
    """Bidirectional Mamba layer over sequence tokens."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if MambaSSM is None:
            raise ImportError(
                "mamba_ssm is not available. "
                "Please install GPU-enabled torch + mamba-ssm for production training."
            )
        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = MambaSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = MambaSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fuse = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)  # [B, L, C]
        fwd = self.mamba_fwd(h)  # [B, L, C]
        bwd = self.mamba_bwd(torch.flip(h, dims=[1]))  # [B, L, C]
        bwd = torch.flip(bwd, dims=[1])  # [B, L, C]
        out = self.fuse(torch.cat([fwd, bwd], dim=-1))  # [B, L, 2C] -> [B, L, C]
        return x + self.dropout(out)  # [B, L, C]


class ResMambaSecretDecoder(nn.Module):
    """
    Residual Mamba regressor for secret latent reconstruction.

    Input:
      - attacked stego latent: [B, 4, 32, 32] (or generally [B, C, H, W], H/W even)
    Output:
      - predicted secret latent: [B, 4, 32, 32] continuous tensor
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        dim: int = 384,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 4 or num_layers > 6:
            raise ValueError(f"num_layers should be in [4, 6], got {num_layers}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        hf_channels = in_channels * 3  # concat(LH, HL, HH)

        self.hf_in_proj = nn.Linear(hf_channels, dim)
        self.layers = nn.ModuleList(
            [
                BiMambaLayer(
                    dim=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # Predict 2x2 sub-pixels for each output channel via PixelShuffle(2).
        self.pred_head = nn.Linear(dim, out_channels * 4)
        self.up = nn.PixelShuffle(upscale_factor=2)

    @staticmethod
    def _to_seq(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        bsz, ch, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(bsz, h * w, ch).contiguous()
        return seq, h, w

    @staticmethod
    def _from_seq(seq: torch.Tensor, h: int, w: int) -> torch.Tensor:
        bsz, length, ch = seq.shape
        if length != h * w:
            raise ValueError(f"Invalid sequence length {length}, expected {h*w}.")
        return seq.reshape(bsz, h, w, ch).permute(0, 3, 1, 2).contiguous()

    def forward(self, z_attacked: torch.Tensor) -> torch.Tensor:
        if z_attacked.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(z_attacked.shape)}.")
        if z_attacked.size(1) != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_channels}, got {z_attacked.size(1)}."
            )

        # Shallow DWT and use high-frequency bands as decoder input.
        _, lh, hl, hh = haar_dwt2d(z_attacked)  # [B, C, H, W] -> each [B, C, H/2, W/2]
        hf = torch.cat([lh, hl, hh], dim=1)  # [B, 3C, H/2, W/2]
        h, h2, w2 = self._to_seq(hf)  # [B, H/2*W/2, 3C]
        h = self.hf_in_proj(h)  # [B, H/2*W/2, dim]

        for layer in self.layers:
            h = layer(h)  # [B, H/2*W/2, dim]

        pred_seq = self.pred_head(h)  # [B, H/2*W/2, out_ch*4]
        pred_map = self._from_seq(pred_seq, h2, w2)  # [B, out_ch*4, H/2, W/2]
        z_secret_pred = self.up(pred_map)  # [B, out_ch*4, H/2, W/2] -> [B, out_ch, H, W]
        return z_secret_pred
