from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as MambaSSM
except Exception:  # pragma: no cover
    MambaSSM = None


def _align_seq_length(seq: torch.Tensor, target_len: int) -> torch.Tensor:
    """Align [B, L, C] or [B, C] input to [B, target_len, C]."""
    if seq.ndim == 2:
        seq = seq.unsqueeze(1)
    if seq.ndim != 3:
        raise ValueError(f"Expected 2D/3D tensor, got shape {tuple(seq.shape)}.")

    if seq.size(1) == target_len:
        return seq
    if seq.size(1) == 1:
        return seq.expand(-1, target_len, -1)
    raise ValueError(
        f"Cannot align sequence length: got {seq.size(1)}, expected 1 or {target_len}."
    )


class LatentStegoDiSBlock(nn.Module):
    """
    Micro block for latent stego DiS:
      1) semantic stream (master): Mamba(text-conditioned), no secret injection
      2) cross-gating: mask = sigmoid(Linear(H_sem_out))
      3) texture stream (slave): dual-channel secret injection on x and z
      4) return (H_sem_out, H_tex_out) without concat/frequency merging
    """

    def __init__(
        self,
        dim: int,
        text_dim: Optional[int],
        secret_dim: int,
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

        self.dim = dim
        self.secret_dim = secret_dim

        self.sem_norm = nn.LayerNorm(dim)
        self.tex_norm = nn.LayerNorm(dim)
        self.sem_mamba = MambaSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.tex_mamba = MambaSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        if text_dim is None:
            self.text_proj = None
        elif text_dim == dim:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(text_dim, dim)

        self.mask_proj = nn.Linear(dim, dim)
        self.tex_to_z = nn.Linear(dim, dim)
        self.secret_to_x = nn.Linear(secret_dim, dim)
        self.secret_to_z = nn.Linear(secret_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_sem: torch.Tensor,
        h_tex: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        secret_seq: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h_sem.ndim != 3 or h_tex.ndim != 3:
            raise ValueError(
                f"h_sem/h_tex must be [B, L, C], got {tuple(h_sem.shape)} and {tuple(h_tex.shape)}."
            )
        if h_sem.shape != h_tex.shape:
            raise ValueError(
                f"h_sem and h_tex must share shape, got {tuple(h_sem.shape)} vs {tuple(h_tex.shape)}."
            )

        bsz, seq_len, _ = h_sem.shape

        # 1) Semantic stream (master): text-conditioned Mamba, no secret injection.
        sem_in = h_sem  # [B, L, C]
        if text_cond is not None:
            if self.text_proj is None:
                raise ValueError("text_cond provided but block was initialized with text_dim=None.")
            text_seq = _align_seq_length(text_cond, target_len=seq_len)  # [B, L, text_dim] or [B, L, C]
            sem_in = sem_in + self.text_proj(text_seq)  # [B, L, C]

        sem_hidden = self.sem_mamba(self.sem_norm(sem_in))  # [B, L, C] -> [B, L, C]
        h_sem_out = h_sem + self.dropout(sem_hidden)  # [B, L, C]

        # 2) Cross-gating from semantic stream.
        mask = torch.sigmoid(self.mask_proj(h_sem_out))  # [B, L, C]

        # 3) Texture stream (slave): dual-channel secret injection.
        tex_base = self.tex_norm(h_tex)  # [B, L, C]
        x = tex_base  # [B, L, C]
        z = self.tex_to_z(tex_base)  # [B, L, C]

        if secret_seq is not None:
            secret_seq = _align_seq_length(secret_seq, target_len=seq_len)  # [B, L, secret_dim]
            x = x + self.secret_to_x(secret_seq)  # [B, L, C]
            z = z + self.secret_to_z(secret_seq)  # [B, L, C]

        # Stabilize mixed-precision training: avoid Inf*0 -> NaN in gated product.
        z = torch.clamp(z, min=-30.0, max=30.0)
        z_gate = torch.nan_to_num(F.silu(z), nan=0.0, posinf=1e4, neginf=-1e4)
        tex_hidden = torch.nan_to_num(self.tex_mamba(x), nan=0.0, posinf=1e4, neginf=-1e4)
        h_tex_out = torch.nan_to_num(tex_hidden * z_gate * mask, nan=0.0, posinf=1e4, neginf=-1e4)  # [B, L, C]

        # 4) Keep frequency streams physically isolated.
        return h_sem_out, h_tex_out
