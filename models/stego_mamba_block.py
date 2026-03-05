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
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        self.dropout = nn.Dropout(dropout)
        self.alpha_x = nn.Parameter(torch.tensor(0.01))
        self.alpha_z = nn.Parameter(torch.tensor(0.01))

    def forward(
        self,
        h_sem: torch.Tensor,
        h_tex: torch.Tensor,
        c_global: torch.Tensor,
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
        if c_global.ndim != 2:
            raise ValueError(f"c_global must be [B, C], got shape {tuple(c_global.shape)}.")
        if c_global.size(0) != h_sem.size(0):
            raise ValueError(f"c_global batch mismatch: got {c_global.size(0)}, expected {h_sem.size(0)}.")
        if c_global.size(1) != self.dim:
            raise ValueError(f"c_global channel mismatch: got {c_global.size(1)}, expected {self.dim}.")

        bsz, seq_len, _ = h_sem.shape

        # 1) Semantic stream (master): text-conditioned Mamba, no secret injection.
        sem_in = h_sem  # [B, L, C]
        if text_cond is not None:
            if self.text_proj is None:
                raise ValueError("text_cond provided but block was initialized with text_dim=None.")
            text_seq = _align_seq_length(text_cond, target_len=seq_len)  # [B, L, text_dim] or [B, L, C]
            sem_in = sem_in + self.text_proj(text_seq)  # [B, L, C]

        # adaLN-Zero modulation for semantic stream.
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c_global).chunk(3, dim=-1)  # each [B, C]
        shift_msa = shift_msa.unsqueeze(1)  # [B, 1, C]
        scale_msa = scale_msa.unsqueeze(1)  # [B, 1, C]
        gate_msa = gate_msa.unsqueeze(1)  # [B, 1, C]
        normed_sem = self.sem_norm(sem_in)  # [B, L, C]
        sem_in_modulated = normed_sem * (1 + scale_msa) + shift_msa  # [B, L, C]
        sem_hidden = self.sem_mamba(sem_in_modulated)  # [B, L, C]
        h_sem_out = h_sem + gate_msa * sem_hidden  # [B, L, C]

        # 2) Cross-gating from semantic stream.
        mask = torch.sigmoid(self.mask_proj(h_sem_out))  # [B, L, C]

        # 3) Texture stream (slave): dual-channel secret injection.
        tex_base = self.tex_norm(h_tex)  # [B, L, C]
        x = tex_base  # [B, L, C]
        z = self.tex_to_z(tex_base)  # [B, L, C]

        if secret_seq is not None:
            secret_seq = _align_seq_length(secret_seq, target_len=seq_len)  # [B, L, secret_dim]
            # Use tanh to cap energy and apply tiny learnable injection scales.
            x_delta = torch.tanh(self.secret_to_x(secret_seq))
            z_delta = torch.tanh(self.secret_to_z(secret_seq))
            x = x + self.alpha_x * x_delta  # [B, L, C]
            z = z + self.alpha_z * z_delta  # [B, L, C]

        # Stabilize mixed-precision training: avoid Inf*0 -> NaN in gated product.
        z = torch.clamp(z, min=-30.0, max=30.0)
        z_gate = torch.nan_to_num(F.silu(z), nan=0.0, posinf=1e4, neginf=-1e4)
        tex_hidden = torch.nan_to_num(self.tex_mamba(x), nan=0.0, posinf=1e4, neginf=-1e4)
        h_tex_out = torch.nan_to_num(tex_hidden * z_gate * mask, nan=0.0, posinf=1e4, neginf=-1e4)  # [B, L, C]

        # 4) Keep frequency streams physically isolated.
        return h_sem_out, h_tex_out
