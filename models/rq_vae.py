"""
RQ-VAE: Residual Quantized VAE
将秘密图像编码为深度 D 的离散索引图 S ∈ {1,...,K}^{D×H'×W'}，
秘密特征 Z ≈ Σ_d e_{k_d}^{(d)}，支持从粗糙语义到精细残差的层级恢复。
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_res: int = 2):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res = nn.Sequential(*[ResidualBlock(out_ch) for _ in range(num_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.res(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_res: int = 2):
        super().__init__()
        self.res = nn.Sequential(*[ResidualBlock(in_ch) for _ in range(num_res)])
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        return self.up(x)


class ResidualQuantizer(nn.Module):
    """
    单层残差量化：对当前残差选择最近码本向量，输出索引与量化向量。
    """

    def __init__(self, dim: int, num_embeddings: int, commitment_cost: float = 0.25):
        super().__init__()
        self.dim = dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, dim)
        nn.init.uniform_(self.embedding.weight, -1 / num_embeddings, 1 / num_embeddings)

    def forward(
        self, z: torch.Tensor, depth: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, C, H, W), 当前残差特征
        Returns:
            indices: (B, H, W) long
            quantized: (B, C, H, W)
            commitment_loss: scalar
        """
        B, C, H, W = z.shape
        z_flat = rearrange(z, "b c h w -> b (h w) c")
        d = self.embedding.weight
        dist = torch.cdist(z_flat, d)
        indices = dist.argmin(dim=-1)
        quantized = F.embedding(indices, self.embedding.weight)
        quantized = rearrange(quantized, "b (h w) c -> b c h w", h=H, w=W)
        commitment_loss = F.mse_loss(z, quantized) * self.commitment_cost
        return indices, quantized, commitment_loss

    def get_quantized_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (B, H, W) -> (B, C, H, W); (B, L) -> (B, L, C)."""
        q = self.embedding(indices)
        if q.dim() == 4:
            return q
        return q

    def get_codebook(self) -> torch.Tensor:
        return self.embedding.weight


class RQVAE(nn.Module):
    """
    RQ-VAE: 多深度残差量化 VAE。
    Z ≈ Σ_d e_{k_d}^{(d)}，输出每层索引图 S_d 与连续特征 Z。
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 256,
        num_embeddings: int = 8192,
        num_layers: int = 4,
        downsample: int = 8,
        commitment_cost: float = 0.25,
        channel_mult: List[int] = None,
    ):
        super().__init__()
        channel_mult = channel_mult or [1, 2, 2, 4]
        self.latent_channels = latent_channels
        self.num_layers = num_layers
        self.num_embeddings = num_embeddings
        self.downsample = downsample
        chs = [in_channels] + [latent_channels * m for m in channel_mult]
        enc_blocks = []
        for i in range(len(chs) - 1):
            enc_blocks.append(EncoderBlock(chs[i], chs[i + 1]))
        self.encoder = nn.Sequential(*enc_blocks)
        self.enc_to_latent = nn.Conv2d(chs[-1], latent_channels, 1)

        self.quantizers = nn.ModuleList([
            ResidualQuantizer(latent_channels, num_embeddings, commitment_cost) for _ in range(num_layers)
        ])

        self.latent_to_dec = nn.Conv2d(latent_channels, chs[-1], 1)
        dec_blocks = []
        for i in range(len(chs) - 1, 0, -1):
            dec_blocks.append(DecoderBlock(chs[i], chs[i - 1]))
        self.decoder = nn.Sequential(*dec_blocks)
        self.dec_out = nn.Conv2d(chs[0], in_channels, 3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.enc_to_latent(h)

    def quantize_residual(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        残差量化：逐层量化残差，返回连续特征、每层索引、每层量化向量、commitment 损失和。
        """
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices = []
        all_quantized = []
        total_commit = 0.0
        for q in self.quantizers:
            residual_cur = residual - quantized_sum
            indices, quantized, commit_loss = q(residual_cur)
            total_commit = total_commit + commit_loss
            quantized_sum = quantized_sum + quantized
            all_indices.append(indices)
            all_quantized.append(quantized)
        return quantized_sum, all_indices, all_quantized, total_commit

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.latent_to_dec(z)
        h = self.decoder(h)
        return torch.tanh(self.dec_out(h))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        x: (B, 3, H, W) 秘密图像
        Returns:
            recon: 重建图像
            z_q: 量化后连续特征 (B, C, H', W')
            indices_list: 每层 (B, H', W')
            quantized_list: 每层 (B, C, H', W')
            commitment_loss: scalar
        """
        z = self.encode(x)
        z_q, indices_list, quantized_list, commitment_loss = self.quantize_residual(z)
        recon = self.decode(z_q)
        return recon, z_q, indices_list, quantized_list, commitment_loss

    def get_indices(self, x: torch.Tensor) -> List[torch.Tensor]:
        """仅编码并返回各层离散索引，用于隐写与解码。"""
        z = self.encode(x)
        _, indices_list, _, _ = self.quantize_residual(z)
        return indices_list

    def decode_from_indices(self, indices_list: List[torch.Tensor]) -> torch.Tensor:
        """从各层索引重建图像。indices_list: 每层 (B, H', W')。"""
        B = indices_list[0].shape[0]
        H, W = indices_list[0].shape[1], indices_list[0].shape[2]
        device = indices_list[0].device
        z_q = torch.zeros(B, self.latent_channels, H, W, device=device)
        for d, (indices, q) in enumerate(zip(indices_list, self.quantizers)):
            q_vec = q.get_quantized_from_indices(indices)
            if q_vec.dim() == 3:
                q_vec = rearrange(q_vec, "b (h w) c -> b c h w", h=H, w=W)
            else:
                q_vec = rearrange(q_vec, "b h w c -> b c h w")
            z_q = z_q + q_vec
        return self.decode(z_q)

    def decode_from_indices_partial(
        self, indices_list: List[torch.Tensor], depth_from: int = 0, depth_to: Optional[int] = None
    ) -> torch.Tensor:
        """仅用 depth_from 到 depth_to 的层重建（用于层级可视化）。"""
        depth_to = depth_to or len(indices_list)
        B = indices_list[0].shape[0]
        H, W = indices_list[0].shape[1], indices_list[0].shape[2]
        device = indices_list[0].device
        z_q = torch.zeros(B, self.latent_channels, H, W, device=device)
        for d in range(depth_from, min(depth_to, len(self.quantizers))):
            indices = indices_list[d]
            q_vec = self.quantizers[d].get_quantized_from_indices(indices)
            if q_vec.dim() == 3:
                q_vec = rearrange(q_vec, "b (h w) c -> b c h w", h=H, w=W)
            else:
                q_vec = rearrange(q_vec, "b h w c -> b c h w")
            z_q = z_q + q_vec
        return self.decode(z_q)
