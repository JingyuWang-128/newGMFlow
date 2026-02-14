"""
干扰流形 (Interference Manifold)：可微/可采样干扰算子
Π ∈ {JPEG, Crop, Blur, Noise}，用于鲁棒解码损失与对抗性梯度引导。
"""

import random
from typing import List, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _jpeg_compress_cpu(x: torch.Tensor, quality: int) -> torch.Tensor:
    """简易 JPEG 模拟：DCT 量化（不可微）。用于训练时采样，梯度经解码器回传。"""
    try:
        import cv2
        B, C, H, W = x.shape
        x_np = (x.permute(0, 2, 3, 1).detach().cpu().numpy() * 127.5 + 127.5).clip(0, 255).astype("uint8")
        out = []
        for i in range(B):
            enc = cv2.imencode(".jpg", cv2.cvtColor(x_np[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            out.append(dec)
        out = torch.from_numpy(np.stack(out)).float().to(x.device) / 127.5 - 1.0
        return out.permute(0, 3, 1, 2)
    except Exception:
        return x


class GaussianNoiseOperator(nn.Module):
    """可微高斯噪声：y = x + σ * ε, ε ~ N(0,1)."""

    def __init__(self, sigma_range: tuple = (0.0, 0.2)):
        super().__init__()
        self.sigma_range = sigma_range

    def forward(self, x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        if sigma is None:
            sigma = random.uniform(*self.sigma_range)
        if sigma <= 0:
            return x
        noise = torch.randn_like(x, device=x.device) * sigma
        return x + noise


class BlurOperator(nn.Module):
    """可微高斯模糊。"""

    def __init__(self, sigma_range: tuple = (0.0, 2.0), kernel_size: int = 9):
        super().__init__()
        self.sigma_range = sigma_range
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        if sigma is None:
            sigma = random.uniform(*self.sigma_range)
        if sigma <= 0:
            return x
        k = self.kernel_size
        kh = torch.arange(k, device=x.device).float() - (k - 1) / 2
        kernel = torch.exp(-kh ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1).expand(3, 1, -1)
        kernel_2d = kernel.unsqueeze(-1) * kernel.unsqueeze(-2)
        kernel_2d = kernel_2d.view(1, 3, k, k)
        padding = k // 2
        return F.conv2d(x, kernel_2d, padding=padding, groups=3)


class CropResizeOperator(nn.Module):
    """随机裁剪后 resize 回原尺寸（可微）。"""

    def __init__(self, crop_ratio_range: tuple = (0.5, 1.0)):
        super().__init__()
        self.crop_ratio_range = crop_ratio_range

    def forward(self, x: torch.Tensor, ratio: Optional[float] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        if ratio is None:
            ratio = random.uniform(*self.crop_ratio_range)
        if ratio >= 1.0:
            return x
        crop_h, crop_w = int(H * ratio), int(W * ratio)
        top = random.randint(0, H - crop_h) if H > crop_h else 0
        left = random.randint(0, W - crop_w) if W > crop_w else 0
        x_crop = x[:, :, top : top + crop_h, left : left + crop_w]
        return F.interpolate(x_crop, size=(H, W), mode="bilinear", align_corners=False)


class JPEGOperator:
    """JPEG 压缩：可指定质量；不可微，仅用于前向与损失计算。"""

    def __init__(self, quality_range: tuple = (30, 90)):
        self.quality_range = quality_range

    def __call__(self, x: torch.Tensor, quality: Optional[int] = None) -> torch.Tensor:
        q = quality if quality is not None else random.randint(*self.quality_range)
        return _jpeg_compress_cpu(x, q)


class InterferenceManifold(nn.Module):
    """
    干扰算子流形 M_Π：采样 Π ~ M_Π 并应用。
    训练时对 x̂_0 施加干扰后送入解码器算 L_robust。
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        ops_cfg = config.get("operators", ["jpeg", "gaussian_noise", "crop", "blur"])
        self.noise_op = GaussianNoiseOperator(config.get("noise_sigma_range", [0.0, 0.2]))
        self.blur_op = BlurOperator(config.get("blur_sigma_range", [0.0, 2.0]))
        self.crop_op = CropResizeOperator(config.get("crop_ratio_range", [0.5, 1.0]))
        self.jpeg_op = JPEGOperator(config.get("jpeg_quality_range", [30, 90]))
        self.ops_list = ops_cfg
        self._jpeg_quality_range = config.get("jpeg_quality_range", [30, 90])

    def sample_one(self, x: torch.Tensor) -> torch.Tensor:
        """随机选一个干扰并应用。"""
        name = random.choice(self.ops_list)
        if name == "gaussian_noise":
            return self.noise_op(x)
        if name == "blur":
            return self.blur_op(x)
        if name == "crop":
            return self.crop_op(x)
        if name == "jpeg":
            return self.jpeg_op(x)
        return x

    def forward(self, x: torch.Tensor, num_apply: int = 1) -> torch.Tensor:
        """对 x 施加 num_apply 次随机干扰（每次可能不同）。"""
        for _ in range(num_apply):
            x = self.sample_one(x)
        return x


def build_interference_operators(config: dict) -> InterferenceManifold:
    return InterferenceManifold(config.get("interference", {}))
