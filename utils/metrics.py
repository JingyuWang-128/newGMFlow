"""
评估指标：FID, CLIP Score, Stego-LPIPS, Bit Accuracy, Recovery PSNR/SSIM, 隐写分析检测率
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np


def compute_psnr_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> tuple:
    """pred/target: (B,3,H,W) in [-1,1]. Returns (psnr_mean, ssim_mean)."""
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    psnr = (10 * torch.log10(data_range ** 2 / (mse + 1e-8))).mean().item()
    ssim_val = _ssim_batch(pred, target, data_range=1.0)
    return psnr, ssim_val


def _ssim_batch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> float:
    C = x.shape[1]
    w = _gaussian_window(window_size, C, x.device)
    mu_x = F.conv2d(x, w, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, w, padding=window_size // 2, groups=C)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, w, padding=window_size // 2, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, w, padding=window_size // 2, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(x * y, w, padding=window_size // 2, groups=C) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = (2 * mu_xy + c1) * (2 * sigma_xy + c2) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return ssim.mean().item()


def _gaussian_window(size: int, channels: int, device: torch.device) -> torch.Tensor:
    sigma = 1.5
    coords = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = g.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size)
    return w


def compute_bit_accuracy(pred_indices: List[torch.Tensor], target_indices: List[torch.Tensor]) -> float:
    """pred/target: 每层 (B, H', W'). 返回平均 token 准确率。"""
    total = 0
    correct = 0
    for p, t in zip(pred_indices, target_indices):
        mask = t >= 0
        total += mask.sum().item()
        correct += ((p == t) & mask).sum().item()
    return correct / max(total, 1)


def compute_lpips(model, x: torch.Tensor, y: torch.Tensor) -> float:
    """x, y: (B,3,H,W). 返回平均 LPIPS 距离。"""
    with torch.no_grad():
        d = model(x, y)
    return d.mean().item()


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*sqrt(Sigma_r*Sigma_f))."""
    mu_r, mu_f = real_features.mean(axis=0), fake_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)
    eps = 1e-6
    diff = mu_r - mu_f
    covmean, _ = np.linalg.sqrtm(sigma_r.dot(sigma_f), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_r) + np.trace(sigma_f) - 2 * np.trace(covmean)
    return float(fid)


def compute_clip_score(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """image_features (B, D), text_features (B, D) normalized. Return (B,) cosine sim."""
    return (image_features * text_features).sum(dim=-1)
