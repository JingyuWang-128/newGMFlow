from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F

try:
    import kornia.metrics as kmetrics
except ImportError:  # pragma: no cover
    kmetrics = None


def to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert image tensor from [-1, 1] to [0, 1]."""
    return x.add(1.0).mul(0.5).clamp(0.0, 1.0)


@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, eps: float = 1e-12) -> float:
    """Compute batch PSNR for [B, C, H, W] tensors."""
    mse = F.mse_loss(x, y, reduction="mean")
    if mse.item() <= eps:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor((data_range**2) / mse.item(), device=x.device)).item())


@torch.no_grad()
def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, window_size: int = 11) -> float:
    """Compute batch SSIM; prefers kornia implementation when available."""
    if kmetrics is not None:
        score_map = kmetrics.ssim(x, y, window_size=window_size, max_val=data_range)
        return float(score_map.mean().item())

    # Fallback global SSIM approximation when kornia is unavailable.
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = x.mean(dim=(-1, -2), keepdim=True)
    mu_y = y.mean(dim=(-1, -2), keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(-1, -2), keepdim=True)
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(ssim_map.mean().item())


class LPIPSMetric:
    """Optional LPIPS metric wrapper. Returns NaN if lpips package is unavailable."""

    def __init__(self, device: torch.device, net: str = "alex") -> None:
        self.available = False
        self.metric = None
        try:
            import lpips  # type: ignore

            self.metric = lpips.LPIPS(net=net).to(device).eval()
            self.available = True
        except Exception:
            self.available = False
            self.metric = None

    @torch.no_grad()
    def __call__(self, x_01: torch.Tensor, y_01: torch.Tensor) -> float:
        if not self.available or self.metric is None:
            return float("nan")
        # LPIPS expects normalized range [-1, 1].
        x = x_01.mul(2.0).sub(1.0)
        y = y_01.mul(2.0).sub(1.0)
        val = self.metric(x, y).mean().item()
        return float(val)


@dataclass
class MetricAverager:
    sums: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for k, v in values.items():
            if isinstance(v, float) and math.isnan(v):
                continue
            self.sums[k] = self.sums.get(k, 0.0) + float(v) * n
            self.counts[k] = self.counts.get(k, 0) + n

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, s in self.sums.items():
            c = max(1, self.counts.get(k, 0))
            out[k] = s / c
        return out
