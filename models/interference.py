from __future__ import annotations

import torch
import torch.nn as nn
import kornia.augmentation as K


def _build_random_jpeg(jpeg_quality: tuple[float, float]) -> nn.Module:
    jpeg_cls = getattr(K, "RandomJpeg", None)
    if jpeg_cls is None:
        jpeg_cls = getattr(K, "RandomJPEG", None)
    if jpeg_cls is None:
        raise ImportError("kornia.augmentation does not provide RandomJpeg/RandomJPEG.")

    try:
        return jpeg_cls(jpeg_quality=jpeg_quality, p=1.0)
    except TypeError:
        try:
            return jpeg_cls(quality=jpeg_quality, p=1.0)
        except TypeError:
            return jpeg_cls(p=1.0)


class LatentInterference(nn.Module):
    """
    GPU-friendly differentiable attack layer for latent-space robust steganography.

    Mix-Attack policy:
      - 20%: identity
      - 80%: uniformly sample one attack from attack pool
    """

    def __init__(
        self,
        latent_hw: tuple[int, int] = (32, 32),
        identity_prob: float = 0.2,
        noise_std: float = 0.03,
        jpeg_quality: tuple[float, float] = (40.0, 90.0),
    ) -> None:
        super().__init__()
        if not 0.0 <= identity_prob <= 1.0:
            raise ValueError(f"identity_prob must be in [0, 1], got {identity_prob}.")

        self.identity_prob = float(identity_prob)
        self.attacks_non_rgb = nn.ModuleList(
            [
                K.RandomResizedCrop(
                    size=latent_hw,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0,
                ),
                K.RandomErasing(
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                    value=0.0,
                    p=1.0,
                ),
                K.RandomGaussianNoise(mean=0.0, std=noise_std, p=1.0),
            ]
        )
        self.jpeg_attack = _build_random_jpeg(jpeg_quality=jpeg_quality)
        self.attacks_rgb = nn.ModuleList([*self.attacks_non_rgb, self.jpeg_attack])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device).item() < self.identity_prob:
            return x

        # JPEG in Kornia only supports 3-channel images; latent tensors are often 4-channel.
        attacks = self.attacks_rgb if x.ndim == 4 and x.size(1) == 3 else self.attacks_non_rgb
        idx = int(torch.randint(0, len(attacks), (1,), device=x.device).item())
        return attacks[idx](x)
