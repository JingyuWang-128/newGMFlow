from __future__ import annotations

import os

import torch
import torch.nn as nn

# Default to HF mirror in restricted network environments.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from diffusers import AutoencoderKL


class ContinuousVAE(nn.Module):
    """Frozen SD-KL VAE wrapper for continuous latent steganography."""

    def __init__(
        self,
        model_name: str = "stabilityai/sd-vae-ft-mse",
        device: torch.device | str | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        load_kwargs: dict[str, object] = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        try:
            self.vae = AutoencoderKL.from_pretrained(model_name, **load_kwargs)
        except Exception as exc:  # pragma: no cover - network dependent
            endpoint = os.environ.get("HF_ENDPOINT", "<unset>")
            raise RuntimeError(
                f"Failed to load VAE '{model_name}' from Hugging Face endpoint '{endpoint}'. "
                "Please verify network connectivity or set HF_ENDPOINT to an accessible mirror."
            ) from exc
        self.vae.eval()
        self.vae.requires_grad_(False)

        if device is not None:
            self.vae.to(device)

        scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        self.register_buffer(
            "_scaling_factor",
            torch.tensor(scaling_factor, dtype=torch.float32),
            persistent=False,
        )

    def _scale_like(self, x: torch.Tensor) -> torch.Tensor:
        return self._scaling_factor.to(device=x.device, dtype=x.dtype)

    def _vae_param_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.vae.parameters())
        return param.device, param.dtype

    @torch.no_grad()
    def encode_to_latent(self, img: torch.Tensor, sample_posterior: bool = True) -> torch.Tensor:
        """
        Encode image tensor to SD latent tensor.

        Args:
            img: input image in shape [B, 3, H, W], typically normalized to [-1, 1].
            sample_posterior: whether to sample from posterior or use mode.
        """
        input_device, input_dtype = img.device, img.dtype
        vae_device, vae_dtype = self._vae_param_device_dtype()
        img_for_vae = img.to(device=vae_device, dtype=vae_dtype)

        posterior = self.vae.encode(img_for_vae).latent_dist
        latent = posterior.sample() if sample_posterior else posterior.mode()
        latent = latent * self._scale_like(latent)
        return latent.to(device=input_device, dtype=input_dtype)

    @torch.no_grad()
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode SD latent tensor back to image tensor.

        Args:
            latent: latent tensor in shape [B, 4, H, W].
        """
        input_device, input_dtype = latent.device, latent.dtype
        vae_device, vae_dtype = self._vae_param_device_dtype()
        latent_for_vae = latent.to(device=vae_device, dtype=vae_dtype)

        scaled_latent = latent_for_vae / self._scale_like(latent_for_vae)
        decoded = self.vae.decode(scaled_latent).sample
        return decoded.to(device=input_device, dtype=input_dtype)
