from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# Use mirror endpoint for any Hugging Face downloads.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.datasets import get_test_dataloader, get_val_dataloader
from models.continuous_vae import ContinuousVAE
from models.decoder import ResMambaSecretDecoder
from models.dis import TriStreamLatentDiS
from utils.metrics import LPIPSMetric, MetricAverager, psnr, ssim, to_01


def _default_config() -> Dict[str, Any]:
    return {
        "data": {
            "batch_size": 4,
            "num_workers": 4,
            "image_size": 256,
            "secret_size": 256,
        },
        "model": {
            "vae_model_name": "stabilityai/sd-vae-ft-mse",
            "latent_channels": 4,
            "generator_dim": 512,
            "generator_depth": 12,
            "generator_d_state": 16,
            "generator_d_conv": 4,
            "generator_expand": 2,
            "generator_dropout": 0.0,
            "secret_dim": 512,
            "decoder_dim": 384,
            "decoder_layers": 4,
            "decoder_d_state": 16,
            "decoder_d_conv": 4,
            "decoder_expand": 2,
            "decoder_dropout": 0.0,
            "text_dim": None,
        },
        "test": {
            "seed": 42,
            "num_samples": 200,
            "batch_size": 4,
            "t_embed": 0.5,
            "save_dir": "eval_outputs",
            "vis_max_per_attack": 20,
            "split": "val",  # val: has cover+secret; test: secret only
        },
    }


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(config_path: str | None) -> Dict[str, Any]:
    cfg = _default_config()
    if config_path is None:
        return cfg
    if yaml is None:
        raise ImportError("PyYAML is required to load config files. Please install `pyyaml`.")
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_update(cfg, user_cfg)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _jpeg_attack(x_01: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """Apply deterministic JPEG compression on [0,1] tensor image batch."""
    outs = []
    for i in range(x_01.size(0)):
        pil_img = TF.to_pil_image(x_01[i].cpu())
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        jpeg_img = Image.open(buf).convert("RGB")
        outs.append(TF.to_tensor(jpeg_img))
    return torch.stack(outs, dim=0).to(device=x_01.device, dtype=x_01.dtype)


def attack_clean(x_m11: torch.Tensor) -> torch.Tensor:
    return x_m11


def attack_crop_center_05(x_m11: torch.Tensor) -> torch.Tensor:
    bsz, _, h, w = x_m11.shape
    ch = max(1, int(round(h * 0.5)))
    cw = max(1, int(round(w * 0.5)))
    top = (h - ch) // 2
    left = (w - cw) // 2
    cropped = x_m11[:, :, top : top + ch, left : left + cw]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


def attack_jpeg_q50(x_m11: torch.Tensor) -> torch.Tensor:
    x_01 = to_01(x_m11)
    out = _jpeg_attack(x_01, quality=50)
    return out.mul(2.0).sub(1.0).clamp(-1.0, 1.0)


def attack_gaussian_blur_sigma2(x_m11: torch.Tensor) -> torch.Tensor:
    x_01 = to_01(x_m11)
    import kornia.filters as KF

    y_01 = KF.gaussian_blur2d(x_01, kernel_size=(9, 9), sigma=(2.0, 2.0))
    return y_01.mul(2.0).sub(1.0).clamp(-1.0, 1.0)


def attack_noise_std01(x_m11: torch.Tensor) -> torch.Tensor:
    x_01 = to_01(x_m11)
    noise = torch.randn_like(x_01) * 0.1
    y_01 = (x_01 + noise).clamp(0.0, 1.0)
    return y_01.mul(2.0).sub(1.0).clamp(-1.0, 1.0)


def build_attack_pool() -> Dict[str, Any]:
    return {
        "Clean": attack_clean,
        "Crop(0.5)": attack_crop_center_05,
        "JPEG(Q=50)": attack_jpeg_q50,
        "GaussianBlur(sigma=2.0)": attack_gaussian_blur_sigma2,
        "Noise(std=0.1)": attack_noise_std01,
    }


@torch.no_grad()
def embed_stego_latent(
    generator: TriStreamLatentDiS,
    z_cover: torch.Tensor,
    z_secret: torch.Tensor,
    t_embed: float = 0.5,
) -> torch.Tensor:
    """
    One-step latent embedding used in evaluation:
      z_noisy = (1-t) * z_cover + t * N(0, I)
      v_pred  = G(z_noisy, secret=z_secret)
      z_stego = z_noisy - t * v_pred
    """
    t = torch.full((z_cover.size(0),), fill_value=t_embed, device=z_cover.device, dtype=z_cover.dtype)  # [B]
    t_view = t.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
    z_noise = torch.randn_like(z_cover)  # [B, 4, H/8, W/8]
    z_noisy = (1.0 - t_view) * z_cover + t_view * z_noise  # [B, 4, H/8, W/8]
    v_pred = generator(z_noisy, t=t, text_cond=None, secret_cond=z_secret)  # [B, 4, H/8, W/8]
    z_stego = z_noisy - t_view * v_pred  # [B, 4, H/8, W/8]
    return z_stego


def _iter_batches(dataloader: DataLoader, split: str) -> Iterable[Dict[str, torch.Tensor]]:
    for batch in dataloader:
        if split == "val":
            yield {"cover": batch["cover"], "secret": batch["secret"]}
        else:
            # test split may only provide secrets, so use zeros as cover proxy.
            secret = batch["secret"]
            cover = torch.zeros_like(secret)
            yield {"cover": cover, "secret": secret}


def evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    test_cfg = cfg["test"]
    model_cfg = cfg["model"]
    set_seed(int(test_cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(test_cfg.get("save_dir", "eval_outputs"))
    vis_dir = save_dir / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    vae = ContinuousVAE(
        model_name=model_cfg["vae_model_name"],
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    generator = TriStreamLatentDiS(
        in_channels=model_cfg["latent_channels"],
        dim=model_cfg["generator_dim"],
        depth=model_cfg["generator_depth"],
        text_dim=model_cfg.get("text_dim"),
        secret_dim=model_cfg["secret_dim"],
        d_state=model_cfg["generator_d_state"],
        d_conv=model_cfg["generator_d_conv"],
        expand=model_cfg["generator_expand"],
        dropout=model_cfg["generator_dropout"],
        use_checkpoint=False,
    ).to(device)
    decoder = ResMambaSecretDecoder(
        in_channels=model_cfg["latent_channels"],
        out_channels=model_cfg["latent_channels"],
        dim=model_cfg["decoder_dim"],
        num_layers=model_cfg["decoder_layers"],
        d_state=model_cfg["decoder_d_state"],
        d_conv=model_cfg["decoder_d_conv"],
        expand=model_cfg["decoder_expand"],
        dropout=model_cfg["decoder_dropout"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    generator.load_state_dict(ckpt["generator"], strict=True)
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    generator.eval()
    decoder.eval()

    split = str(test_cfg.get("split", "val"))
    batch_size = int(args.batch_size or test_cfg.get("batch_size", cfg["data"]["batch_size"]))
    if split == "val":
        dataloader = get_val_dataloader(cfg, batch_size=batch_size)
    else:
        dataloader = get_test_dataloader(cfg, batch_size=batch_size)

    attack_pool = build_attack_pool()
    attack_meters = {name: MetricAverager() for name in attack_pool}
    carrier_meter = MetricAverager()
    lpips_metric = LPIPSMetric(device=device, net=args.lpips_net)
    vis_max = int(test_cfg.get("vis_max_per_attack", 20))
    vis_count = {name: 0 for name in attack_pool}

    num_samples = int(args.num_samples or test_cfg.get("num_samples", 200))
    t_embed = float(args.t_embed or test_cfg.get("t_embed", 0.5))
    seen = 0

    pbar = tqdm(total=num_samples, desc="Robustness Evaluation", dynamic_ncols=True)
    for batch in _iter_batches(dataloader, split=split):
        if seen >= num_samples:
            break

        cover = batch["cover"].to(device, non_blocking=True)
        secret_gt = batch["secret"].to(device, non_blocking=True)
        bsz = cover.size(0)
        keep = min(bsz, num_samples - seen)
        cover = cover[:keep]
        secret_gt = secret_gt[:keep]

        with torch.no_grad():
            z_cover = vae.encode_to_latent(cover, sample_posterior=False)  # [B, 3, H, W] -> [B, 4, H/8, W/8]
            z_secret_gt = vae.encode_to_latent(secret_gt, sample_posterior=False)  # [B, 3, H, W] -> [B, 4, H/8, W/8]

            # Stego generation in latent, then decode to carrier pixel space.
            z_stego = embed_stego_latent(generator, z_cover, z_secret_gt, t_embed=t_embed)  # [B, 4, H/8, W/8]
            stego_img = vae.decode_from_latent(z_stego)  # [B, 3, H, W]

        # Carrier quality on clean stego.
        cover_01 = to_01(cover)
        stego_01 = to_01(stego_img)
        carrier_meter.update(
            {
                "carrier_psnr": psnr(stego_01, cover_01),
                "carrier_ssim": ssim(stego_01, cover_01),
            },
            n=keep,
        )

        # Attack loop and extraction pipeline.
        for attack_name, attack_fn in attack_pool.items():
            attacked_stego = attack_fn(stego_img)  # [B, 3, H, W]

            with torch.no_grad():
                z_attacked = vae.encode_to_latent(attacked_stego, sample_posterior=False)  # [B, 4, H/8, W/8]
                z_secret_pred = decoder(z_attacked)  # [B, 4, H/8, W/8]
                secret_rec = vae.decode_from_latent(z_secret_pred)  # [B, 3, H, W]

            secret_gt_01 = to_01(secret_gt)
            secret_rec_01 = to_01(secret_rec)
            metrics = {
                "psnr": psnr(secret_rec_01, secret_gt_01),
                "ssim": ssim(secret_rec_01, secret_gt_01),
                "lpips": lpips_metric(secret_rec_01, secret_gt_01),
            }
            attack_meters[attack_name].update(metrics, n=keep)

            # Visualization: [secret_gt | stego_clean | attacked_stego | secret_rec]
            can_save = vis_count[attack_name] < vis_max
            if can_save:
                for i in range(keep):
                    if vis_count[attack_name] >= vis_max:
                        break
                    panel = torch.cat(
                        [
                            secret_gt[i : i + 1],
                            stego_img[i : i + 1],
                            attacked_stego[i : i + 1],
                            secret_rec[i : i + 1],
                        ],
                        dim=-1,
                    )
                    out_path = vis_dir / f"{attack_name.replace('/', '_')}_{vis_count[attack_name]:04d}.png"
                    save_image(to_01(panel), str(out_path))
                    vis_count[attack_name] += 1

        seen += keep
        pbar.update(keep)
    pbar.close()

    results = {
        "checkpoint": str(args.checkpoint),
        "num_samples": seen,
        "lpips_available": lpips_metric.available,
        "carrier_quality": carrier_meter.compute(),
        "attacks": {name: meter.compute() for name, meter in attack_meters.items()},
        "visualizations_dir": str(vis_dir),
    }

    result_path = save_dir / "metrics.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[Saved] metrics: {result_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent-DiS-Stego robustness evaluation.")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override test batch size.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override evaluated sample count.")
    parser.add_argument("--t-embed", type=float, default=None, help="Override one-step latent embed t.")
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
