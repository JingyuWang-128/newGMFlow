from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# Use mirror endpoint for any Hugging Face downloads.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.datasets import get_train_dataloader
from models.continuous_vae import ContinuousVAE
from models.decoder import ResMambaSecretDecoder
from models.dis import TriStreamLatentDiS
from models.interference import LatentInterference


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0


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
        "train": {
            "epochs": 10,
            "lr": 1e-4,
            "min_lr_ratio": 0.1,
            "warmup_steps": 500,
            "weight_decay": 1e-2,
            "lambda_secret": 1.0,
            "max_grad_norm": 1.0,
            "amp": True,
            "use_checkpoint": True,
            "detach_for_robust": True,
            "detach_warmup_steps": 2000,
            "save_dir": "checkpoints",
            "save_every": 1000,
            "log_every": 50,
            "log_file": "train_metrics.jsonl",
            "seed": 42,
        },
        "attack": {
            "identity_prob": 0.2,
            "noise_std": 0.03,
            "jpeg_quality": [40.0, 90.0],
            "latent_hw": [32, 32],
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


def flow_matching_target(z_data: torch.Tensor, z_noise: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rectified-flow style interpolation:
      z_t = (1 - t) * z_data + t * z_noise
      v*  = z_noise - z_data
    """
    t_view = t.view(-1, 1, 1, 1)
    z_t = (1.0 - t_view) * z_data + t_view * z_noise
    v_target = z_noise - z_data
    return z_t, v_target


def should_detach_for_robust(train_cfg: Dict[str, Any], global_step: int) -> bool:
    base_detach = bool(train_cfg.get("detach_for_robust", True))
    warmup_steps = int(train_cfg.get("detach_warmup_steps", 0))
    if not base_detach:
        return False
    return global_step < warmup_steps


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    min_lr_ratio = float(min(max(min_lr_ratio, 0.0), 1.0))
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        # Linear warmup: [0, warmup_steps] -> [0, 1]
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        # Cosine decay: [warmup_steps, total_steps] -> [1, min_lr_ratio]
        progress_denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(progress_denom)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_checkpoint(
    save_dir: str,
    state: TrainState,
    generator: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: Dict[str, Any],
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"step_{state.global_step:08d}.pt")
    torch.save(
        {
            "state": {"epoch": state.epoch, "global_step": state.global_step},
            "generator": generator.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
        },
        ckpt_path,
    )
    return ckpt_path


def train(config: Dict[str, Any], resume_path: str | None = None) -> None:
    train_cfg = config["train"]
    model_cfg = config["model"]
    attack_cfg = config["attack"]

    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Frozen continuous VAE for image <-> latent mapping.
    vae = ContinuousVAE(
        model_name=model_cfg["vae_model_name"],
        device=device,
        torch_dtype=torch.float16 if train_cfg.get("amp", True) and torch.cuda.is_available() else None,
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
        use_checkpoint=bool(train_cfg.get("use_checkpoint", True)),
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

    interference = LatentInterference(
        latent_hw=tuple(attack_cfg.get("latent_hw", [32, 32])),
        identity_prob=float(attack_cfg.get("identity_prob", 0.2)),
        noise_std=float(attack_cfg.get("noise_std", 0.03)),
        jpeg_quality=tuple(attack_cfg.get("jpeg_quality", [40.0, 90.0])),
    ).to(device)

    params = list(generator.parameters()) + list(decoder.parameters())
    optimizer = AdamW(
        params,
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(train_cfg.get("amp", True) and device.type == "cuda"),
    )

    dataloader = get_train_dataloader(config)
    state = TrainState()
    lambda_secret = float(train_cfg.get("lambda_secret", 1.0))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    save_dir = str(train_cfg.get("save_dir", "checkpoints"))
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, str(train_cfg.get("log_file", "train_metrics.jsonl")))
    epochs = int(train_cfg.get("epochs", 10))
    total_steps = max(1, epochs * len(dataloader))
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=int(train_cfg.get("warmup_steps", 500)),
        min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.1)),
    )

    generator.train()
    decoder.train()
    interference.train()

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        decoder.load_state_dict(ckpt["decoder"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        ckpt_state = ckpt.get("state", {})
        state.epoch = int(ckpt_state.get("epoch", 0))
        state.global_step = int(ckpt_state.get("global_step", 0))
        print(f"[Resume] loaded checkpoint: {resume_path} (global_step={state.global_step})")

    start_step = state.global_step
    start_time = time.time()
    if state.global_step == 0:
        _append_jsonl(
            log_path,
            {
                "event": "train_start",
                "time": datetime.now().isoformat(timespec="seconds"),
                "device": str(device),
                "total_steps": total_steps,
                "config": config,
            },
        )
    else:
        _append_jsonl(
            log_path,
            {
                "event": "train_resume",
                "time": datetime.now().isoformat(timespec="seconds"),
                "device": str(device),
                "resume_path": resume_path,
                "global_step": state.global_step,
                "total_steps": total_steps,
            },
        )

    pbar = tqdm(
        total=total_steps,
        initial=min(state.global_step, total_steps),
        desc="Latent-DiS-Stego Training",
        dynamic_ncols=True,
    )
    for epoch in range(epochs):
        state.epoch = epoch
        for batch in dataloader:
            if state.global_step >= total_steps:
                break
            cover = batch["cover"].to(device, non_blocking=True)
            secret = batch["secret"].to(device, non_blocking=True)

            with torch.no_grad():
                z_cover = vae.encode_to_latent(cover)  # [B, 3, H, W] -> [B, 4, H/8, W/8]
                z_secret = vae.encode_to_latent(secret)  # [B, 3, H, W] -> [B, 4, H/8, W/8]

            # Rectified-flow training target.
            bsz = z_cover.size(0)
            t = torch.rand(bsz, device=device, dtype=z_cover.dtype)
            z_noise = torch.randn_like(z_cover)  # [B, 4, H/8, W/8]
            z_noisy, v_target = flow_matching_target(z_cover, z_noise, t)  # both [B, 4, H/8, W/8]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                v_pred = generator(z_noisy, text_cond=None, secret_cond=z_secret)  # [B, 4, H/8, W/8]
                l_gen = F.mse_loss(v_pred, v_target)

                t_view = t.view(-1, 1, 1, 1)
                z_stego_hat = z_noisy - t_view * v_pred  # [B, 4, H/8, W/8]

                detach_flag = should_detach_for_robust(train_cfg, state.global_step)
                z_attacked = interference(z_stego_hat.detach() if detach_flag else z_stego_hat)  # [B, 4, H/8, W/8]

                z_secret_pred = decoder(z_attacked)  # [B, 4, H/8, W/8]
                l_secret = F.mse_loss(z_secret_pred, z_secret)

                l_total = l_gen + lambda_secret * l_secret

            non_finite_reason = None
            if not torch.isfinite(v_pred).all():
                non_finite_reason = "v_pred_non_finite"
            elif not torch.isfinite(z_attacked).all():
                non_finite_reason = "z_attacked_non_finite"
            elif not torch.isfinite(z_secret_pred).all():
                non_finite_reason = "z_secret_pred_non_finite"
            elif not torch.isfinite(l_gen):
                non_finite_reason = "l_gen_non_finite"
            elif not torch.isfinite(l_secret):
                non_finite_reason = "l_secret_non_finite"
            elif not torch.isfinite(l_total):
                non_finite_reason = "l_total_non_finite"

            if non_finite_reason is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                scheduler.step()
                state.global_step += 1
                pbar.update(1)
                _append_jsonl(
                    log_path,
                    {
                        "event": "non_finite_skip",
                        "epoch": state.epoch,
                        "step": state.global_step,
                        "reason": non_finite_reason,
                        "lr": _current_lr(optimizer),
                        "time": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                continue

            scaler.scale(l_total).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            grad_norm_value = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

            if not math.isfinite(grad_norm_value):
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                scheduler.step()
                state.global_step += 1
                pbar.update(1)
                _append_jsonl(
                    log_path,
                    {
                        "event": "non_finite_skip",
                        "epoch": state.epoch,
                        "step": state.global_step,
                        "reason": "grad_norm_non_finite",
                        "lr": _current_lr(optimizer),
                        "time": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                continue

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            state.global_step += 1
            pbar.update(1)

            if state.global_step % log_every == 0:
                elapsed = time.time() - start_time
                effective_steps = max(1, state.global_step - start_step)
                steps_per_sec = effective_steps / max(1e-6, elapsed)
                eta_sec = (total_steps - state.global_step) / max(1e-6, steps_per_sec)
                lr = _current_lr(optimizer)
                log_item = {
                    "event": "train_step",
                    "epoch": state.epoch,
                    "step": state.global_step,
                    "loss": float(l_total.detach().item()),
                    "l_gen": float(l_gen.detach().item()),
                    "l_secret": float(l_secret.detach().item()),
                    "grad_norm": grad_norm_value,
                    "lr": lr,
                    "detach": int(detach_flag),
                    "steps_per_sec": steps_per_sec,
                    "eta_sec": eta_sec,
                    "time": datetime.now().isoformat(timespec="seconds"),
                }
                _append_jsonl(log_path, log_item)
                pbar.set_postfix(
                    step=state.global_step,
                    loss=float(l_total.detach().item()),
                    l_gen=float(l_gen.detach().item()),
                    l_secret=float(l_secret.detach().item()),
                    detach=int(detach_flag),
                    lr=f"{lr:.2e}",
                    gnorm=f"{grad_norm_value:.3f}",
                )

            if state.global_step % save_every == 0:
                ckpt_path = save_checkpoint(save_dir, state, generator, decoder, optimizer, scheduler, scaler, config)
                _append_jsonl(
                    log_path,
                    {
                        "event": "checkpoint",
                        "step": state.global_step,
                        "path": ckpt_path,
                        "time": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                tqdm.write(f"[Checkpoint] saved to {ckpt_path}")
        if state.global_step >= total_steps:
            break

    pbar.close()
    final_ckpt = save_checkpoint(save_dir, state, generator, decoder, optimizer, scheduler, scaler, config)
    _append_jsonl(
        log_path,
        {
            "event": "train_end",
            "step": state.global_step,
            "final_checkpoint": final_ckpt,
            "elapsed_sec": time.time() - start_time,
            "time": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(f"[Done] final checkpoint saved to: {final_ckpt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Latent-DiS-Stego.")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config.")
    parser.add_argument("--dump-default-config", type=str, default=None, help="Dump default config json path.")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dump_default_config:
        cfg = _default_config()
        dump_path = Path(args.dump_default_config)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"Default config written to: {dump_path}")
    else:
        cfg = load_config(args.config)
        train(cfg, resume_path=args.resume)
