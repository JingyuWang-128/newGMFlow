"""
GenMamba-Flow 训练脚本
阶段：1) RQ-VAE 预训练（可选） 2) 生成器 + 解码器联合训练（Flow + rSMI + Robust + 解码器 CE/hDCE）
"""

import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.config import get_config
from data.datasets import get_train_dataloader
from models.rq_vae import RQVAE
from models.tri_stream_mamba import TriStreamMambaUNet
from models.rectified_flow import RectifiedFlowGenerator
from models.decoder import RobustDecoder
from models.interference import InterferenceManifold
from losses.alignment import rSMIAlignmentLoss
from losses.robust import RobustDecodingLoss
from losses.contrastive import hDCELoss


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_text_encoder(config, device):
    """简单文本编码：占位或 CLIP。"""
    dim = config.get("generator", {}).get("text_embed_dim", 768)
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        def encode(text_list):
            with torch.no_grad():
                t = tokenizer(text_list).to(device)
                return model.encode_text(t).float()
        return encode, dim
    except Exception:
        class PlaceholderTextEncoder(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.embed = nn.Embedding(1000, dim)
            def forward(self, text_list):
                B = len(text_list)
                return self.embed(torch.randint(0, 1000, (B,), device=next(self.parameters()).device))
        enc = PlaceholderTextEncoder(dim).to(device)
        return lambda texts: enc(texts), dim


def build_models(config, device):
    rq_cfg = config.get("rq_vae", {})
    gen_cfg = config.get("generator", {})
    dec_cfg = config.get("decoder", {})
    rq_vae = RQVAE(
        in_channels=rq_cfg.get("in_channels", 3),
        latent_channels=rq_cfg.get("latent_channels", 256),
        num_embeddings=rq_cfg.get("num_embeddings", 8192),
        num_layers=rq_cfg.get("num_layers", 4),
        downsample=rq_cfg.get("downsample", 8),
        commitment_cost=rq_cfg.get("commitment_cost", 0.25),
    ).to(device)
    unet = TriStreamMambaUNet(
        in_channels=gen_cfg.get("in_channels", 3),
        model_channels=gen_cfg.get("model_channels", 256),
        out_channels=gen_cfg.get("out_channels", 3),
        num_res_blocks=gen_cfg.get("num_res_blocks", 2),
        channel_mult=gen_cfg.get("channel_mult", [1, 2, 3, 4]),
        d_state=gen_cfg.get("d_state", 16),
        d_conv=gen_cfg.get("d_conv", 4),
        expand=gen_cfg.get("expand", 2),
        text_embed_dim=gen_cfg.get("text_embed_dim", 768),
        secret_embed_dim=gen_cfg.get("secret_embed_dim", 256),
    ).to(device)
    flow = RectifiedFlowGenerator(unet, num_steps=gen_cfg.get("num_flow_steps", 1000)).to(device)
    decoder = RobustDecoder(
        in_channels=dec_cfg.get("in_channels", 3),
        hidden_dim=dec_cfg.get("hidden_dim", 256),
        num_layers=dec_cfg.get("num_layers", 4),
        d_state=dec_cfg.get("d_state", 16),
        d_conv=dec_cfg.get("d_conv", 4),
        expand=dec_cfg.get("expand", 2),
        num_rq_depths=dec_cfg.get("num_rq_depths", 4),
        num_embeddings=dec_cfg.get("num_embeddings", 8192),
    ).to(device)
    interference = InterferenceManifold(config.get("interference", {})).to(device)
    return rq_vae, flow, decoder, interference


def train_rq_vae(rq_vae, dataloader, config, device, save_dir):
    """预训练 RQ-VAE 仅用秘密图像重建。"""
    opt = torch.optim.AdamW(rq_vae.parameters(), lr=config.get("train", {}).get("lr_rq_vae", 1e-4), weight_decay=0.01)
    steps = config.get("data", {}).get("num_train_samples") or 10000
    steps = min(steps, 20000)
    pbar = tqdm(range(steps), desc="RQ-VAE pretrain")
    it = iter(dataloader)
    for s in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)
        secret = batch["secret"].to(device)
        recon, z_q, indices_list, quantized_list, commit_loss = rq_vae(secret)
        recon_loss = F.mse_loss(recon, secret)
        loss = recon_loss + commit_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rq_vae.parameters(), 1.0)
        opt.step()
        pbar.set_postfix(recon=recon_loss.item(), commit=commit_loss.item())
        if (s + 1) % 5000 == 0:
            torch.save(rq_vae.state_dict(), Path(save_dir) / "rq_vae_pretrain.pt")
    torch.save(rq_vae.state_dict(), Path(save_dir) / "rq_vae.pt")


def train_main(config, device, resume_path=None):
    proj = config.get("project", {})
    save_dir = Path(proj.get("output_dir", "./outputs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(proj.get("seed", 42))

    dataloader = get_train_dataloader(config)
    rq_vae, flow, decoder, interference = build_models(config, device)
    text_encoder, text_dim = get_text_encoder(config, device)
    if isinstance(text_encoder, nn.Module):
        text_encoder = text_encoder

    loss_cfg = config.get("loss", {})
    L_align_fn = rSMIAlignmentLoss(
        temperature=loss_cfg.get("align_temperature", 0.07),
        lambda_reg=0.01,
    )
    L_robust_fn = RobustDecodingLoss(depth_weights=loss_cfg.get("robust_depth_weights", [1.0, 0.8, 0.6, 0.4]))
    lambda_flow = loss_cfg.get("lambda_flow", 1.0)
    lambda_align = loss_cfg.get("lambda_align", 0.1)
    lambda_robust = loss_cfg.get("lambda_robust", 0.5)

    opt_gen = torch.optim.AdamW(
        list(flow.parameters()) + list(rq_vae.parameters()),
        lr=config.get("train", {}).get("lr_generator", 1e-4),
        weight_decay=config.get("train", {}).get("weight_decay", 0.01),
    )
    opt_dec = torch.optim.AdamW(
        decoder.parameters(),
        lr=config.get("train", {}).get("lr_decoder", 2e-4),
        weight_decay=config.get("train", {}).get("weight_decay", 0.01),
    )
    scaler = GradScaler() if device.type == "cuda" else None
    grad_clip = config.get("train", {}).get("grad_clip", 1.0)

    # 可选：加载预训练 RQ-VAE
    rq_path = save_dir / "rq_vae.pt"
    if rq_path.exists():
        rq_vae.load_state_dict(torch.load(rq_path, map_location=device))
    elif config.get("train", {}).get("pretrain_rq", True):
        train_rq_vae(rq_vae, dataloader, config, device, save_dir)

    rq_vae.eval()
    num_steps = config.get("train", {}).get("epochs", 50) * len(dataloader)
    log_every = proj.get("log_every", 100)
    save_every = proj.get("save_every", 5000)
    global_step = 0
    for epoch in range(config.get("train", {}).get("epochs", 50)):
        flow.train()
        decoder.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            cover = batch["cover"].to(device)
            secret = batch["secret"].to(device)
            texts = batch.get("text", ["a natural image"] * cover.shape[0])
            if isinstance(texts, str):
                texts = [texts] * cover.shape[0]
            B = cover.shape[0]
            with torch.no_grad():
                f_sec = rq_vae.encode(secret)
                _, _, quantized_list, _ = rq_vae.quantize_residual(f_sec)
                f_sec = sum(quantized_list)
                indices_list = rq_vae.get_indices(secret)
            if callable(text_encoder) and not isinstance(text_encoder, nn.Module):
                c_txt = text_encoder(texts)
            else:
                c_txt = text_encoder(texts)
            if c_txt.dim() == 1:
                c_txt = c_txt.unsqueeze(0)
            x_0 = torch.randn_like(cover, device=device)
            t = torch.rand(B, device=device)
            x_t = flow.get_x_t(x_0, cover, t)
            v_target = flow.get_velocity_target(x_0, cover)
            with autocast(enabled=(device.type == "cuda")):
                v_pred, all_struc, all_tex = flow(x_t, t, f_sec, c_txt)
                L_flow = F.mse_loss(v_pred, v_target)
                L_align = L_align_fn(all_struc, all_tex)
                x0_hat = flow.predict_x0(x_t, t, v_pred)
                x0_pert = interference(x0_hat.detach(), num_apply=1)
                logits_list = decoder(x0_pert)
                L_robust = L_robust_fn(logits_list, indices_list)
                L_gen = lambda_flow * L_flow + lambda_align * L_align + lambda_robust * L_robust
            opt_gen.zero_grad()
            if scaler:
                scaler.scale(L_gen).backward()
                scaler.unscale_(opt_gen, None)
                torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
                scaler.step(opt_gen)
            else:
                L_gen.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
                opt_gen.step()

            with autocast(enabled=(device.type == "cuda")):
                with torch.no_grad():
                    stego = flow.sample(cover.shape, f_sec, c_txt, num_steps=8, device=device)
                logits_list_d = decoder(stego)
                L_dec = L_robust_fn(logits_list_d, indices_list)
            opt_dec.zero_grad()
            if scaler:
                scaler.scale(L_dec).backward()
                scaler.step(opt_dec)
                scaler.update()
            else:
                L_dec.backward()
                opt_dec.step()

            global_step += 1
            if global_step % log_every == 0:
                pbar.set_postfix(flow=L_flow.item(), align=L_align.item(), robust=L_robust.item(), dec=L_dec.item())
            if global_step % save_every == 0:
                torch.save({
                    "flow": flow.state_dict(),
                    "decoder": decoder.state_dict(),
                    "rq_vae": rq_vae.state_dict(),
                    "step": global_step,
                }, save_dir / "checkpoint.pt")
    torch.save({
        "flow": flow.state_dict(),
        "decoder": decoder.state_dict(),
        "rq_vae": rq_vae.state_dict(),
    }, save_dir / "final.pt")
    print("Training done. Saved to", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = get_config(args.config, args.override)
    device = torch.device(args.device)
    train_main(config, device, resume_path=args.resume)
