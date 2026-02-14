"""
GenMamba-Flow 训练脚本（单脚本完成全部阶段，支持多卡 DDP）
阶段：1) RQ-VAE 预训练 2) 生成器 + 解码器联合训练
可视化：损失曲线、隐写对比图（定期保存）
"""

import argparse
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.config import get_config
from data.datasets import get_train_dataloader
from models.rq_vae import RQVAE
from models.dis import TriStreamDiS
from models.rectified_flow import RectifiedFlowGenerator
from models.decoder import RobustDecoder
from models.interference import InterferenceManifold
from losses.alignment import rSMIAlignmentLoss
from losses.robust import RobustDecodingLoss
from utils.vis_plots import plot_loss_curves, save_loss_history


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_text_encoder(config, device):
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


def build_backbone(config):
    """根据 config 构建 DiS 主干。"""
    gen_cfg = config.get("generator", {})
    return TriStreamDiS(
        img_size=gen_cfg.get("img_size", 256),
        patch_size=gen_cfg.get("patch_size", 4),
        in_channels=gen_cfg.get("in_channels", 3),
        model_channels=gen_cfg.get("model_channels", 256),
        out_channels=gen_cfg.get("out_channels", 3),
        num_layers=gen_cfg.get("num_layers", 12),
        d_state=gen_cfg.get("d_state", 16),
        d_conv=gen_cfg.get("d_conv", 4),
        expand=gen_cfg.get("expand", 2),
        text_embed_dim=gen_cfg.get("text_embed_dim", 768),
        secret_embed_dim=gen_cfg.get("secret_embed_dim", 256),
    )


def build_models(config, device, use_ddp: bool = False, rank: int = 0):
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
    backbone = build_backbone(config).to(device)
    flow = RectifiedFlowGenerator(backbone, num_steps=gen_cfg.get("num_flow_steps", 1000)).to(device)
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
    if use_ddp:
        flow = DDP(flow, device_ids=[rank])
        decoder = DDP(decoder, device_ids=[rank])
    return rq_vae, flow, decoder, interference


def train_rq_vae(rq_vae, dataloader, config, device, save_dir, rank: int = 0):
    """阶段1：预训练 RQ-VAE（仅 rank0 执行并保存）。"""
    if rank != 0:
        return
    opt = torch.optim.AdamW(rq_vae.parameters(), lr=config.get("train", {}).get("lr_rq_vae", 1e-4), weight_decay=0.01)
    steps = config.get("data", {}).get("num_train_samples") or 10000
    steps = min(steps, 20000)
    pbar = tqdm(range(steps), desc="[Stage1] RQ-VAE pretrain")
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


def train_main(config, device, rank: int = 0, world_size: int = 1, resume_path=None):
    proj = config.get("project", {})
    save_dir = Path(proj.get("output_dir", "./outputs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    set_seed(proj.get("seed", 42) + rank)

    use_ddp = world_size > 1
    dataloader = get_train_dataloader(config, rank=rank, world_size=world_size)
    rq_vae, flow, decoder, interference = build_models(config, device, use_ddp=use_ddp, rank=rank)
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

    flow_module = flow.module if use_ddp else flow
    decoder_module = decoder.module if use_ddp else decoder
    opt_gen = torch.optim.AdamW(
        list(flow_module.parameters()) + list(rq_vae.parameters()),
        lr=config.get("train", {}).get("lr_generator", 1e-4),
        weight_decay=config.get("train", {}).get("weight_decay", 0.01),
    )
    opt_dec = torch.optim.AdamW(
        decoder_module.parameters(),
        lr=config.get("train", {}).get("lr_decoder", 2e-4),
        weight_decay=config.get("train", {}).get("weight_decay", 0.01),
    )
    scaler = GradScaler() if device.type == "cuda" else None
    grad_clip = config.get("train", {}).get("grad_clip", 1.0)

    rq_path = save_dir / "rq_vae.pt"
    if rq_path.exists():
        rq_vae.load_state_dict(torch.load(rq_path, map_location=device))
    elif config.get("train", {}).get("pretrain_rq", True):
        dataloader_single = get_train_dataloader(config, rank=0, world_size=1) if use_ddp else dataloader
        train_rq_vae(rq_vae, dataloader_single, config, device, save_dir, rank=rank)
        if use_ddp:
            torch.save(rq_vae.state_dict(), rq_path)
    if use_ddp:
        dist.barrier()
        if rank != 0:
            rq_vae.load_state_dict(torch.load(rq_path, map_location=device))

    rq_vae.eval()
    log_every = proj.get("log_every", 100)
    save_every = proj.get("save_every", 5000)
    vis_every = proj.get("vis_every", 1000)
    loss_history = []
    global_step = 0

    for epoch in range(config.get("train", {}).get("epochs", 50)):
        if use_ddp:
            dataloader.sampler.set_epoch(epoch)
        flow.train()
        decoder.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
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
            c_txt = text_encoder(texts)
            if c_txt.dim() == 1:
                c_txt = c_txt.unsqueeze(0)
            x_0 = torch.randn_like(cover, device=device)
            t = torch.rand(B, device=device)
            x_t = flow_module.get_x_t(x_0, cover, t)
            v_target = flow_module.get_velocity_target(x_0, cover)
            with autocast(enabled=(device.type == "cuda")):
                v_pred, all_struc, all_tex = flow(x_t, t, f_sec, c_txt)
                L_flow = F.mse_loss(v_pred, v_target)
                L_align = L_align_fn(all_struc, all_tex)
                x0_hat = flow_module.predict_x0(x_t, t, v_pred)
                x0_pert = interference(x0_hat.detach(), num_apply=1)
                logits_list = decoder(x0_pert)
                L_robust = L_robust_fn(logits_list, indices_list)
                L_gen = lambda_flow * L_flow + lambda_align * L_align + lambda_robust * L_robust
            opt_gen.zero_grad()
            if scaler:
                scaler.scale(L_gen).backward()
                scaler.unscale_(opt_gen, None)
                torch.nn.utils.clip_grad_norm_(flow_module.parameters(), grad_clip)
                scaler.step(opt_gen)
            else:
                L_gen.backward()
                torch.nn.utils.clip_grad_norm_(flow_module.parameters(), grad_clip)
                opt_gen.step()

            with autocast(enabled=(device.type == "cuda")):
                with torch.no_grad():
                    stego = flow_module.sample(cover.shape, f_sec, c_txt, num_steps=8, device=device)
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
            if rank == 0:
                loss_history.append({
                    "step": global_step,
                    "loss_flow": L_flow.item(),
                    "loss_align": L_align.item(),
                    "loss_robust": L_robust.item(),
                    "loss_dec": L_dec.item(),
                    "loss_gen": L_gen.item(),
                })
            if global_step % log_every == 0 and rank == 0:
                pbar.set_postfix(flow=L_flow.item(), align=L_align.item(), robust=L_robust.item(), dec=L_dec.item())
            if global_step % vis_every == 0 and rank == 0:
                plot_loss_curves(loss_history, str(vis_dir / "loss_curves.png"), smooth=min(50, len(loss_history) // 2))
                save_loss_history(loss_history, str(save_dir / "loss_history.json"))
                with torch.no_grad():
                    stego_vis = flow_module.sample(cover[:4].shape, f_sec[:4], c_txt[:4] if c_txt is not None else None, num_steps=8, device=device)
                    pred_indices = decoder_module.predict_indices(stego_vis)
                    recovered = rq_vae.decode_from_indices(pred_indices)
                from utils.visualization import save_stego_comparison
                save_stego_comparison(cover[:4], stego_vis, secret[:4], recovered, str(vis_dir / f"stego_step{global_step}.png"), nrow=4)
            if global_step % save_every == 0:
                if rank == 0:
                    torch.save({
                        "flow": flow_module.state_dict(),
                        "decoder": decoder_module.state_dict(),
                        "rq_vae": rq_vae.state_dict(),
                        "step": global_step,
                    }, save_dir / "checkpoint.pt")
                if use_ddp:
                    dist.barrier()

    if rank == 0:
        plot_loss_curves(loss_history, str(vis_dir / "loss_curves_final.png"), smooth=min(50, max(1, len(loss_history) // 2)))
        save_loss_history(loss_history, str(save_dir / "loss_history.json"))
        torch.save({
            "flow": flow_module.state_dict(),
            "decoder": decoder_module.state_dict(),
            "rq_vae": rq_vae.state_dict(),
        }, save_dir / "final.pt")
        print("Training done. Saved to", save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = get_config(args.config, args.override)
    proj = config.get("project", {})

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_main(config, device, rank=rank, world_size=world_size, resume_path=args.resume)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
