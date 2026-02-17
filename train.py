"""
GenMamba-Flow 训练脚本（单脚本完成全部阶段，支持多卡 DDP）
阶段：1) RQ-VAE 预训练 2) 生成器 + 解码器联合训练
可视化：损失曲线、隐写对比图（定期保存）
"""

import argparse
import os
import random
import sys
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
from data.datasets import get_train_dataloader, get_secret_train_dataloader
from models.rq_vae import RQVAE
from models.dis import TriStreamDiS
from models.rectified_flow import RectifiedFlowGenerator
from models.decoder import RobustDecoder
from models.interference import InterferenceManifold
from losses.alignment import rSMIAlignmentLoss
from losses.robust import RobustDecodingLoss
from losses.contrastive import hDCELoss
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
    """根据 config 构建 DiS 主干（更深/更宽以提升隐写与重建能力）。"""
    gen_cfg = config.get("generator", {})
    return TriStreamDiS(
        img_size=gen_cfg.get("img_size", 256),
        patch_size=gen_cfg.get("patch_size", 4),
        in_channels=gen_cfg.get("in_channels", 3),
        model_channels=gen_cfg.get("model_channels", 384),
        out_channels=gen_cfg.get("out_channels", 3),
        num_layers=gen_cfg.get("num_layers", 18),
        d_state=gen_cfg.get("d_state", 16),
        d_conv=gen_cfg.get("d_conv", 4),
        expand=gen_cfg.get("expand", 2),
        text_embed_dim=gen_cfg.get("text_embed_dim", 768),
        secret_embed_dim=gen_cfg.get("secret_embed_dim", 256),
        use_mamba_ssm=gen_cfg.get("use_mamba_ssm", True),
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
        channel_mult=rq_cfg.get("channel_mult", [1, 2, 2, 4]),
        num_res=rq_cfg.get("num_res", 3),
    ).to(device)
    backbone = build_backbone(config).to(device)
    flow = RectifiedFlowGenerator(backbone, num_steps=gen_cfg.get("num_flow_steps", 1000)).to(device)
    decoder = RobustDecoder(
        in_channels=dec_cfg.get("in_channels", 3),
        hidden_dim=dec_cfg.get("hidden_dim", 384),
        num_layers=dec_cfg.get("num_layers", 8),
        d_state=dec_cfg.get("d_state", 16),
        d_conv=dec_cfg.get("d_conv", 4),
        expand=dec_cfg.get("expand", 2),
        num_rq_depths=dec_cfg.get("num_rq_depths", 4),
        num_embeddings=dec_cfg.get("num_embeddings", 8192),
        latent_channels=rq_cfg.get("latent_channels", 256),
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
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr_rq_vae", 1e-4)
    lr = float(lr) if isinstance(lr, str) else lr
    opt = torch.optim.AdamW(rq_vae.parameters(), lr=lr, weight_decay=0.01)
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


def train_main(config, device, rank: int = 0, world_size: int = 1, resume_path=None, stage: str = "all"):
    """stage: 1=仅RQ-VAE, 2=仅生成器+解码器, all=两阶段都跑。"""
    proj = config.get("project", {})
    save_dir = Path(proj.get("output_dir", "./outputs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    set_seed(proj.get("seed", 42) + rank)

    use_ddp = world_size > 1
    rq_path = save_dir / "rq_vae.pt"

    if rank == 0:
        from data.datasets import print_split_stats
        print_split_stats(config)

    dataloader = get_train_dataloader(config, rank=rank, world_size=world_size)
    rq_vae, flow, decoder, interference = build_models(config, device, use_ddp=use_ddp, rank=rank)

    # ---------- 阶段 1 ----------
    if stage == "1":
        if rank == 0:
            dataloader_s1 = get_secret_train_dataloader(config)
            train_rq_vae(rq_vae, dataloader_s1, config, device, save_dir, rank=rank)
            torch.save(rq_vae.state_dict(), rq_path)
        if use_ddp:
            dist.barrier()
            if rank != 0:
                rq_vae.load_state_dict(torch.load(rq_path, map_location=device))
        if rank == 0:
            print("Stage 1 done. RQ-VAE saved to", rq_path)
        return

    # ---------- 阶段 2（stage=all 时若无 rq_vae.pt 先跑阶段1再阶段2）----------
    # 仅当明确指定 --stage 2 且没有预训练权重时报错
    if stage == "2" and not rq_path.exists():
        raise FileNotFoundError("Stage 2 需要已预训练的 RQ-VAE: %s 不存在。请先运行阶段1: python train.py --stage 1" % rq_path)

    if stage == "all" and not rq_path.exists():
        if rank == 0:
            dataloader_s1 = get_secret_train_dataloader(config)
            train_rq_vae(rq_vae, dataloader_s1, config, device, save_dir, rank=rank)
            torch.save(rq_vae.state_dict(), rq_path)
        if use_ddp:
            dist.barrier()
            if rank != 0:
                rq_vae.load_state_dict(torch.load(rq_path, map_location=device))
        if rank == 0:
            print("Stage 1 done. Starting Stage 2.")
    else:
        rq_vae.load_state_dict(torch.load(rq_path, map_location=device))
        if use_ddp and rank != 0:
            rq_vae.load_state_dict(torch.load(rq_path, map_location=device))
    rq_vae.eval()

    text_encoder, _ = get_text_encoder(config, device)
    if isinstance(text_encoder, nn.Module):
        text_encoder = text_encoder

    loss_cfg = config.get("loss", {})
    L_align_fn = rSMIAlignmentLoss(temperature=loss_cfg.get("align_temperature", 0.07), lambda_reg=0.01)
    L_robust_fn = RobustDecodingLoss(depth_weights=loss_cfg.get("robust_depth_weights", [1.0, 0.8, 0.6, 0.4]))
    use_hdce = loss_cfg.get("use_hdce", True)
    lambda_hdce = loss_cfg.get("lambda_hdce", 0.5)
    hdce_fn = hDCELoss(temperature=loss_cfg.get("hdce_temperature", 0.07), num_hard=loss_cfg.get("hdce_num_hard", 16)) if use_hdce else None
    lambda_flow = loss_cfg.get("lambda_flow", 1.0)
    lambda_align = loss_cfg.get("lambda_align", 0.1)
    lambda_robust = loss_cfg.get("lambda_robust", 0.5)
    detach_for_robust = loss_cfg.get("detach_for_robust", True)

    flow_module = flow.module if use_ddp else flow
    decoder_module = decoder.module if use_ddp else decoder
    train_cfg = config.get("train", {})
    def _num(k, default):
        v = train_cfg.get(k, default)
        return float(v) if isinstance(v, str) else v
    opt_gen = torch.optim.AdamW(
        list(flow_module.parameters()) + list(rq_vae.parameters()),
        lr=_num("lr_generator", 1e-4),
        weight_decay=_num("weight_decay", 0.01),
    )
    opt_dec = torch.optim.AdamW(
        decoder_module.parameters(),
        lr=_num("lr_decoder", 2e-4),
        weight_decay=_num("weight_decay", 0.01),
    )
    scaler = GradScaler() if device.type == "cuda" else None
    grad_clip = _num("grad_clip", 1.0)

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
        pbar = tqdm(dataloader, desc="Epoch %d" % (epoch + 1), disable=(rank != 0))
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
                x0_pert = interference(x0_hat.detach() if detach_for_robust else x0_hat, num_apply=1)
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
                if use_hdce and hdce_fn is not None:
                    logits_list_d, feat_256 = decoder(stego, return_feat=True)
                    L_dec_ce = L_robust_fn(logits_list_d, indices_list)
                    L_hdce = 0.0
                    for d in range(len(indices_list)):
                        cb = rq_vae.quantizers[d].get_codebook()
                        idx = indices_list[d].flatten(1)
                        L_hdce = L_hdce + hdce_fn(feat_256, cb, idx)
                    L_dec = L_dec_ce + lambda_hdce * (L_hdce / max(len(indices_list), 1))
                else:
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
                plot_loss_curves(loss_history, str(vis_dir / "loss_curves.png"), smooth=min(50, len(loss_history) // 2 or 1))
                save_loss_history(loss_history, str(save_dir / "loss_history.json"))
                with torch.no_grad():
                    stego_vis = flow_module.sample(cover[:4].shape, f_sec[:4], c_txt[:4] if c_txt is not None else None, num_steps=8, device=device)
                    pred_indices = decoder_module.predict_indices(stego_vis)
                    recovered = rq_vae.decode_from_indices(pred_indices)
                from utils.visualization import save_stego_comparison
                save_stego_comparison(cover[:4], stego_vis, secret[:4], recovered, str(vis_dir / ("stego_step%d.png" % global_step)), nrow=4)
            if global_step % save_every == 0:
                if rank == 0:
                    torch.save({"flow": flow_module.state_dict(), "decoder": decoder_module.state_dict(), "rq_vae": rq_vae.state_dict(), "step": global_step}, save_dir / "checkpoint.pt")
                if use_ddp:
                    dist.barrier()

    if rank == 0:
        plot_loss_curves(loss_history, str(vis_dir / "loss_curves_final.png"), smooth=min(50, max(1, len(loss_history) // 2)))
        save_loss_history(loss_history, str(save_dir / "loss_history.json"))
        torch.save({"flow": flow_module.state_dict(), "decoder": decoder_module.state_dict(), "rq_vae": rq_vae.state_dict()}, save_dir / "final.pt")
        print("Training done. Saved to", save_dir)


def main():
    def _excepthook(etype, value, tb):
        import traceback
        traceback.print_exception(etype, value, tb)
        sys.__excepthook__(etype, value, tb)
    sys.excepthook = _excepthook

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--stage", default="all", choices=["1", "2", "all"], help="1=仅RQ-VAE, 2=仅生成器+解码器, all=两阶段")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if not Path(args.config).is_absolute():
        args.config = str(script_dir / args.config)
    if args.override and not Path(args.override).is_absolute():
        args.override = str(script_dir / args.override)
    config = get_config(args.config, args.override)
    proj = config.get("project", {})

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        try:
            torch.cuda.set_device(device)
            _ = torch.zeros(1, device=device)
        except Exception as e:
            print("[Rank %d] CUDA device %d 不可用: %s" % (rank, local_rank, e), flush=True)
            raise
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_main(config, device, rank=rank, world_size=world_size, resume_path=args.resume, stage=args.stage)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
