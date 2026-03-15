import argparse
import math
import os
import random
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from glob import glob

# 必须在导入 diffusers/huggingface_hub 之前设置；默认用官方源，hf-mirror 不可用时无需改
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", os.environ["HF_ENDPOINT"])

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from continuous_vae import ContinuousVAE
from decoder import ResMambaSecretDecoder
from interference import LatentInterference
from models_dis import DiS_models, StegoDiSModel
from tools.dataset import CelebADataset, StratifiedDistributedBatchSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


STEGO_MODEL_CONFIGS = {
    "DiS-S/2": {"patch_size": 2, "embed_dim": 368, "depth": 24},
    "DiS-S/4": {"patch_size": 4, "embed_dim": 368, "depth": 24},
    "DiS-B/2": {"patch_size": 2, "embed_dim": 768, "depth": 24},
    "DiS-B/4": {"patch_size": 4, "embed_dim": 768, "depth": 24},
    "DiS-M/2": {"patch_size": 2, "embed_dim": 768, "depth": 48},
    "DiS-M/4": {"patch_size": 4, "embed_dim": 768, "depth": 48},
    "DiS-L/2": {"patch_size": 2, "embed_dim": 1024, "depth": 48},
    "DiS-L/4": {"patch_size": 4, "embed_dim": 1024, "depth": 48},
    "DiS-H/2": {"patch_size": 2, "embed_dim": 1536, "depth": 48},
    "DiS-H/4": {"patch_size": 4, "embed_dim": 1536, "depth": 48},
}


def configure_hf_endpoint(endpoint: str | None, force: bool = False) -> str | None:
    """Set huggingface endpoint env vars used by different hub versions."""
    if not endpoint:
        return os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_HUB_ENDPOINT")
    if force or "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = endpoint
    if force or "HUGGINGFACE_HUB_ENDPOINT" not in os.environ:
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint
    return os.environ.get("HF_ENDPOINT")

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag



def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def adjust_learning_rate(optimizer, global_step, args):
    """
    学习率策略：
    1. warmup阶段：保持高学习率（生成模型充分学习）
    2. rampup阶段：保持较高学习率（鲁棒性隐写训练起步）
    3. 隐写训练期：保持高学习率充分训练
    4. 最后阶段：缓慢余弦衰减
    """
    warmup_steps = args.warmup_steps
    rampup_steps = args.rampup_steps
    # 估算总训练步数：每个epoch约warmup_steps步
    total_steps = args.epochs * warmup_steps
    # 最后20%训练时间才进行学习率衰减
    decay_start_step = int(total_steps * 0.8)

    if global_step < warmup_steps:
        # warmup阶段：保持完整学习率
        lr = args.lr
    elif global_step < warmup_steps + rampup_steps:
        # rampup阶段：保持较高学习率
        lr = args.lr * 0.9
    elif global_step < decay_start_step:
        # 隐写充分训练期：保持较高学习率
        lr = args.lr * 0.8
    else:
        # 最后20%阶段：余弦衰减到min_lr
        decay_steps = total_steps - decay_start_step
        decay_progress = (global_step - decay_start_step) / max(decay_steps, 1)
        lr = args.min_lr + (args.lr * 0.8 - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * min(decay_progress, 1.0)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def build_stego_model(args, img_size, channels, secret_dim):
    if args.model not in STEGO_MODEL_CONFIGS:
        raise ValueError(f"Unknown stego model config: {args.model}")
    model_kwargs = dict(STEGO_MODEL_CONFIGS[args.model])
    model_kwargs.update(
        img_size=img_size,
        channels=channels,
        num_classes=args.num_classes,
        secret_dim=secret_dim,
        learn_sigma=False,
    )
    return StegoDiSModel(**model_kwargs)


def unwrap_module(module):
    return module.module if isinstance(module, DDP) else module


def maybe_wrap_ddp(module, rank):
    if any(p.requires_grad for p in module.parameters()):
        return DDP(module, device_ids=[rank])
    return module


def get_secret_latent_channels(secret_vae):
    vae_config = getattr(getattr(secret_vae, "vae", None), "config", None)
    latent_channels = getattr(vae_config, "latent_channels", None)
    if latent_channels is None:
        latent_channels = getattr(vae_config, "z_channels", None)
    return int(latent_channels or 4)


def secret_latent_to_seq(z_secret_latent):
    if z_secret_latent.ndim != 4:
        raise ValueError(f"Expected secret latent [B,C,H,W], got {tuple(z_secret_latent.shape)}")
    return z_secret_latent.flatten(2).transpose(1, 2).contiguous()


def align_secret_prediction(z_secret_pred, z_secret):
    if z_secret_pred.shape == z_secret.shape:
        return z_secret_pred

    if z_secret_pred.dim() == 4 and z_secret.dim() == 4 and z_secret_pred.shape[1] == z_secret.shape[1]:
        return F.interpolate(
            z_secret_pred,
            size=z_secret.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    if z_secret_pred.numel() == z_secret.numel():
        return z_secret_pred.reshape_as(z_secret)

    raise ValueError(
        f"Cannot align predicted secret shape {tuple(z_secret_pred.shape)} "
        f"to target shape {tuple(z_secret.shape)}"
    )


def encode_secret(secret_vae, secret_images):
    with torch.no_grad():
        return secret_vae.encode_to_latent(secret_images, sample_posterior=False)


def prepare_decoder_input(secret_vae, z_attacked, latent_space):
    if latent_space:
        return z_attacked
    return secret_vae.encode_to_latent(z_attacked, sample_posterior=False)


def decode_secret(decoder, decoder_input):
    return decoder(decoder_input)


def curriculum_lambda(global_step, target_lambda, warmup_steps=2000, rampup_steps=3000):
    if global_step < warmup_steps:
        return 0.0
    if global_step < warmup_steps + rampup_steps:
        return target_lambda * ((global_step - warmup_steps) / rampup_steps)
    return target_lambda


def main(args): 
    endpoint = configure_hf_endpoint(args.hf_endpoint, force=args.hf_endpoint_force)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()

    if args.global_seed < 0:
        if rank == 0:
            args.global_seed = random.randint(0, 2**31 - 1)
        t = torch.tensor([args.global_seed], dtype=torch.long, device=device)
        dist.broadcast(t, 0)
        args.global_seed = t.item()
    if getattr(args, "split_seed", 42) < 0:
        if rank == 0:
            args.split_seed = random.randint(0, 2**31 - 1)
        t = torch.tensor([args.split_seed], dtype=torch.long, device=device)
        dist.broadcast(t, 0)
        args.split_seed = t.item()

    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    per_gpu_batch = args.global_batch_size // dist.get_world_size()
    assert per_gpu_batch >= 2, (
        f"per-GPU batch size must be >= 2 for meaningful steganography (got {per_gpu_batch}). "
    )
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    print(args)
    if rank == 0 and endpoint:
        print(f"[HF] endpoint={endpoint}")

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True) 
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{model_string_name}-{args.dataset_type}-{args.task_type}-{args.image_size}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

    if args.latent_space == True:
        print('train from latent space!')
        cover_channels = 4
        model_img_size = args.image_size // 8
    else:
        print('train from raw space!')
        cover_channels = 3
        model_img_size = args.image_size

    secret_vae = ContinuousVAE(
        model_name=args.secret_vae_model,
        device=device,
        torch_dtype=torch.float32,
    ).to(device)
    secret_channels = get_secret_latent_channels(secret_vae)

    model = build_stego_model(
        args,
        img_size=model_img_size,
        channels=cover_channels,
        secret_dim=secret_channels,
    )
    decoder = ResMambaSecretDecoder(
        in_channels=secret_channels,
        out_channels=secret_channels,
    ).to(device)
    interference = LatentInterference(
        latent_hw=(model_img_size, model_img_size),
        identity_prob=args.interference_identity_prob,
        noise_std=args.interference_noise_std,
        jpeg_quality=tuple(args.interference_jpeg_quality),
    ).to(device)

    secret_vae.eval()
    requires_grad(secret_vae, False)
    interference.eval()
    requires_grad(interference, False)

    if args.freeze_secret_decoder:
        decoder.eval()
        requires_grad(decoder, False)
    else:
        decoder.train()

    if args.latent_space == True:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
        vae.eval()
        requires_grad(vae, False)
    else:
        vae = None

    checkpoint = None
    global_step = 0
    if args.resume is not None:
        print('resume model')
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model_state = checkpoint.get("model", checkpoint.get("ema", checkpoint))
        model.load_state_dict(model_state, strict=False)
        if "decoder" in checkpoint:
            decoder.load_state_dict(checkpoint["decoder"], strict=False)
        global_step = checkpoint.get("global_step", 0)

    ema = deepcopy(model).to(device)
    if checkpoint is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"], strict=False)
    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[rank])
    decoder = maybe_wrap_ddp(decoder, rank)

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optim_params.extend(p for p in decoder.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=0)
    if checkpoint is not None and "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])
    use_amp = getattr(args, "amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if checkpoint is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    if args.resize_only == False: 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else: 
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
    if args.dataset_type == "cifar-10": 
        dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=False, transform=transform)
    elif args.dataset_type == "imagenet": 
        dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform)
    else:
        dataset = CelebADataset(data_path=args.data_path, transform=transform, split=args.data_split, test_ratio=args.test_ratio, split_seed=args.split_seed)

    per_gpu_batch = int(args.global_batch_size // dist.get_world_size())
    if args.dataset_type == "celeba":
        batch_sampler = StratifiedDistributedBatchSampler(
            dataset, batch_size=per_gpu_batch, num_replicas=dist.get_world_size(),
            rank=rank, shuffle=True, seed=args.global_seed, drop_last=True,
        )
        loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        sampler = batch_sampler
    else:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
        loader = DataLoader(dataset, batch_size=per_gpu_batch, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    update_ema(ema, model.module, decay=0) 
    model.train() 
    ema.eval()
    opt.zero_grad(set_to_none=True)

    running_loss = 0.0
    running_gen_loss = 0.0
    running_secret_loss = 0.0

    for epoch in range(args.epochs): 
        sampler.set_epoch(epoch) 
        running_loss = 0.0
        running_gen_loss = 0.0
        running_secret_loss = 0.0
        train_steps = 0
        with tqdm(enumerate(loader), total=len(loader), disable=rank != 0) as tq:
            for data_iter_step, samples in tq:
                if data_iter_step % args.accum_iter == 0:
                    adjust_learning_rate(opt, global_step, args)
                
                cover = samples[0].to(device)
                perm = torch.randperm(cover.shape[0], device=device)
                secret = cover[perm]
                same = (perm == torch.arange(perm.shape[0], device=device)).all()
                if same:
                    secret = cover.roll(1, dims=0)
                
                if args.num_classes > 0: 
                    labels = samples[1].to(device) 
                else:
                    labels = None 

                if args.latent_space == True:
                    with torch.no_grad():
                        z_cover = vae.encode(cover).latent_dist.sample().mul_(0.18215)
                else:
                    z_cover = cover

                z_secret = encode_secret(secret_vae, secret).to(device=device)
                secret_seq = secret_latent_to_seq(z_secret).to(dtype=z_cover.dtype)

                t = torch.rand(z_cover.shape[0], device=device, dtype=z_cover.dtype)
                t_view = t.view(-1, 1, 1, 1)
                z_noise = torch.randn_like(z_cover)
                z_noisy = (1 - t_view) * z_cover + t_view * z_noise
                v_target = z_noise - z_cover

                current_lambda = curriculum_lambda(
                    global_step,
                    args.target_lambda,
                    warmup_steps=args.warmup_steps,
                    rampup_steps=args.rampup_steps,
                )

                with torch.cuda.amp.autocast(enabled=use_amp):
                    # 1. 生成器推理
                    v_pred = model(z_noisy, t=t, labels=labels, secret_seq=secret_seq)
                    l_gen = F.mse_loss(v_pred, v_target)

                    # 2. 隐写去噪推导
                    z_stego_hat = z_noisy - t_view * v_pred

                    # ================= 核心修改区：保护机制 =================
                    # 设定时间步阈值，保护生成器不受高噪样本隐写梯度的破坏
                    t_threshold = 0.5
                    is_low_noise = (t_view <= t_threshold).float()
                    z_stego_safe = z_stego_hat * is_low_noise + z_stego_hat.detach() * (1.0 - is_low_noise)

                    # 3. 攻击与解码
                    z_attacked = interference(z_stego_safe)
                    decoder_input = prepare_decoder_input(secret_vae, z_attacked, latent_space=args.latent_space)
                    z_secret_pred = decode_secret(decoder, decoder_input)
                    z_secret_pred = align_secret_prediction(z_secret_pred, z_secret)
                    
                    # 4. 计算隐写损失，并引入动态时间步权重 (t=0时权重最大)
                    t_weight = (1.0 - t).view(-1, 1, 1, 1) ** 2
                    l_secret_unweighted = F.mse_loss(z_secret_pred, z_secret, reduction='none')
                    l_secret = (l_secret_unweighted * t_weight).mean()
                    # ========================================================

                    # 5. 总损失
                    loss = l_gen + current_lambda * l_secret
                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    opt.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss / args.accum_iter).backward()
                should_step = (data_iter_step + 1) % args.accum_iter == 0

                if should_step:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    update_ema(ema, model.module)
                    global_step += 1

                    running_loss += loss_value
                    running_gen_loss += l_gen.item()
                    running_secret_loss += l_secret.item()
                    train_steps += 1

                if train_steps > 0:
                    tq.set_description('Epoch %i' % epoch)
                    tq.set_postfix(
                        loss=running_loss / train_steps,
                        l_gen=running_gen_loss / train_steps,
                        l_secret=running_secret_loss / train_steps,
                        lam=current_lambda,
                        step=global_step,
                    )

                if should_step and global_step % args.eval_steps == 0 and global_step > 0:
                    dist.barrier()
                    if rank == 0:
                        with torch.no_grad():
                            eval_n = min(args.eval_batch_size, cover.shape[0])
                            eval_cover = cover[:eval_n]
                            eval_secret = secret[:eval_n]
                            eval_labels = labels[:eval_n] if labels is not None else None
                            if args.latent_space:
                                eval_z_cover = vae.encode(eval_cover).latent_dist.sample().mul_(0.18215)
                            else:
                                eval_z_cover = eval_cover
                            eval_z_secret = encode_secret(secret_vae, eval_secret).to(device=device)
                            eval_secret_seq = secret_latent_to_seq(eval_z_secret).to(dtype=eval_z_cover.dtype)
                            eval_z_stego = torch.randn_like(eval_z_cover)
                            dt = 1.0 / args.eval_num_steps
                            for k in range(args.eval_num_steps):
                                t_val = 1.0 - k / args.eval_num_steps
                                eval_t = torch.full((eval_n,), t_val, device=device, dtype=eval_z_cover.dtype)
                                eval_v_pred = ema(eval_z_stego, t=eval_t, labels=eval_labels, secret_seq=eval_secret_seq)
                                eval_z_stego = eval_z_stego - dt * eval_v_pred
                            if args.latent_space:
                                stego_preview = vae.decode(eval_z_stego / 0.18215).sample
                            else:
                                stego_preview = eval_z_stego
                            preview = torch.cat([eval_cover, eval_secret, stego_preview], dim=0)
                            save_image(preview, os.path.join(experiment_dir, f"sample_{global_step:07d}.png"), nrow=eval_n, normalize=True, value_range=(-1, 1))
                    dist.barrier()

        if rank == 0:
            with torch.no_grad():
                eval_n = min(args.eval_batch_size, cover.shape[0])
                eval_cover = cover[:eval_n]
                eval_secret = secret[:eval_n]
                eval_labels = labels[:eval_n] if labels is not None else None
                if args.latent_space:
                    eval_z_cover = vae.encode(eval_cover).latent_dist.sample().mul_(0.18215)
                else:
                    eval_z_cover = eval_cover
                eval_z_secret = encode_secret(secret_vae, eval_secret).to(device=device)
                eval_secret_seq = secret_latent_to_seq(eval_z_secret).to(dtype=eval_z_cover.dtype)
                eval_z_stego = torch.randn_like(eval_z_cover)
                dt = 1.0 / args.eval_num_steps
                for k in range(args.eval_num_steps):
                    t_val = 1.0 - k / args.eval_num_steps
                    eval_t = torch.full((eval_n,), t_val, device=device, dtype=eval_z_cover.dtype)
                    eval_v_pred = ema(eval_z_stego, t=eval_t, labels=eval_labels, secret_seq=eval_secret_seq)
                    eval_z_stego = eval_z_stego - dt * eval_v_pred
                if args.latent_space:
                    stego_preview = vae.decode(eval_z_stego / 0.18215).sample
                else:
                    stego_preview = eval_z_stego
                preview = torch.cat([eval_cover, eval_secret, stego_preview], dim=0)
                save_image(preview, os.path.join(experiment_dir, f"sample_epoch_{epoch+1:02d}.png"), nrow=eval_n, normalize=True, value_range=(-1, 1))
        dist.barrier()

        ckpt_epochs_0indexed = list(range(args.epochs // 2, args.epochs))
        if epoch in ckpt_epochs_0indexed and rank == 0:
            checkpoint = {
                "model": model.module.state_dict(), "ema": ema.state_dict(),
                "decoder": unwrap_module(decoder).state_dict(), "secret_vae": secret_vae.state_dict(),
                "interference": unwrap_module(interference).state_dict(),
                "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "args": args, "global_step": global_step, "epoch": epoch,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/epoch_{epoch+1:02d}.pt")
        if epoch in ckpt_epochs_0indexed:
            dist.barrier()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=str(Path(__file__).resolve().parent / "data" / "placeholder"))
    parser.add_argument("--data-split", type=str, choices=["train", "test", "all"], default="train")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=-1)
    parser.add_argument("--results-dir", type=str, default="./outputs/run_train_local")
    parser.add_argument("--task-type", type=str, choices=['uncond', 'class-cond', 'text-cond'], default='uncond')
    parser.add_argument("--dataset-type", type=str, choices=['cifar-10', 'imagenet', 'celeba'], default='celeba')
    parser.add_argument("--resize-only", type=bool, default=False)
    parser.add_argument("--num-classes", type=int, default=-1)
    parser.add_argument("--resume", type=str, default=None)
    
    parser.add_argument("--model", type=str, choices=list(DiS_models.keys()), default="DiS-L/2")
    # ✅ 硬件限制对齐参数
    parser.add_argument("--image-size", type=int, choices=[1024, 512, 256, 64, 32], default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=-1) 

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--accum_iter', default=2, type=int)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--eval_steps', default=5000, type=int)
    parser.add_argument('--eval_num_steps', default=10, type=int)

    parser.add_argument('--latent_space', type=bool, default=True) 
    parser.add_argument('--vae_path', type=str, default='stabilityai/sd-vae-ft-mse')
    parser.add_argument('--secret-vae-model', type=str, default='stabilityai/sd-vae-ft-mse')
    parser.add_argument('--secret_dim', type=int, default=None)
    
    parser.add_argument('--target-lambda', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=5300)   # 约 1 个 Epoch 完全预热
    parser.add_argument('--rampup-steps', type=int, default=16000)  # 约 3 个 Epoch 缓慢释放隐写压力
    
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--freeze-secret-decoder', action='store_true')
    parser.add_argument('--hf-endpoint', type=str, default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    parser.add_argument('--hf-endpoint-force', action='store_true', default=True)

    parser.add_argument('--interference-identity-prob', type=float, default=0.5)
    parser.add_argument('--interference-noise-std', type=float, default=0.05)
    parser.add_argument('--interference-jpeg-quality', type=float, nargs=2, default=[40.0, 90.0])

    args = parser.parse_args()
    args.amp = not args.no_amp
    main(args)