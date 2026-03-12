import argparse
import math
import os
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
from tools.dataset import CelebADataset

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
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
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
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
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


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
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
    # Setup DDP
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    per_gpu_batch = args.global_batch_size // dist.get_world_size()
    assert per_gpu_batch >= 2, (
        f"per-GPU batch size must be >= 2 for meaningful steganography (got {per_gpu_batch}). "
        "When batch=1, secret=cover[randperm(1)] yields secret==cover always."
    )
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    print(args)
    if rank == 0 and endpoint:
        print(f"[HF] endpoint={endpoint}")

    # Setup an experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True) 
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}-{args.dataset_type}-{args.task_type}-{args.image_size}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
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
    if args.secret_dim is not None and args.secret_dim != secret_channels:
        raise ValueError(
            f"`--secret_dim` ({args.secret_dim}) must match ContinuousVAE latent channels ({secret_channels})."
        )

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
        if "secret_vae" in checkpoint:
            secret_vae.load_state_dict(checkpoint["secret_vae"], strict=False)
        if "interference" in checkpoint:
            interference.load_state_dict(checkpoint["interference"], strict=False)
        global_step = checkpoint.get("global_step", 0)

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
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
    print('lr: ', args.lr)

    # Setup data
    if args.resize_only == False: 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else: 
        # for model image size << data image size 
        print('image size << data image size')
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    

    if args.dataset_type == "cifar-10": 
        dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=False,
            transform=transform,
        )
    elif args.dataset_type == "imagenet": 
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'train'),
            transform=transform,
        )
    else:
        dataset = CelebADataset(
            data_path=args.data_path,
            transform=transform,
            split=args.data_split,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
        )
        if rank == 0:
            print(
                f"[Data] mixed split={args.data_split}, size={len(dataset)}, "
                f"test_ratio={args.test_ratio}, seed={args.split_seed}"
            )


    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    update_ema(ema, model.module, decay=0) 
    model.train() 
    ema.eval()
    opt.zero_grad(set_to_none=True)

    # Variables for monitoring/logging purposes
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
                # we use a per iteration (instead of per epoch) lr scheduler
                if data_iter_step % args.accum_iter == 0:
                    adjust_learning_rate(opt, data_iter_step / len(loader) + epoch, args)
                
                cover = samples[0].to(device)
                perm = torch.randperm(cover.shape[0], device=device)
                secret = cover[perm]
                # 保证 cover 与 secret 不同：randperm 有概率为恒等排列（尤其 batch=2 时约 50%）导致所有 pair 相同
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

                z_secret = encode_secret(secret_vae, secret).to(
                    device=device,
                )
                secret_seq = secret_latent_to_seq(z_secret).to(dtype=z_cover.dtype)

                # Rectified Flow
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

                # 1. 生成器推理 (必须传入 t 和 secret_seq)
                v_pred = model(z_noisy, t=t, labels=labels, secret_seq=secret_seq)
                l_gen = F.mse_loss(v_pred, v_target)

                # 2. 隐写去噪推导
                z_stego_hat = z_noisy - t_view * v_pred

                # 3. 攻击与解码
                z_attacked = interference(z_stego_hat)
                decoder_input = prepare_decoder_input(secret_vae, z_attacked, latent_space=args.latent_space)
                z_secret_pred = decode_secret(decoder, decoder_input)
                z_secret_pred = align_secret_prediction(z_secret_pred, z_secret)
                l_secret = F.mse_loss(z_secret_pred, z_secret)

                # 4. 总损失
                loss = l_gen + current_lambda * l_secret
                loss_value = loss.item()

                if not math.isfinite(loss_value): 
                    opt.zero_grad(set_to_none=True)
                    continue
                
                (loss / args.accum_iter).backward()
                should_step = (data_iter_step + 1) % args.accum_iter == 0

                if should_step:
                    torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
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

                # Sample images: 每 eval_steps (默认5000) 步保存一次
                if should_step and global_step % args.eval_steps == 0 and global_step > 0:
                    dist.barrier()
                    if rank == 0:
                        with torch.no_grad():
                            eval_n = min(args.eval_batch_size, cover.shape[0])
                            eval_cover = cover[:eval_n]
                            eval_secret = secret[:eval_n]
                            eval_labels = labels[:eval_n] if labels is not None else None

                            if args.latent_space == True:
                                eval_z_cover = vae.encode(eval_cover).latent_dist.sample().mul_(0.18215)
                            else:
                                eval_z_cover = eval_cover

                            eval_z_secret = encode_secret(secret_vae, eval_secret).to(
                                device=device,
                            )
                            eval_secret_seq = secret_latent_to_seq(eval_z_secret).to(dtype=eval_z_cover.dtype)
                            # 多步 ODE 采样：从 t=1(纯噪声) 积分到 t=0，提升生成质量
                            eval_z_stego = torch.randn_like(eval_z_cover)
                            dt = 1.0 / args.eval_num_steps
                            for k in range(args.eval_num_steps):
                                t_val = 1.0 - k / args.eval_num_steps
                                eval_t = torch.full((eval_n,), t_val, device=device, dtype=eval_z_cover.dtype)
                                eval_v_pred = ema(eval_z_stego, t=eval_t, labels=eval_labels, secret_seq=eval_secret_seq)
                                eval_z_stego = eval_z_stego - dt * eval_v_pred

                            if args.latent_space == True:
                                stego_preview = vae.decode(eval_z_stego / 0.18215).sample
                            else:
                                stego_preview = eval_z_stego

                            preview = torch.cat([eval_cover, eval_secret, stego_preview], dim=0)
                            save_image(
                                preview,
                                os.path.join(experiment_dir, f"sample_{global_step:07d}.png"),
                                nrow=eval_n,
                                normalize=True,
                                value_range=(-1, 1),
                            )
                        
                    dist.barrier()

        # Checkpoint: 仅保存两个——warmup 结束后、整个训练结束后
        ckpt_epochs_0indexed = sorted(set([
            args.warmup_epochs - 1,   # warmup 结束
            args.epochs - 1,          # 训练全部结束
        ]))
        if epoch in ckpt_epochs_0indexed and rank == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "decoder": unwrap_module(decoder).state_dict(),
                "secret_vae": secret_vae.state_dict(),
                "interference": unwrap_module(interference).state_dict(),
                "opt": opt.state_dict(),
                "args": args,
                "global_step": global_step,
                "epoch": epoch,
            }
            checkpoint_path = f"{checkpoint_dir}/epoch_{epoch+1:02d}.pt"
            torch.save(checkpoint, checkpoint_path)
        if epoch in ckpt_epochs_0indexed:
            dist.barrier()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=str(Path(__file__).resolve().parent / "data" / "placeholder"))
    parser.add_argument("--data-split", type=str, choices=["train", "test", "all"], default="train")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="/home/wangjingyu/newGMFlow/DiS-main/outputs/run_train_local")
    parser.add_argument("--task-type", type=str, choices=['uncond', 'class-cond', 'text-cond'], default='uncond')
    parser.add_argument("--dataset-type", type=str, choices=['cifar-10', 'imagenet', 'celeba'], default='celeba')
    parser.add_argument("--resize-only", type=bool, default=False)
    parser.add_argument("--num-classes", type=int, default=-1)
    parser.add_argument("--resume", type=str, default=None)
    
    parser.add_argument("--model", type=str, choices=list(DiS_models.keys()), default="DiS-L/2")
    parser.add_argument("--image-size", type=int, choices=[1024, 512, 256, 64, 32], default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=420) 

    parser.add_argument('--lr', type=float, default=5e-5, help='学习率，稍低有利于 l_gen 收敛') 
    parser.add_argument('--min_lr', type=float, default=1e-6,)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--accum_iter', default=1, type=int,) 
    parser.add_argument('--eval_steps', default=5000, type=int, help='每多少步保存一次 sample 图片')
    parser.add_argument('--eval_num_steps', default=10, type=int, help='sample 生成时的 ODE 积分步数，多步可提升清晰度')

    parser.add_argument('--latent_space', type=bool, default=True) 
    parser.add_argument('--vae_path', type=str, default='stabilityai/sd-vae-ft-mse',
                        help='VAE 路径：HuggingFace 模型 ID (如 stabilityai/sd-vae-ft-mse) 或本地含 config.json 的文件夹。'
                             '若网络不通，可先运行: huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir ./vae ，再用 --vae_path ./vae')
    parser.add_argument('--secret-vae-model', type=str, default='stabilityai/sd-vae-ft-mse')
    parser.add_argument('--secret_dim', type=int, default=None)
    parser.add_argument('--target-lambda', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--rampup-steps', type=int, default=3000)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--freeze-secret-decoder', action='store_true')
    parser.add_argument('--hf-endpoint', type=str,
                        default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'),
                        help='HuggingFace 下载镜像。hf-mirror 不可用时请用 https://huggingface.co ，国内无代理时可试 https://hf-mirror.com')
    parser.add_argument('--hf-endpoint-force', action='store_true', default=True)
    args = parser.parse_args()
    main(args)
