import torch 
import argparse 
import random 
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from diffusion import create_diffusion
from models_dis import DiS_models 


def configure_hf_endpoint(endpoint: str | None, force: bool = False) -> str | None:
    if not endpoint:
        return os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_HUB_ENDPOINT")
    if force or "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = endpoint
    if force or "HUGGINGFACE_HUB_ENDPOINT" not in os.environ:
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint
    return os.environ.get("HF_ENDPOINT")


def main(args):
    print("Sample images from a trained DiS.")
    endpoint = configure_hf_endpoint(args.hf_endpoint, force=args.hf_endpoint_force)
    if endpoint:
        print(f"[HF] endpoint={endpoint}")
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False) 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.latent_space == True: 
        model = DiS_models[args.model](
            img_size=args.image_size // 8,
            num_classes=args.num_classes,
            channels=4,
        ) 
    else:
        model = DiS_models[args.model](
            img_size=args.image_size,
            num_classes=args.num_classes,
            channels=3,
        ) 

    checkponit = torch.load(args.ckpt, map_location=lambda storage, loc: storage)['ema'] 
    model.load_state_dict(checkponit) 
    model = model.to(device)
    model.eval() 
    diffusion = create_diffusion(str(args.num_sampling_steps)) 
    if args.latent_space == True: 
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    
    n = 8
    if args.num_classes > 0: 
        class_labels=[]
        for i in range(n):
            class_labels.append(random.randint(0, args.num_classes - 1))
            y = torch.tensor(class_labels, device=device)
            y_null = torch.tensor([args.num_classes] * n, device=device)
    
            y = torch.cat([y, y_null], 0)
    
    if args.latent_space == True: 
        z = torch.randn(n, 4, args.image_size//8, args.image_size//8, device=device)
    else:
        z = torch.randn(n, 3, args.image_size, args.image_size, device=device)
    
    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    
    if args.num_classes > 0: 
        labels = y
    else:
        labels = None

    model_kwargs = dict(labels=labels,)
    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    eval_samples, _ = samples.chunk(2, dim=0) 
    
    if args.latent_space == True: 
        eval_samples = vae.decode(eval_samples / 0.18215).sample
    
    save_image(eval_samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))



if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str, choices=list(DiS_models.keys()), default="DiS-H/2")
    parser.add_argument("--image-size", type=int, choices=[32, 64, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5) 
    parser.add_argument("--num-sampling-steps", type=int, default=250) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="/TrainData/Multimodal/zhengcong.fei/dis/results/DiS-H-2-imagenet-class-cond-256/checkpoints/ckpt.pt",) 
    
    parser.add_argument('--latent_space', type=bool, default=True,) 
    parser.add_argument('--vae_path', type=str, default='stabilityai/sd-vae-ft-mse',
                        help='VAE 路径：HF 模型 ID 或本地含 config.json 的文件夹')
    parser.add_argument('--hf-endpoint', type=str, default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'),
                        help='HuggingFace 下载镜像，hf-mirror 不可用时用 https://huggingface.co')
    parser.add_argument('--hf-endpoint-force', action='store_true')
    args = parser.parse_args()

    main(args)