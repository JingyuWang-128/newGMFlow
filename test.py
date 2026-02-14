"""
GenMamba-Flow 测试与评估脚本
评估：保真性(FID/CLIP/LPIPS)、隐写性能(Bit Accuracy, Recovery PSNR/SSIM)、鲁棒性曲线
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from utils.config import get_config
from data.datasets import get_train_dataloader, get_secret_dataloader
from models.rq_vae import RQVAE
from models.tri_stream_mamba import TriStreamMambaUNet
from models.rectified_flow import RectifiedFlowGenerator
from models.decoder import RobustDecoder
from models.interference import InterferenceManifold
from utils.metrics import compute_psnr_ssim, compute_bit_accuracy
from utils.visualization import save_stego_comparison, save_recovery_grid, save_depth_recovery


def build_models(config, device):
    from train import build_models as _build
    return _build(config, device)


def load_checkpoint(config, device, ckpt_path: str):
    rq_vae, flow, decoder, interference = build_models(config, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "rq_vae" in ckpt:
        rq_vae.load_state_dict(ckpt["rq_vae"])
    if "flow" in ckpt:
        flow.load_state_dict(ckpt["flow"])
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
    return rq_vae, flow, decoder


def run_eval(config, device, ckpt_path: str, output_dir: str, num_batches: int = 20):
    rq_vae, flow, decoder = load_checkpoint(config, device, ckpt_path)
    rq_vae.eval()
    flow.eval()
    decoder.eval()
    dataloader = get_train_dataloader(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ba = []
    all_psnr = []
    all_ssim = []
    from train import get_text_encoder
    text_encoder, _ = get_text_encoder(config, device)
    num_steps = config.get("generator", {}).get("num_flow_steps", 1000)
    num_steps_sample = 16

    for bi, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Eval")):
        if bi >= num_batches:
            break
        cover = batch["cover"].to(device)
        secret = batch["secret"].to(device)
        texts = batch.get("text", ["a natural image"] * cover.shape[0])
        if isinstance(texts, str):
            texts = [texts] * cover.shape[0]
        with torch.no_grad():
            indices_list = rq_vae.get_indices(secret)
            f_sec = rq_vae.encode(secret)
            _, _, qlist, _ = rq_vae.quantize_residual(f_sec)
            f_sec = sum(qlist)
            c_txt = text_encoder(texts)
            if c_txt.dim() == 1:
                c_txt = c_txt.unsqueeze(0)
            stego = flow.sample(cover.shape, f_sec, c_txt, num_steps=num_steps_sample, device=device)
            pred_indices = decoder.predict_indices(stego)
            recovered = rq_vae.decode_from_indices(pred_indices)
        ba = compute_bit_accuracy(pred_indices, indices_list)
        psnr, ssim = compute_psnr_ssim(recovered, secret)
        all_ba.append(ba)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

        if bi < 3:
            save_stego_comparison(
                cover, stego, secret, recovered,
                str(output_dir / f"compare_batch{bi}.png"),
                nrow=min(4, cover.shape[0]),
            )
            save_depth_recovery(rq_vae, pred_indices, str(output_dir / f"depth_recovery_batch{bi}.png"))

    print("Bit Accuracy:", np.mean(all_ba))
    print("Recovery PSNR:", np.mean(all_psnr))
    print("Recovery SSIM:", np.mean(all_ssim))
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"Bit Accuracy: {np.mean(all_ba)}\n")
        f.write(f"Recovery PSNR: {np.mean(all_psnr)}\n")
        f.write(f"Recovery SSIM: {np.mean(all_ssim)}\n")
    return {"ba": np.mean(all_ba), "psnr": np.mean(all_psnr), "ssim": np.mean(all_ssim)}


def run_robustness_curves(config, device, ckpt_path: str, output_dir: str):
    """在不同 JPEG 质量下计算 Bit Accuracy 曲线。"""
    rq_vae, flow, decoder = load_checkpoint(config, device, ckpt_path)
    rq_vae.eval()
    flow.eval()
    decoder.eval()
    interference = InterferenceManifold(config.get("interference", {}))
    dataloader = get_train_dataloader(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jpeg_qualities = [30, 50, 70, 90]
    results = {q: [] for q in jpeg_qualities}
    from train import get_text_encoder
    text_encoder, _ = get_text_encoder(config, device)

    for batch in tqdm(list(dataloader)[:10], desc="Robustness"):
        cover = batch["cover"].to(device)
        secret = batch["secret"].to(device)
        texts = batch.get("text", ["a natural image"] * cover.shape[0])
        with torch.no_grad():
            indices_list = rq_vae.get_indices(secret)
            f_sec = rq_vae.encode(secret)
            _, _, qlist, _ = rq_vae.quantize_residual(f_sec)
            f_sec = sum(qlist)
            c_txt = text_encoder(texts) if callable(text_encoder) else text_encoder(texts)
            stego = flow.sample(cover.shape, f_sec, c_txt, num_steps=16, device=device)
        for q in jpeg_qualities:
            stego_jpeg = interference.jpeg_op(stego, quality=q)
            if stego_jpeg.device != device:
                stego_jpeg = stego_jpeg.to(device)
            with torch.no_grad():
                pred = decoder.predict_indices(stego_jpeg)
            ba = compute_bit_accuracy(pred, indices_list)
            results[q].append(ba)
    for q in jpeg_qualities:
        print(f"JPEG Q{q}: BA = {np.mean(results[q])}")
    with open(output_dir / "robustness_jpeg.txt", "w") as f:
        for q in jpeg_qualities:
            f.write(f"JPEG Q{q}: {np.mean(results[q])}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--checkpoint", default="outputs/final.pt")
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--robustness_only", action="store_true")
    args = parser.parse_args()
    config = get_config(args.config, args.override)
    device = torch.device(args.device)
    if args.robustness_only:
        run_robustness_curves(config, device, args.checkpoint, args.output_dir)
    else:
        run_eval(config, device, args.checkpoint, args.output_dir, args.num_batches)
