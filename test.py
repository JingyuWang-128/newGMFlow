"""
GenMamba-Flow 测试与评估脚本（单脚本跑全部分析，支持多卡）
评估：Bit Accuracy、Recovery PSNR/SSIM、鲁棒性曲线；全部结果可视化并保存。
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from utils.config import get_config
from data.datasets import get_train_dataloader
from models.rq_vae import RQVAE
from models.rectified_flow import RectifiedFlowGenerator
from models.decoder import RobustDecoder
from models.interference import InterferenceManifold
from utils.metrics import compute_psnr_ssim, compute_bit_accuracy
from utils.visualization import save_stego_comparison, save_recovery_grid, save_depth_recovery
from utils.vis_plots import plot_metrics_bars, plot_robustness_curves


def build_models(config, device, use_multi_gpu: bool = False):
    from train import build_models as _build
    rq_vae, flow, decoder, interference = _build(config, device, use_ddp=False)
    if use_multi_gpu and torch.cuda.device_count() > 1:
        flow = nn.DataParallel(flow)
        decoder = nn.DataParallel(decoder)
    return rq_vae, flow, decoder, interference


def load_checkpoint(config, device, ckpt_path: str, use_multi_gpu: bool = False):
    rq_vae, flow, decoder, _ = build_models(config, device, use_multi_gpu=use_multi_gpu)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "rq_vae" in ckpt:
        rq_vae.load_state_dict(ckpt["rq_vae"])
    if "flow" in ckpt:
        (flow.module if isinstance(flow, nn.DataParallel) else flow).load_state_dict(ckpt["flow"])
    if "decoder" in ckpt:
        (decoder.module if isinstance(decoder, nn.DataParallel) else decoder).load_state_dict(ckpt["decoder"])
    return rq_vae, flow, decoder


def _flow_sample(flow, shape, f_sec, c_txt, num_steps, device):
    m = flow.module if isinstance(flow, nn.DataParallel) else flow
    return m.sample(shape, f_sec, c_txt, num_steps=num_steps, device=device)


def _decoder_predict_indices(decoder, x):
    m = decoder.module if isinstance(decoder, nn.DataParallel) else decoder
    return m.predict_indices(x)


def run_eval(config, device, ckpt_path: str, output_dir: str, num_batches: int = 20, use_multi_gpu: bool = False):
    rq_vae, flow, decoder = load_checkpoint(config, device, ckpt_path, use_multi_gpu)
    rq_vae.eval()
    flow.eval()
    decoder.eval()
    dataloader = get_train_dataloader(config, rank=0, world_size=1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ba = []
    all_psnr = []
    all_ssim = []
    from train import get_text_encoder
    text_encoder, _ = get_text_encoder(config, device)
    num_steps_sample = config.get("generator", {}).get("num_flow_steps", 1000)
    num_steps_sample = min(32, num_steps_sample)

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
            stego = _flow_sample(flow, cover.shape, f_sec, c_txt, num_steps_sample, device)
            pred_indices = _decoder_predict_indices(decoder, stego)
            recovered = rq_vae.decode_from_indices(pred_indices)
        ba = compute_bit_accuracy(pred_indices, indices_list)
        psnr, ssim = compute_psnr_ssim(recovered, secret)
        all_ba.append(ba)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

        if bi < 5:
            save_stego_comparison(
                cover, stego, secret, recovered,
                str(output_dir / f"compare_batch{bi}.png"),
                nrow=min(4, cover.shape[0]),
            )
            save_recovery_grid(secret, recovered, str(output_dir / f"recovery_batch{bi}.png"), nrow=4)
            save_depth_recovery(rq_vae, pred_indices, str(output_dir / f"depth_recovery_batch{bi}.png"), depth_steps=[1, 2, 3, 4])

    metrics = {
        "Bit Accuracy": float(np.mean(all_ba)),
        "Recovery PSNR": float(np.mean(all_psnr)),
        "Recovery SSIM": float(np.mean(all_ssim)),
    }
    print("Bit Accuracy:", metrics["Bit Accuracy"])
    print("Recovery PSNR:", metrics["Recovery PSNR"])
    print("Recovery SSIM:", metrics["Recovery SSIM"])
    with open(output_dir / "metrics.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plot_metrics_bars(metrics, str(output_dir / "metrics_bars.png"), title="Evaluation Metrics")
    return metrics


def run_robustness_curves(config, device, ckpt_path: str, output_dir: str, use_multi_gpu: bool = False):
    rq_vae, flow, decoder = load_checkpoint(config, device, ckpt_path, use_multi_gpu)
    rq_vae.eval()
    flow.eval()
    decoder.eval()
    interference = InterferenceManifold(config.get("interference", {}))
    dataloader = get_train_dataloader(config, rank=0, world_size=1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jpeg_qualities = [30, 50, 70, 90]
    results = {str(q): [] for q in jpeg_qualities}
    from train import get_text_encoder
    text_encoder, _ = get_text_encoder(config, device)

    for batch in tqdm(list(dataloader)[:15], desc="Robustness"):
        cover = batch["cover"].to(device)
        secret = batch["secret"].to(device)
        texts = batch.get("text", ["a natural image"] * cover.shape[0])
        with torch.no_grad():
            indices_list = rq_vae.get_indices(secret)
            f_sec = rq_vae.encode(secret)
            _, _, qlist, _ = rq_vae.quantize_residual(f_sec)
            f_sec = sum(qlist)
            c_txt = text_encoder(texts)
            stego = _flow_sample(flow, cover.shape, f_sec, c_txt, 16, device)
        for q in jpeg_qualities:
            stego_jpeg = interference.jpeg_op(stego, quality=q)
            if stego_jpeg.device != device:
                stego_jpeg = stego_jpeg.to(device)
            with torch.no_grad():
                pred = _decoder_predict_indices(decoder, stego_jpeg)
            ba = compute_bit_accuracy(pred, indices_list)
            results[str(q)].append(ba)
    mean_results = {q: float(np.mean(results[str(q)])) for q in jpeg_qualities}
    for q in jpeg_qualities:
        print(f"JPEG Q{q}: BA = {mean_results[q]}")
    with open(output_dir / "robustness_jpeg.txt", "w", encoding="utf-8") as f:
        for q in jpeg_qualities:
            f.write(f"JPEG Q{q}: {mean_results[q]}\n")
    with open(output_dir / "robustness_jpeg.json", "w", encoding="utf-8") as f:
        json.dump(mean_results, f, indent=2)
    plot_robustness_curves(
        {"GenMamba-Flow": [mean_results[q] for q in jpeg_qualities]},
        [str(q) for q in jpeg_qualities],
        str(output_dir / "robustness_jpeg_curve.png"),
        ylabel="Bit Accuracy",
        title="Robustness under JPEG Compression",
    )
    return mean_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--checkpoint", default="outputs/final.pt")
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--multi_gpu", action="store_true", help="Use all available GPUs (DataParallel)")
    parser.add_argument("--robustness_only", action="store_true")
    args = parser.parse_args()
    config = get_config(args.config, args.override)
    device = torch.device(args.device)
    use_multi_gpu = args.multi_gpu and torch.cuda.device_count() > 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.robustness_only:
        run_robustness_curves(config, device, args.checkpoint, args.output_dir, use_multi_gpu)
    else:
        metrics = run_eval(config, device, args.checkpoint, args.output_dir, args.num_batches, use_multi_gpu)
        robustness = run_robustness_curves(config, device, args.checkpoint, args.output_dir, use_multi_gpu)
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "robustness_jpeg": robustness}, f, indent=2)
        print("All results and figures saved to", output_dir)


if __name__ == "__main__":
    main()
