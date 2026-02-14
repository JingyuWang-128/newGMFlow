#!/usr/bin/env python3
"""
生成隐写对比与层级恢复可视化图。
用法: python scripts/visualize.py --checkpoint outputs/final.pt --output_dir outputs/figures --num_samples 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import get_config
from train import build_models, get_text_encoder
from data.datasets import get_train_dataloader
import torch
from tqdm import tqdm
from utils.visualization import save_stego_comparison, save_depth_recovery, save_recovery_grid


def load_ckpt(config, device, ckpt_path):
    rq_vae, flow, decoder, _ = build_models(config, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "rq_vae" in ckpt:
        rq_vae.load_state_dict(ckpt["rq_vae"])
    if "flow" in ckpt:
        flow.load_state_dict(ckpt["flow"])
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
    return rq_vae, flow, decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/final.pt")
    parser.add_argument("--output_dir", default="outputs/figures")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = get_config(args.config)
    device = torch.device(args.device)
    rq_vae, flow, decoder = load_ckpt(config, device, args.checkpoint)
    rq_vae.eval()
    flow.eval()
    decoder.eval()
    text_encoder, _ = get_text_encoder(config, device)
    dataloader = get_train_dataloader(config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_steps = config.get("generator", {}).get("num_flow_steps", 1000)
    num_steps_sample = min(32, num_steps)

    for bi, batch in enumerate(tqdm(dataloader, total=min(3, args.num_samples))):
        if bi >= args.num_samples:
            break
        cover = batch["cover"].to(device)
        secret = batch["secret"].to(device)
        texts = batch.get("text", ["a natural image"] * cover.shape[0])
        with torch.no_grad():
            indices_list = rq_vae.get_indices(secret)
            f_sec = rq_vae.encode(secret)
            _, _, qlist, _ = rq_vae.quantize_residual(f_sec)
            f_sec = sum(qlist)
            c_txt = text_encoder(texts)
            stego = flow.sample(cover.shape, f_sec, c_txt, num_steps=num_steps_sample, device=device)
            pred_indices = decoder.predict_indices(stego)
            recovered = rq_vae.decode_from_indices(pred_indices)
        save_stego_comparison(cover, stego, secret, recovered, str(out_dir / f"compare_{bi}.png"), nrow=4)
        save_recovery_grid(secret, recovered, str(out_dir / f"recovery_{bi}.png"), nrow=4)
        save_depth_recovery(rq_vae, pred_indices, str(out_dir / f"depth_{bi}.png"), depth_steps=[1, 2, 3, 4])
    print("Figures saved to", out_dir)


if __name__ == "__main__":
    main()
