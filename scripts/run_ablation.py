#!/usr/bin/env python3
"""
消融实验：单脚本依次训练/评估多种配置，并生成对比可视化与表格。
用法:
  仅评估（已有各变体 checkpoint）: python scripts/run_ablation.py --mode eval --configs configs/ablation/*.yaml
  训练+评估: python scripts/run_ablation.py --mode full --output_base outputs/ablation
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_cmd(cmd: list, cwd: str = None):
    cwd = cwd or str(ROOT)
    return subprocess.run(cmd, cwd=cwd, env={**__import__("os").environ, "PYTHONPATH": ROOT})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "eval"], default="eval",
                        help="full: 训练各变体再评估; eval: 仅对已有 checkpoint 评估")
    parser.add_argument("--output_base", default="outputs/ablation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint_dir", default="outputs",
                        help="eval 模式下各变体 checkpoint 所在目录，或包含子目录的根目录")
    parser.add_argument("--variants", nargs="+",
                        default=["full", "no_align", "no_robust"],
                        help="变体名: full, no_align, no_robust 等，对应不同 override 或 checkpoint 子路径")
    parser.add_argument("--num_batches", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # 变体配置：eval 时 checkpoint 路径为 checkpoint_dir / name / final.pt 或 checkpoint_dir / final.pt (full)
    variant_specs = {
        "full": {"override": None},
        "no_align": {"override": "configs/ablation_no_align.yaml"},
        "no_robust": {"override": "configs/ablation_no_robust.yaml"},
    }
    for v in args.variants:
        if v not in variant_specs:
            variant_specs[v] = {"override": None}

    results = []
    checkpoint_dir = Path(args.checkpoint_dir)

    for name in args.variants:
        spec = variant_specs.get(name, {})
        ckpt_path = checkpoint_dir / name / "final.pt"
        if not ckpt_path.exists() and name == "full":
            ckpt_path = checkpoint_dir / "final.pt"
        out_dir = output_base / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "full":
            override = spec.get("override")
            cmd = [sys.executable, "train.py", "--config", args.config]
            if override:
                cmd += ["--override", override]
            run_cmd(cmd)
            import shutil
            for f in ["final.pt", "loss_history.json"]:
                src = ROOT / "outputs" / f
                if src.exists():
                    (out_dir / f).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, out_dir / f)
            ckpt_path = out_dir / "final.pt"
            if not ckpt_path.exists():
                ckpt_path = ROOT / "outputs" / "final.pt"

        if not ckpt_path.exists():
            print(f"Skip {name}: checkpoint not found at {ckpt_path}")
            continue

        cmd = [
            sys.executable, "test.py",
            "--config", args.config,
            "--checkpoint", str(ckpt_path),
            "--output_dir", str(out_dir),
            "--num_batches", str(args.num_batches),
            "--multi_gpu",
        ]
        run_cmd(cmd)
        metrics_file = out_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                m = json.load(f)
            m["method"] = name
            results.append(m)

    if not results:
        print("No results to plot.")
        return

    from utils.vis_plots import plot_ablation_comparison
    metric_keys = ["Bit Accuracy", "Recovery PSNR", "Recovery SSIM"]
    plot_ablation_comparison(results, metric_keys, str(output_base / "ablation_comparison.png"), method_key="method")
    with open(output_base / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Ablation results and figure saved to", output_base)


if __name__ == "__main__":
    main()
