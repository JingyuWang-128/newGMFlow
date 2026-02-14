"""
训练/测试/消融的可视化：损失曲线、指标柱状图、鲁棒性曲线、对比图说明
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_loss_curves(
    history: List[Dict[str, float]],
    save_path: str,
    keys: Optional[List[str]] = None,
    smooth: int = 1,
):
    """绘制损失曲线。history: [{"step": i, "loss_flow": ..., "loss_align": ...}, ...]"""
    plt = _ensure_matplotlib()
    if not history:
        return
    keys = keys or [k for k in history[0].keys() if k != "step"]
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, key in enumerate(keys[:4]):
        ax = axes[i]
        vals = [h.get(key, 0) for h in history]
        if smooth > 1:
            vals = np.convolve(vals, np.ones(smooth) / smooth, mode="valid")
            steps_plot = steps[smooth - 1:]
        else:
            steps_plot = steps
        ax.plot(steps_plot, vals)
        ax.set_title(key)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_metrics_bars(
    metrics: Dict[str, float],
    save_path: str,
    title: str = "Evaluation Metrics",
):
    """指标柱状图。metrics: {"Bit Accuracy": 0.95, "PSNR": 28.5, ...}"""
    plt = _ensure_matplotlib()
    names = list(metrics.keys())
    values = list(metrics.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color="steelblue", edgecolor="black")
    ax.set_ylabel("Value")
    ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_robustness_curves(
    results: Dict[str, List[float]],
    x_labels: List[str],
    save_path: str,
    ylabel: str = "Bit Accuracy",
    title: str = "Robustness under JPEG Compression",
):
    """鲁棒性曲线。results: {"GenMamba-Flow": [0.9, 0.85, ...], "Baseline": [...]}"""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, vals in results.items():
        ax.plot(x_labels[: len(vals)], vals, marker="o", label=name)
    ax.set_xlabel("JPEG Quality")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_ablation_comparison(
    results: List[Dict[str, Any]],
    metric_keys: List[str],
    save_path: str,
    method_key: str = "method",
):
    """消融对比：多方法 x 多指标 柱状图。results: [{"method": "Full", "BA": 0.9, "PSNR": 28}, ...]"""
    plt = _ensure_matplotlib()
    methods = [r[method_key] for r in results]
    x = np.arange(len(methods))
    width = 0.8 / len(metric_keys)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(metric_keys):
        vals = [r.get(key, 0) for r in results]
        offset = (i - len(metric_keys) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylabel("Value")
    ax.set_title("Ablation Study")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_loss_history(history: List[Dict[str, float]], path: str):
    """将损失历史保存为 JSON，便于后续重绘。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_loss_history(path: str) -> List[Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
