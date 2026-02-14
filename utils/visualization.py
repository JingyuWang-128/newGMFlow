"""
可视化：隐写对比、层级恢复、鲁棒性曲线等
"""

from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from torchvision.utils import make_grid, save_image


def tensor_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in [-1,1] -> (B,3,H,W) uint8 [0,255]."""
    x = (x.clamp(-1, 1) + 1) / 2
    return (x * 255).byte()


def save_recovery_grid(
    secret: torch.Tensor,
    recovered: torch.Tensor,
    path: str,
    nrow: int = 4,
):
    """secret, recovered: (B,3,H,W). 保存 [secret | recovered] 网格图。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    s = tensor_to_uint8(secret).float() / 255
    r = tensor_to_uint8(recovered).float() / 255
    grid = torch.cat([s, r], dim=0)
    save_image(grid, path, nrow=nrow, padding=2)


def save_stego_comparison(
    cover: torch.Tensor,
    stego: torch.Tensor,
    secret: torch.Tensor,
    recovered: torch.Tensor,
    path: str,
    nrow: int = 4,
):
    """cover/stego/secret/recovered (B,3,H,W). 保存四宫格对比。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    to_img = lambda t: tensor_to_uint8(t).float() / 255
    rows = []
    for i in range(cover.shape[0]):
        row = torch.cat([to_img(cover[i:i+1]), to_img(stego[i:i+1]), to_img(secret[i:i+1]), to_img(recovered[i:i+1])], dim=0)
        rows.append(row)
    grid = torch.cat(rows, dim=0)
    save_image(grid, path, nrow=4, padding=2)


def save_depth_recovery(
    rq_vae,
    indices_list: List[torch.Tensor],
    path: str,
    depth_steps: Optional[List[int]] = None,
):
    """按深度逐步恢复并保存多张图。depth_steps e.g. [1,2,3,4] 表示用前 1/2/3/4 层恢复。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    depth_steps = depth_steps or list(range(1, len(indices_list) + 1))
    imgs = []
    for d in depth_steps:
        partial = rq_vae.decode_from_indices_partial(indices_list, depth_to=d)
        imgs.append(tensor_to_uint8(partial).float() / 255)
    grid = torch.cat(imgs, dim=0)
    save_image(grid, path, nrow=len(depth_steps), padding=2)
