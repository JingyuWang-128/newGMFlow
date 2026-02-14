"""
Rectified Flow 生成器
x_t = t*x_1 + (1-t)*x_0, 学习速度场 v_θ(x_t, t) ≈ x_1 - x_0，单步估计 x̂_0 = x_t - (1-t)*v_θ.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import math

from .tri_stream_mamba import TriStreamMambaUNet


class RectifiedFlowGenerator(nn.Module):
    """
    基于三流 Mamba U-Net 的 Rectified Flow。
    前向：给定 x_0（噪声）, x_1（目标图）, t，计算 x_t 与目标速度，用 v_θ 预测并算损失。
    采样：从 x_1 ~ N(0,1) 沿 ODE 积分到 t=0 得到 x_0。
    """

    def __init__(self, unet: TriStreamMambaUNet, num_steps: int = 1000):
        super().__init__()
        self.unet = unet
        self.num_steps = num_steps

    def get_x_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t = t * x_1 + (1-t) * x_0. t 标量或 (B,) 在 [0,1]。"""
        if t.dim() == 0:
            t = t.view(1).expand(x_0.shape[0])
        t = t.to(x_0.device).float().view(-1, 1, 1, 1)
        return t * x_1 + (1 - t) * x_0

    def get_velocity_target(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Rectified Flow 目标速度: v = x_1 - x_0."""
        return x_1 - x_0

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        f_sec: torch.Tensor,
        c_txt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list, list]:
        """预测速度及 struc/tex 特征（用于 rSMI）。"""
        return self.unet(x_t, t, f_sec, c_txt)

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """单步估计: x̂_0 = x_t - (1-t) * v."""
        if t.dim() == 0:
            t = t.view(1).expand(x_t.shape[0])
        t = t.to(x_t.device).float().view(-1, 1, 1, 1)
        return x_t - (1 - t) * v

    def sample(
        self,
        shape: Tuple[int, ...],
        f_sec: torch.Tensor,
        c_txt: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        从 x_1 ~ N(0,1) 积分 ODE dx/dt = v_θ(x,t) 到 t=0。
        使用 Euler 或 Heun。这里用简单 Euler: x_{t-dt} = x_t - dt * v_θ(x_t, t).
        dt = 1/num_steps, 从 t=1 到 t=0.
        """
        num_steps = num_steps or self.num_steps
        device = device or next(self.parameters()).device
        B = shape[0]
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (1.0 - (i + 0.5) * dt)
            v, _, _ = self.unet(x, t, f_sec, c_txt)
            x = x - dt * v
        return x
