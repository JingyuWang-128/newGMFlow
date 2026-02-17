"""
Rectified Flow 生成器
x_t = t*x_1 + (1-t)*x_0, 学习速度场 v_θ(x_t, t) ≈ x_1 - x_0，单步估计 x̂_0 = x_t - (1-t)*v_θ.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import math

from .dis import TriStreamDiS


class RectifiedFlowGenerator(nn.Module):
    """
    基于三流 Mamba DiS 主干的 Rectified Flow。
    前向：给定 x_0（噪声）, x_1（目标图）, t，计算 x_t 与目标速度，用 v_θ 预测并算损失。
    采样：从 x_1 ~ N(0,1) 沿 ODE 积分到 t=0 得到 x_0。
    """

    def __init__(self, backbone: nn.Module, num_steps: int = 1000):
        super().__init__()
        self.backbone = backbone
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
        """预测速度及 struc/tex 特征（用于正交/去相关损失）。"""
        return self.backbone(x_t, t, f_sec, c_txt)

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """单步估计: x̂_0 = x_t - (1-t) * v."""
        if t.dim() == 0:
            t = t.view(1).expand(x_t.shape[0])
        t = t.to(x_t.device).float().view(-1, 1, 1, 1)
        return x_t - (1 - t) * v

    def sample_pc(
        self,
        shape: Tuple[int, ...],
        f_sec: torch.Tensor,
        c_txt: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        corrector_steps: int = 1,
        snr: float = 0.15,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Predictor-Corrector 纠错采样算法。
        - Predictor: 一阶 Euler 积分前进。
        - Corrector: Langevin 随机动力学，在每步注入噪声并去噪，实现内在的对抗鲁棒性。
        """
        device = device or next(self.parameters()).device
        B = shape[0]
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            # t 从 1 降到 0
            t = torch.ones(B, device=device) * (1.0 - i * dt)
            v, _, _ = self.backbone(x, t, f_sec, c_txt)
            
            # 1. Predictor (Euler)
            x_prev = x - dt * v
            
            # 2. Corrector (Langevin-like Score Correction)
            # 在最后几步不加噪声以保证收敛质量
            if i < num_steps - 1 and corrector_steps > 0:
                for _ in range(corrector_steps):
                    t_prev = torch.ones(B, device=device) * (1.0 - (i + 1) * dt)
                    v_corr, _, _ = self.backbone(x_prev, t_prev, f_sec, c_txt)
                    
                    # 生成纠错噪声
                    noise = torch.randn_like(x_prev)
                    
                    # 计算基于信噪比(SNR)的校正步长
                    step_size = (snr * dt) ** 2
                    
                    # 假定 v 近似正比于 -score，执行 Langevin 更新
                    x_prev = x_prev - step_size * v_corr + torch.sqrt(torch.tensor(2 * step_size)) * noise

            x = x_prev
            
        return x

    def sample(
        self,
        shape: Tuple[int, ...],
        f_sec: torch.Tensor,
        c_txt: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        corrector_steps: int = 1,
        snr: float = 0.15,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        从 x_1 ~ N(0,1) 沿 ODE 积分到 t=0。默认使用 Predictor-Corrector 纠错采样；
        生成即防御在采样器层面通过 Corrector 实现，而非在直线插值轨迹上做手脚。
        """
        num_steps = num_steps or self.num_steps
        return self.sample_pc(shape, f_sec, c_txt, num_steps=num_steps, corrector_steps=corrector_steps, snr=snr, device=device)
