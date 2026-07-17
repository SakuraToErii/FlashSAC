"""策略分布相关数值工具。

SAC / FlashSAC 使用对角高斯策略，经 tanh 压到有界动作空间。
对密度做换元时需扣除 Jacobian 行列式的对数，见 Haarnoja et al. SAC。
"""

import math

import torch
import torch.nn.functional as F


def safe_tanh_log_det_jacobian(x: torch.Tensor) -> torch.Tensor:
    """计算 log |det ∂tanh(x)/∂x| 的数值稳定形式。

    对每个维度: log(1 - tanh(x)^2) = 2*(log 2 - x - softplus(-2x))。
    相对直接 log(1-tanh^2) 在 |x| 较大时更不易下溢。

    用于: log π(a|s) = log N(u|μ,σ) - ∑_i log|1-tanh(u_i)^2|，
    其中 a = tanh(u)，u 为未压动作（reparameterized sample）。

    参考: https://github.com/google-deepmind/distrax/issues/216
    """
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
