"""FlashSAC 网络层（论文 §4.2 Stable Training + Figure 2）。

架构增量（论文 Figure 9 消融顺序）:
  MLP → + Residual (倒残差块) → + Pre-BN → + Post-RMSNorm
      → + Distributional Critic → + Weight Norm

对应论文机制:
  - Inverted Residual Backbone: FlashSACBlock（扩维 4× → ReLU → 压回 + skip）
  - Pre-activation BatchNorm: 非线性前的 UnitBatchNorm
  - Post RMSNorm: 价值/策略头前的 UnitRMSNorm，约束样本特征范数
  - Weight Normalization: normalize_parameters()，每步优化后投影
  - Distributional Critic: EnsembleCategoricalValue 在 [min_v, max_v] 上的 categorical
  - Clipped Double Q: Ensemble* 两套 critic 共享结构、批维 N=2

Unit* 前缀: 权重/BN 仿射参数受 unit-norm 约束（§4.2 Weight Normalization）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_rl.agents.utils.distribution import safe_tanh_log_det_jacobian


class UnitLinear(nn.Module):
    """无偏置线性层 + 行向量单位化（§4.2 Weight Normalization）。

    优化后把每个输出通道对应的权重向量投影到单位球，信息主要编码在方向上，
    抑制权重范数膨胀，从而限制 bootstrapping 下 Q 方差放大。
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)  # type: ignore[no-any-return]

    def normalize_parameters(self) -> None:
        """对 weight 最后一维（输入特征维）做 L2 归一化。"""
        self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    """预激活 BatchNorm（§4.2 Pre-activation Batch Normalization）。

    Replay 数据来自不断变化的行为策略，输入分布非平稳；BN 在非线性前
    维持激活尺度，大 batch 统计也有助于更平滑的损失景观。

    仿射参数 (γ, β) 同样被 normalize_parameters 约束到范数 √d。
    """

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def normalize_parameters(self) -> None:
        """约束 (γ, β) 使 ‖(γ,β)‖ = √d（论文 d 为通道维）。"""
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class UnitRMSNorm(nn.Module):
    """块后 / 头前 RMSNorm（§4.2，Figure 2 最右侧 RMSNorm）。

    限制进入策略/价值头的每样本特征范数，减轻 OOD 输入导致的无界激活
    与不稳定 bootstrap。

    前向用显式公式（与 F.rms_norm 等价），便于 ONNX 导出。
    """

    def __init__(self, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * (self.weight / rms)

    def normalize_parameters(self) -> None:
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class FlashSACEmbedder(nn.Module):
    """输入嵌入: BN → UnitLinear，把观测投到 hidden_dim。"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class FlashSACBlock(nn.Module):
    """倒残差块（§4.2 Inverted Residual Backbone, Figure 2）。

    结构（与 Transformer FFN / inverted bottleneck 同类）:
      x → Linear(d→4d) → BN → ReLU → Linear(4d→d) → BN → ReLU → +x

    expansion=4 对应图中 4d 中间维；残差连接稳定深层梯度。
    """

    def __init__(self, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        x = x + residual
        return x


class NormalTanhPolicy(nn.Module):
    """对角高斯 + tanh 有界动作（§3.2 SAC 随机策略）。

    输出 mean 与 std；采样 u ~ N(μ,σ)，a = tanh(u)。
    log π 扣除 tanh Jacobian（safe_tanh_log_det_jacobian）。
    log_std 经 tanh 软限幅到 [log_std_min, log_std_max]，训练更稳。
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.mean_w = UnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))

        self.std_w = UnitLinear(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def get_mean_and_std(
        self,
        x: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del training  # 头本身无 BN；签名与 trunk 一致
        # F.linear 便于 AMP 与权重归一后的 functional 前向
        mean = F.linear(x, self.mean_w.w.weight, self.mean_bias)
        raw_log_std = F.linear(x, self.std_w.w.weight, self.std_bias)

        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(raw_log_std))
        std = torch.exp(log_std)

        return mean, std

    def forward(
        self,
        x: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean, std = self.get_mean_and_std(x, training)

        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()  # 重参数化，可反传
        tanh_action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action)  # type: ignore[no-untyped-call]
        log_prob = log_prob - safe_tanh_log_det_jacobian(raw_action)
        log_prob = log_prob.sum(1)

        info: dict[str, torch.Tensor] = {"log_prob": log_prob}
        return tanh_action, info


# ---------------------------------------------------------------------------
# Double-Q 集成层: 张量布局 (num_ensemble, batch, dim)，一次前向两套 critic
# ---------------------------------------------------------------------------


class EnsembleUnitLinear(nn.Module):
    """num_ensemble 个 UnitLinear 的批量化 einsum 实现。"""

    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for i in range(num_ensemble):
            nn.init.orthogonal_(self.weight.data[i], gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, B, in] @ [N, out, in]^T → [N, B, out]
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    def normalize_parameters(self) -> None:
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    """集成版 BN：在 batch 维（dim=1）上算统计，每个 ensemble 成员独立 γ/β。"""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_ensemble: int, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                B = x.shape[1]
                # running stats 保持 float32（即使 AMP）
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_((var.squeeze(1) * (B / (B - 1))).float(), self.momentum)
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(self.running_var.unsqueeze(1) + self.eps)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    def normalize_parameters(self) -> None:
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class EnsembleUnitRMSNorm(nn.Module):
    """集成版 RMSNorm，布局 (N, B, d)。"""

    def __init__(self, num_ensemble: int, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    def normalize_parameters(self) -> None:
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class EnsembleFlashSACEmbedder(nn.Module):
    """集成 critic 的输入嵌入。"""

    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class EnsembleFlashSACBlock(nn.Module):
    """集成 critic 的倒残差块（结构同 FlashSACBlock）。"""

    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        x = x + residual
        return x


class EnsembleCategoricalValue(nn.Module):
    """分布型 Q 头（§4.2 Distributional Critic）。

    在 [min_v, max_v] 上均匀放置 num_bins 个原子（论文 G_min…G_max），
    网络输出各原子 logits → softmax 得 p_i，期望 Q = ∑ p_i z_i。
    训练时对投影后的 Bellman 目标做交叉熵（见 update._compute_categorical_td_target）。
    """

    bin_values: torch.Tensor

    def __init__(
        self,
        num_ensemble: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
    ):
        super().__init__()
        self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))
        # 固定原子位置 z_i，不参与梯度
        self.register_buffer(
            "bin_values",
            torch.linspace(start=min_v, end=max_v, steps=num_bins, dtype=torch.float32).reshape(1, 1, -1),
        )

    def forward(
        self,
        x: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del training
        value = self.w(x) + self.bias.unsqueeze(1)
        log_prob = F.log_softmax(value, dim=-1)
        # 期望 Q 值 (N, B)
        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1)
        info: dict[str, torch.Tensor] = {"log_prob": log_prob}
        return value, info
