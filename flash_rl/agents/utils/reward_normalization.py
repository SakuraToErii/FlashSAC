"""回报自适应奖励缩放（论文 §4.2 Distributional Critic with Adaptive Reward Scaling）。

分布型 critic 的支撑固定为 [G_min, G_max]（代码中 critic_min_v / critic_max_v，
由 normalized_G_max 导出）。直接用原始回报时，不同任务的回报尺度差异会导致
概率质量压在支撑两端。论文公式 (6):

    r̄_t = r_t / max(√(σ_t² + ε), G_{t,max} / G_max)

其中 σ_t² 是折扣回报 G 的运行方差，G_{t,max} 是 |G| 的历史峰值，
G_max 是 critic 支撑半宽（配置项 normalized_G_max，默认 5.0）。

实现上在环境步进时维护折扣回报统计，在 sample batch 更新 critic 前缩放 reward。
"""

import os
from typing import TypeVar

import torch

Config = TypeVar("Config")


@torch.compile
def _update_reward_stats(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    G_r: torch.Tensor,
    G_r_max: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """递推更新折扣回报 G 及其绝对值峰值。

    G ← γ * (1 - done) * G + r
    episode 结束 (terminated | truncated) 时截断折扣链。
    """
    done = torch.logical_or(terminated, truncated).float()
    new_G_r = gamma * (1.0 - done) * G_r + reward
    new_G_r_max = torch.maximum(G_r_max, torch.max(torch.abs(new_G_r)))
    return new_G_r, new_G_r_max


@torch.compile
def _scale_reward(
    rewards: torch.Tensor,
    G_var: torch.Tensor,
    G_r_max: torch.Tensor,
    G_max: float,
    eps: float,
) -> torch.Tensor:
    """论文式 (6): 用方差项与峰值项的较大者做分母，缩放 batch 内奖励。"""
    var_denominator = torch.sqrt(G_var + eps)
    min_required_denominator = G_r_max / G_max
    denominator = torch.maximum(var_denominator, min_required_denominator)
    return rewards / denominator


@torch.compile
def _update_mean_var_count_from_moments(
    samples: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    running_count: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Welford / 并行合并算法更新运行均值与方差。"""
    sample_mean = torch.mean(samples, dim=0)
    sample_var = torch.var(samples, dim=0, unbiased=False)
    sample_count = float(samples.shape[0])

    delta = sample_mean - running_mean
    total_count = running_count + sample_count
    ratio = sample_count / total_count

    new_mean = running_mean + delta * ratio
    m_a = running_var * (running_count + epsilon)
    m_b = sample_var * sample_count
    M2 = m_a + m_b + torch.square(delta) * running_count * ratio
    new_var = M2 / total_count

    return (
        new_mean,
        new_var,
        total_count,
    )


class RewardNormalizer:
    """在线奖励归一化器（§4.2）。

    在 process_transition 中用环境奖励更新 G 的统计；
    在 update() 采样 batch 后对 reward 做 normalize_rewards。
    """

    def __init__(
        self,
        gamma: float,
        G_max: float,
        load_rms: bool,
        device: torch.device,
        epsilon: float = 1e-8,
    ):
        self.gamma = gamma
        # 当前折扣回报轨迹（按 env 维；首步后会与 reward 广播对齐）
        self.G_r = torch.zeros(1, dtype=torch.float32, device=device)
        self.G_r_max = torch.zeros(1, dtype=torch.float32, device=device)
        self.G_rms = RunningMeanStd(
            shape=(1,),
            device=device,
            dtype=torch.float32,
        )
        self.G_max = G_max
        self.load_rms = load_rms
        self.epsilon = epsilon
        self.device = device

    def update_reward_stats(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """每步环境交互后调用：更新 G 与 running var。"""
        self.G_r, self.G_r_max = _update_reward_stats(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            G_r=self.G_r,
            G_r_max=self.G_r_max,
            gamma=self.gamma,
        )
        self.G_rms.update(self.G_r)

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """更新 critic 前缩放 batch 奖励（式 6）。"""
        normalized_rewards = _scale_reward(
            rewards=rewards,
            G_var=self.G_rms.var,
            G_r_max=self.G_r_max,
            G_max=self.G_max,
            eps=self.epsilon,
        )
        return normalized_rewards

    def save(self, path: str) -> None:
        """保存归一化统计，便于断点续训。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "G_r": self.G_r,
            "G_r_max": self.G_r_max,
            "G_rms_mean": self.G_rms.mean,
            "G_rms_var": self.G_rms.var,
            "G_rms_count": self.G_rms.count,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """加载归一化统计。"""
        state = torch.load(path, map_location=self.device)
        self.G_r = state["G_r"]
        self.G_r_max = state["G_r_max"]
        self.G_rms.mean = state["G_rms_mean"]
        self.G_rms.var = state["G_rms_var"]
        self.G_rms.count = state["G_rms_count"]


class RunningMeanStd:
    """运行均值 / 方差（与 OpenAI baselines 同类实现）。"""

    def __init__(
        self,
        device: torch.device,
        epsilon: float = 1e-4,
        shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
    ):
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(0.0, dtype=dtype, device=device)
        self.epsilon = epsilon
        self.device = device

    def update(self, x: torch.Tensor) -> None:
        self.mean, self.var, self.count = _update_mean_var_count_from_moments(
            samples=x,
            running_mean=self.mean,
            running_var=self.var,
            running_count=self.count,
            epsilon=self.epsilon,
        )
