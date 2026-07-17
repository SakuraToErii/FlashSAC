"""FlashSAC 智能体：训练循环接口（论文 §4 整体编排）。

三大机制在运行时的落点:
  §4.1 Fast Training
    - 大并行环境 / 大 buffer / 大 batch / 低 UTD：由 configs + train.py 控制
      （如 num_train_envs=1024, buffer 10M, batch 2048, updates_per_interaction_step）
    - 本类: TorchUniformBuffer、AMP、torch.compile、actor 延迟更新周期
  §4.2 Stable Training
    - 网络结构: network/layer
    - 权重归一、交叉 batch BN、分布型目标: update.py
    - 奖励缩放: RewardNormalizer（式 6）
  §4.3 Exploration
    - 统一目标熵 Ḧ（式 7）: temp_target_sigma → temp_target_entropy
    - Noise Repetition: Zeta 采样重复长度 k，复用 ϵ ~ N(0,I)

对外接口与 BaseAgent 一致: sample_actions / process_transition / update / save|load。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any, MutableMapping, Optional, cast

import gymnasium as gym
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler

from flash_rl.agents.base_agent import BaseAgent
from flash_rl.agents.flashSAC.network import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
)
from flash_rl.agents.flashSAC.update import (
    update_actor,
    update_critic,
    update_target_network,
    update_temperature,
)
from flash_rl.agents.utils.network import Network
from flash_rl.agents.utils.reward_normalization import RewardNormalizer
from flash_rl.agents.utils.scheduler import warmup_cosine_decay_scheduler
from flash_rl.buffers.torch_buffer import TorchUniformBuffer
from flash_rl.types import NDArray, Tensor


@dataclass
class FlashSACConfig:
    """与 configs/agent/flashSAC.yaml 字段对齐的超参容器。

    论文 GPU 默认量级（§4.1 / 实验设置）:
      buffer_max_length ~ 10M, sample_batch_size 2048,
      actor/critic 多 block 宽网, UTD 由 train 侧 updates_per_interaction_step 体现。
    """

    seed: int
    normalize_reward: bool
    normalized_G_max: float  # 分布 critic 支撑半宽 G_max（式 6 / [−G_max, G_max]）

    asymmetric_observation: bool  # True: actor 只用 policy 维（sim2real 非对称 AC）
    device_type: str

    buffer_max_length: int
    buffer_min_length: int
    buffer_device_type: str
    sample_batch_size: int

    learning_rate_init: float
    learning_rate_peak: float
    learning_rate_end: float
    learning_rate_warmup_rate: float
    learning_rate_warmup_step: int
    learning_rate_decay_rate: float
    learning_rate_decay_step: int

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_bc_alpha: float
    # §4.3 Noise Repetition: P(k) ∝ k^{−μ}, k ∈ [1, max_n]
    actor_noise_zeta_mu: float
    actor_noise_zeta_max: int
    actor_update_period: int  # 每隔多少次 critic 步才更新 actor/温度（降 UTD）

    critic_num_blocks: int
    critic_hidden_dim: int
    critic_num_bins: int  # 分布原子数
    critic_min_v: float
    critic_max_v: float
    critic_target_update_tau: float  # 式 (3) 的 τ

    temp_initial_value: float
    temp_target_sigma: float  # 式 (7) 的 σ_tgt，论文默认 0.15
    temp_target_entropy: float  # 由 σ_tgt 与 |A| 在构造时写入

    gamma: float
    n_step: int

    use_compile: bool  # §4.1 Code Optimization: torch.compile
    compile_mode: str
    use_amp: bool  # 混合精度

    load_optimizer: bool
    load_reward_normalizer: bool


def _init_flashsac_networks(
    actor_observation_dim: int,
    critic_observation_dim: int,
    action_dim: int,
    cfg: FlashSACConfig,
    device: torch.device,
) -> tuple[Network, Network, Network, Network]:
    """构建 actor / critic / target_critic / temperature 及优化器。

    Network 封装: compile、weight-norm 钩子、target 的 EMA 源。
    """
    warmup_cosine_decay_lr = warmup_cosine_decay_scheduler(
        init_value=cfg.learning_rate_init,
        peak_value=cfg.learning_rate_peak,
        end_value=cfg.learning_rate_end,
        warmup_steps=cfg.learning_rate_warmup_step,
        decay_steps=cfg.learning_rate_decay_step,
    )

    # --- Actor π_θ ---
    actor_net = FlashSACActor(
        num_blocks=cfg.actor_num_blocks,
        input_dim=actor_observation_dim,
        hidden_dim=cfg.actor_hidden_dim,
        action_dim=action_dim,
    ).to(device)

    use_fused = device.type == "cuda" and torch.cuda.is_available()
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=cfg.learning_rate_peak, fused=use_fused)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
        actor_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    actor = Network(
        network=actor_net,
        optimizer=actor_optimizer,
        scheduler=actor_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )
    # 部署路径用的 get_mean_and_std 单独 compile
    if cfg.use_compile:
        actor.network.get_mean_and_std = torch.compile(actor.network.get_mean_and_std, mode=cfg.compile_mode)  # type: ignore

    # --- Online Critic Q_ϕ（双 Q 分布型）---
    critic_net = FlashSACDoubleCritic(
        num_blocks=cfg.critic_num_blocks,
        input_dim=critic_observation_dim + action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
    ).to(device)

    critic_optimizer = optim.Adam(
        critic_net.parameters(),
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    critic = Network(
        network=critic_net,
        optimizer=critic_optimizer,
        scheduler=critic_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )

    # --- Target Critic Q̄（式 3 EMA）---
    target_critic_net = FlashSACDoubleCritic(
        num_blocks=cfg.critic_num_blocks,
        input_dim=critic_observation_dim + action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
    ).to(device)
    target_critic_net.load_state_dict(critic_net.state_dict())
    target_critic = Network(
        network=target_critic_net,
        optimizer=None,
        scheduler=None,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
        ema_source=critic,
        ema_tau=cfg.critic_target_update_tau,
    )

    # --- Temperature α ---
    temp_net = FlashSACTemperature(cfg.temp_initial_value).to(device)
    temp_optimizer = optim.Adam(
        temp_net.parameters(),
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    temp_scheduler = torch.optim.lr_scheduler.LambdaLR(
        temp_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    temperature = Network(
        network=temp_net,
        optimizer=temp_optimizer,
        scheduler=temp_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=False,
    )

    # 初始化后即做一次权重归一
    actor.normalize_parameters()
    critic.normalize_parameters()
    target_critic.normalize_parameters()

    return actor, critic, target_critic, temperature


@torch.compile
def _build_truncated_zeta_cdf(mu: float, max_n: int) -> torch.Tensor:
    """截断 Zeta 分布 CDF（§4.3 Noise Repetition）。

    P(k) ∝ k^{−μ}，k=1…max_n，再归一化。短重复更常出现，偶尔长相关轨迹。
    """
    ns = torch.arange(1, max_n + 1, dtype=torch.float32)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    cdf = torch.cumsum(pmf, dim=0)
    return cdf


@torch.compile
def _sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    """从预计算 CDF 采样整数 k（逆变换采样）。"""
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return (idx + 1).to(torch.int32)


def _sample_flashsac_actions(
    actor: Network,
    noise: torch.Tensor,
    observations: torch.Tensor,
    temperature: float,
    cur_count: torch.Tensor,
    cur_n: torch.Tensor,
    zeta_cdf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """动作采样 + Noise Repetition（§4.3）。

    训练 (temperature=1):
      每隔 k 步重采样 ϵ ~ N(0,I)，k ~ Zeta；
      a = tanh(μ + σ ⊙ ϵ)  （ϵ 在重复窗口内固定 → 时间相关探索）
    评估 (temperature=0):
      a = tanh(μ)，确定性策略（部署 / ONNX 同路径）。
    """
    mean, std = actor.apply(
        "get_mean_and_std",
        observations=observations,
        training=False,
    )
    if temperature == 0.0:
        actions = torch.tanh(mean)
        return noise, actions, cur_count, cur_n

    # 到达重复上限则换新噪声与新的 k
    reinit = (cur_count == 0) | (cur_count >= cur_n)

    new_noise = torch.randn_like(mean)
    new_n = _sample_integer_from_cdf(zeta_cdf)

    noise = torch.where(reinit, new_noise, noise)
    cur_n = torch.where(reinit, new_n, cur_n)
    cur_count = torch.where(reinit, torch.zeros_like(cur_count), cur_count)

    actions = torch.tanh(mean + std * noise * temperature)

    return noise, actions, cur_count + 1, cur_n


def _update_networks(
    batch: dict[str, torch.Tensor],
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    cfg: FlashSACConfig,
    do_actor_update: bool,
    device: torch.device,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """单次梯度步: 可选 actor+温度，始终 critic + target EMA。

    actor_update_period>1 时降低策略更新频率，配合 §4.1「更少梯度步」。
    """
    if do_actor_update:
        actor_info = update_actor(
            actor=actor,
            critic=critic,
            temperature=temperature,
            batch=batch,  # type: ignore
            bc_alpha=cfg.actor_bc_alpha,
            device=device,
            use_amp=cfg.use_amp,
            grad_scaler=grad_scaler,
        )

        temperature_info = update_temperature(
            temperature=temperature,
            entropy=actor_info["actor/entropy"],
            target_entropy=cfg.temp_target_entropy,
        )
    else:
        actor_info = {}
        temperature_info = {}

    critic_info = update_critic(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,
        batch=batch,  # type: ignore
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
        num_bins=cfg.critic_num_bins,
        gamma=cfg.gamma,
        n_step=cfg.n_step,
        device=device,
        use_amp=cfg.use_amp,
        grad_scaler=grad_scaler,
    )

    target_critic_info = update_target_network(
        target_network=target_critic,
    )

    update_info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return update_info


def _resolve_compile_mode(mode: str) -> str:
    """将 compile_mode='auto' 映射到具体 torch.compile mode。"""
    if mode != "auto":
        return mode
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        return "max-autotune"
    return "reduce-overhead"


class FlashSACAgent(BaseAgent[FlashSACConfig]):
    """FlashSAC 完整智能体（对应论文算法在 PyTorch 中的落地）。"""

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        env_info: dict[str, Any],
        cfg: FlashSACConfig,
    ):
        # 观测维: critic 用完整 obs（可含 privileged）；actor 可切 policy 前缀
        self._critic_observation_dim: int = observation_space.shape[-1]  # type: ignore
        self._action_dim: int = action_space.shape[-1]  # type: ignore
        if cfg.asymmetric_observation:
            self._actor_observation_dim = env_info["actor_observation_size"][-1]
        else:
            self._actor_observation_dim = self._critic_observation_dim

        # §4.3 式 (7): Ḧ = (1/2) |A| log(2πe σ_tgt²)
        # 实现写为 0.5 * |A| * log(...)，与上式一致
        temp_target_entropy = 0.5 * self._action_dim * math.log(2 * math.pi * math.e * cfg.temp_target_sigma**2)
        compile_mode = _resolve_compile_mode(cfg.compile_mode)
        cfg = replace(cfg, temp_target_entropy=temp_target_entropy, compile_mode=compile_mode)

        super().__init__(
            observation_space,
            action_space,
            env_info,
            cfg,
        )
        self._cfg = cfg

        device_type = cfg.device_type
        device_type = (
            device_type
            if device_type.startswith("cuda") and ":" in device_type
            else ("cuda:0" if device_type.startswith("cuda") else "cpu")
        )
        self._device = torch.device(device_type)

        (
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_flashsac_networks(
            actor_observation_dim=self._actor_observation_dim,
            critic_observation_dim=self._critic_observation_dim,
            action_dim=self._action_dim,
            cfg=self._cfg,
            device=self._device,
        )
        self._update_step = 0

        # §4.1 混合精度
        self._grad_scaler = GradScaler(device=self._device.type, enabled=self._cfg.use_amp)

        # §4.3 Noise Repetition 状态（跨 env 步保持）
        self._zeta_cdf = _build_truncated_zeta_cdf(
            mu=self._cfg.actor_noise_zeta_mu, max_n=self._cfg.actor_noise_zeta_max
        ).to(self._device)
        self._cur_noise_repeat_n = torch.tensor(1, dtype=torch.int32, device=self._device)
        self._cur_noise_repeat_count = torch.tensor(0, dtype=torch.int32, device=self._device)
        action_shape = tuple(action_space.shape) if action_space.shape is not None else ()
        self._cached_noise = torch.randn(action_shape, device=self._device)

        # §4.2 自适应奖励缩放
        self.reward_normalizer = None
        if self._cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=self._cfg.gamma,
                G_max=self._cfg.normalized_G_max,
                load_rms=self._cfg.load_reward_normalizer,
                device=self._device,
            )

        # §4.1 大容量 replay + n-step 回报
        self._replay_buffer = TorchUniformBuffer(
            observation_space=observation_space,
            action_space=action_space,
            n_step=self._cfg.n_step,
            gamma=self._cfg.gamma,
            max_length=self._cfg.buffer_max_length,
            min_length=self._cfg.buffer_min_length,
            sample_batch_size=self._cfg.sample_batch_size,
            device_type=self._cfg.buffer_device_type,
        )

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        """与环境交互时的动作。training=False 时确定性 tanh(mean)。"""
        del interaction_step
        if training:
            temperature = 1.0
        else:
            temperature = 0.0

        observations = prev_transition["next_observation"]
        if self._cfg.asymmetric_observation:
            observations = observations[:, : self._actor_observation_dim]

        observations = torch.as_tensor(observations, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            (
                self._cached_noise,
                actions,
                self._cur_noise_repeat_count,
                self._cur_noise_repeat_n,
            ) = _sample_flashsac_actions(
                actor=self._actor,
                noise=self._cached_noise,
                observations=observations,
                temperature=temperature,
                cur_count=self._cur_noise_repeat_count,
                cur_n=self._cur_noise_repeat_n,
                zeta_cdf=self._zeta_cdf,
            )

        return actions.cpu().numpy()

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None:
        """写入 replay，并更新奖励归一化统计。"""
        self._replay_buffer.add(transition)

        if self._cfg.normalize_reward:
            assert "reward" in transition and self.reward_normalizer is not None
            self.reward_normalizer.update_reward_stats(
                reward=torch.as_tensor(transition["reward"], device=self._device),
                terminated=torch.as_tensor(transition["terminated"], device=self._device),
                truncated=torch.as_tensor(transition["truncated"], device=self._device),
            )

    def can_start_training(self) -> bool:
        """buffer 达到 min_length 后才开始梯度更新。"""
        return self._replay_buffer.can_sample()

    def update(self) -> dict[str, Any]:
        """一次完整更新: sample batch → 归一化 reward → actor/critic/温度/EMA。"""
        batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())

        for k, v in batch.items():
            batch[k] = v.to(self._device, non_blocking=True)

        # 非对称: 策略只看 policy 前缀；critic 看完整 obs
        if self._cfg.asymmetric_observation:
            batch["actor_observation"] = batch["observation"][:, : self._actor_observation_dim]
            batch["actor_next_observation"] = batch["next_observation"][:, : self._actor_observation_dim]
        else:
            batch["actor_observation"] = batch["observation"]
            batch["actor_next_observation"] = batch["next_observation"]

        if self._cfg.normalize_reward:
            assert self.reward_normalizer is not None
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

        _update_info = _update_networks(
            batch=batch,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            cfg=self._cfg,
            do_actor_update=(self._update_step % self._cfg.actor_update_period == 0),
            device=self._device,
            grad_scaler=self._grad_scaler,
        )
        self._update_step += 1

        update_info: dict[str, float] = {}
        for key, value in _update_info.items():
            if isinstance(value, torch.Tensor):
                update_info[key] = value.item()
            elif not isinstance(value, dict):
                update_info[key] = float(value)

        return update_info

    def save(self, path: str) -> None:
        """保存 actor/critic/target/温度/奖励归一/AMP 状态。"""
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "actor.pt"))
        self._critic.save(os.path.join(path, "critic.pt"))
        self._target_critic.save(os.path.join(path, "target_critic.pt"))
        self._temperature.save(os.path.join(path, "temperature.pt"))
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(os.path.join(path, "reward_normalizer.pt"))

        agent_state: dict[str, Any] = {
            "update_step": self._update_step,
            "grad_scaler_state_dict": self._grad_scaler.state_dict(),
        }
        torch.save(agent_state, os.path.join(path, "agent_state.pt"))
        print(f"\033[32m[FlashSAC]\033[0m Successfully saved checkpoint {self._update_step} at {path}.")

    def save_replay_buffer(self, path: str) -> None:
        self._replay_buffer.save(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[FlashSAC]\033[0m Successfully saved replay buffer at {path}.")

    def load(self, path: str) -> None:
        load_optimizer = self._cfg.load_optimizer
        self._actor.load(os.path.join(path, "actor.pt"), load_optimizer=load_optimizer)
        self._critic.load(os.path.join(path, "critic.pt"), load_optimizer=load_optimizer)
        self._target_critic.load(os.path.join(path, "target_critic.pt"), load_optimizer=False)
        self._temperature.load(os.path.join(path, "temperature.pt"), load_optimizer=load_optimizer)

        if load_optimizer:
            agent_state_path = os.path.join(path, "agent_state.pt")
            assert os.path.exists(agent_state_path)
            agent_state = torch.load(agent_state_path, map_location=self._device)
            self._update_step = agent_state["update_step"]
            self._grad_scaler.load_state_dict(agent_state["grad_scaler_state_dict"])

        if self._cfg.load_reward_normalizer:
            assert self.reward_normalizer is not None
            self.reward_normalizer.load(os.path.join(path, "reward_normalizer.pt"))

        print(f"\033[32m[FlashSAC]\033[0m Successfully loaded checkpoint from {path}.")

    def load_replay_buffer(self, path: str) -> None:
        self._replay_buffer.load(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[FlashSAC]\033[0m Successfully loaded replay buffer from {path}.")

    def get_metrics(self) -> dict[str, Any]:
        return {}
