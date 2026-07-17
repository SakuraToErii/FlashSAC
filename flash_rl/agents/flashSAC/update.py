"""FlashSAC 参数更新（论文 §3.2 SAC 目标 + §4.2 稳定化）。

主要目标（与论文对应）:
  策略 (2):  L_π = E[ α log π(a|s) − min_i Q_i(s,a) ]
  Critic (4)(5): 对分布型 Q 用投影 Bellman 目标 + 交叉熵
  目标网 (3):  ϕ̄ ← τ ϕ + (1−τ) ϕ̄  （代码里对 target 做 EMA）
  温度:      自动调 α 使熵接近 Ḧ（§4.3）

§4.2 Cross-Batch Value Prediction:
  将 (s,a) 与 (s',a') 在 batch 维拼接后一次过 critic/BN，
  使当前 Q 与目标 Q 共享同一批 BN 统计，避免目标与预测归一不一致。
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.amp.grad_scaler import GradScaler

from flash_rl.agents.utils.network import Network
from flash_rl.buffers import Batch


def add_prefix_to_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}


@torch.compile
def _select_min_q_log_probs(
    next_qs: torch.Tensor,  # (2, B)
    next_q_log_probs: torch.Tensor,  # (2, B, num_bins)
) -> torch.Tensor:
    """Clipped Double Q: 取期望 Q 更小的那条 critic 的原子 log_prob。

    对应式 (5) 中 min_{j=1,2} Q̄_j；分布型情形对「被选中」的那条分布做备份。
    返回形状 (B, num_bins)。
    """
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)  # (B,)
    selected = torch.gather(
        next_q_log_probs,
        dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins),
    )[
        0
    ]  # (B, num_bins)
    return selected


@torch.compile
def _compute_categorical_td_target(
    target_log_probs: torch.Tensor,  # (B, num_bins)
    reward: torch.Tensor,  # (B,)
    done: torch.Tensor,  # (B,)
    actor_entropy: torch.Tensor,  # (B,)  即 α log π(a'|s')
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    """分布型 Bellman 目标投影（Bellemare et al. C51 风格）。

    对每个原子位置 z_i:
      Tz_i = r + γ (z_i − α log π) (1−done)   （soft Bellman，含熵项）
    将 Tz_i clamp 到 [min_v, max_v]，再按距离把质量线性分到相邻两个 bin
    （HL-Gauss / C51 projection）。返回目标概率质量 (B, num_bins)。
    """
    batch_size = reward.shape[0]

    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(
        min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
    ).view(1, -1)

    # soft 备份: Q 目标分布支撑上的「推后」位置
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    # 投影到离散原子格点
    b = (target_bin_values - min_v) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)

    frac = b - lower.float()

    target_probs_exp = target_log_probs.exp()
    m_l = target_probs_exp * (1.0 - frac)
    m_u = target_probs_exp * frac

    target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device)

    target_probs.scatter_add_(1, lower, m_l)
    target_probs.scatter_add_(1, upper, m_u)

    return target_probs


def update_actor(
    actor: Network,
    critic: Network,
    temperature: Network,
    batch: Batch,
    bc_alpha: float,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """策略更新（论文式 (2)）。

    L_π = E[ α log π(a|s) − min_i Q_i(s,a) ]

    Cross-batch: 将 actor 在 s 与 s' 上的观测拼接，一次 BN 前向（与 critic 侧一致）。
    可选 bc_alpha>0 时加 BC 正则（TD3+BC 类，论文主结果默认 0）。
    优化后调用 weight normalize（§4.2）。
    """
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        # Cross-batch: 当前 obs 与 next obs 共享 actor 侧 BN 统计
        actor_obs_all = torch.cat([batch["actor_observation"], batch["actor_next_observation"]], dim=0)  # type: ignore
        actions_all, info = actor(
            observations=actor_obs_all,
            training=True,
        )
        log_probs_all = info["log_prob"]

        actions = torch.chunk(actions_all, 2, dim=0)[0]
        log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]

        # 冻结 critic 参数图，只作评估；避免 CUDA graph / 多余反传
        critic.network.requires_grad_(False)
        qs, q_infos = critic(
            observations=batch["observation"],
            actions=actions,
            training=False,
        )
        del q_infos
        q = torch.minimum(qs[0], qs[1])  # clipped double Q
        critic.network.requires_grad_(True)

        temp_value = temperature().detach()
        # 式 (2): α log π − Q  （最小化该式 ≡ 最大化 Q − α log π）
        actor_loss = (log_probs * temp_value - q).mean()

        if bc_alpha > 0:
            # 可选行为克隆正则（Fujimoto et al. 类）；默认关闭
            q_abs = torch.abs(q).mean().detach()
            bc_loss = ((actions - batch["action"]) ** 2).mean()
            actor_loss = actor_loss + bc_alpha * q_abs * bc_loss

        entropy = -log_probs.mean()
        mean_action = actions.mean()

    assert actor.optimizer is not None
    actor.optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(actor_loss).backward()
        grad_scaler.step(actor.optimizer)
        grad_scaler.update()
    else:
        actor_loss.backward()
        actor.optimizer.step()

    if actor.scheduler is not None:
        actor.scheduler.step()

    # §4.2 Weight Normalization：投影权重 / BN 仿射参数
    actor.normalize_parameters()

    update_info = {
        "loss": actor_loss,
        "entropy": entropy,
        "mean_action": mean_action,
    }
    update_info = add_prefix_to_keys(update_info, "actor")

    return update_info


def update_critic(
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    batch: Batch,
    min_v: float,
    max_v: float,
    num_bins: int,
    gamma: float,
    n_step: int,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """Critic 更新（式 (4)(5) + 分布型交叉熵 + Cross-Batch BN）。

    1. 无梯度: a' ~ π(·|s')，用 target_critic 在拼接 batch 上前向
    2. min-Q 选分布，投影 soft Bellman 目标
    3. 在线 critic 对 (s,a) 半段做 CE；损失对两套 Q 平均
    4. 权重归一化
    """
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        with torch.no_grad():
            next_actions, info = actor(
                observations=batch["actor_next_observation"],
                training=False,
            )
            next_actions = next_actions.clone()
            next_actor_log_probs = info["log_prob"].clone()

            temp_value = temperature()
            # 式 (5) 中的 α log π(a'|s')
            next_actor_entropy = temp_value * next_actor_log_probs

            # Cross-Batch (§4.2): (s,a) 与 (s',a') 拼成 2B，共享 BN 统计
            obs_all = torch.cat([batch["observation"], batch["next_observation"]], dim=0)  # type: ignore
            act_all = torch.cat([batch["action"], next_actions], dim=0)  # type: ignore

            # qs_all: (2, 2B)；后半段对应 next
            qs_all, q_infos_all = target_critic(
                observations=obs_all,
                actions=act_all,
                training=True,
            )
            next_qs = qs_all.chunk(2, dim=1)[1]
            next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
            next_q_log_probs = _select_min_q_log_probs(next_qs, next_q_log_probs)

            # n-step 回报时折扣为 γ^n
            target_probs = _compute_categorical_td_target(
                target_log_probs=next_q_log_probs,
                reward=batch["reward"],  # type: ignore
                done=batch["terminated"],  # type: ignore
                actor_entropy=next_actor_entropy,
                gamma=gamma**n_step,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            max_entropy_bonus = next_actor_entropy.max()

        # 在线 critic 同样 cross-batch 前向；损失只用当前半段 log_prob
        pred_qs_all, pred_q_infos = critic(
            observations=obs_all,
            actions=act_all,
            training=True,
        )
        del pred_qs_all
        pred_log_probs = torch.chunk(pred_q_infos["log_prob"], 2, dim=1)[0]

        # 分布型：−∑ target_prob * log pred  （交叉熵）
        ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)  # (2, B)
        critic_loss = ce_loss.mean()

    assert critic.optimizer is not None
    critic.optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(critic_loss).backward()  # type: ignore
        grad_scaler.step(critic.optimizer)
        grad_scaler.update()
    else:
        critic_loss.backward()  # type: ignore
        critic.optimizer.step()

    if critic.scheduler is not None:
        critic.scheduler.step()

    critic.normalize_parameters()

    update_info = {
        "loss": critic_loss,
        "max_entropy_bonus": max_entropy_bonus,
    }
    update_info = add_prefix_to_keys(update_info, "critic")

    return update_info


@torch.no_grad()
def update_target_network(
    target_network: Network,
) -> dict[str, torch.Tensor]:
    """目标网络软更新（式 (3)）: θ̄ ← τ θ + (1−τ) θ̄。

    τ = critic_target_update_tau，在 Network 构造时写入 ema_tau。
    """
    target_network.ema_update_parameters()
    info: dict[str, torch.Tensor] = {}
    return info


def update_temperature(
    temperature: Network,
    entropy: torch.Tensor,
    target_entropy: float,
) -> dict[str, torch.Tensor]:
    """自动温度（SAC）: 使策略熵接近目标熵 Ḧ。

    Ḧ 由 §4.3 式 (7) 用统一 σ_tgt 与动作维 |A| 给出（在 agent 初始化时算好）。
    损失形式: α * (H − Ḧ)，H 为当前策略熵估计。
    """
    temperature_value = temperature().clone()
    temperature_loss = temperature_value * (entropy.detach() - target_entropy).mean()

    assert temperature.optimizer is not None
    temperature.optimizer.zero_grad(set_to_none=True)
    temperature_loss.backward()
    temperature.optimizer.step()
    if temperature.scheduler is not None:
        temperature.scheduler.step()

    update_info = {
        "value": temperature_value,
        "loss": temperature_loss,
    }
    update_info = add_prefix_to_keys(update_info, "temperature")

    return update_info
