"""Environment factory for FlashSAC.

This fork only supports Isaac Lab (``env_type='isaaclab'``), including
official Isaac-* tasks and unitree_rl_lab Unitree-* tasks once registered.
"""

from __future__ import annotations

from typing import Any, Optional

from gymnasium.vector import VectorEnv

from ..types import NDArray


def create_envs(
    env_type: str,
    seed: int,
    env_name: str,
    num_train_envs: int,
    num_eval_envs: int,
    num_record_envs: int,
    rescale_action: bool,
    max_episode_steps: Optional[int],
    **kwargs: Any,
) -> tuple[
    VectorEnv[NDArray, NDArray, NDArray],
    VectorEnv[NDArray, NDArray, NDArray],
    VectorEnv[NDArray, NDArray, NDArray],
]:
    """Create train / eval / record envs (all the same object under Isaac Lab).

    Isaac Sim allows only one SimulationApp per process, so eval and record
    aliases share the training env. Config must set ``num_eval_envs``,
    ``num_record_envs``, and ``rescale_action`` to null for this backend.
    """
    del kwargs  # Reserved for future backend-specific options.
    if env_type != "isaaclab":
        raise NotImplementedError(f"Only env_type='isaaclab' is supported, got {env_type!r}")

    from flash_rl.envs.isaaclab import make_isaaclab_env

    assert rescale_action is None, "Unused hyperparameter in IsaacLab."
    assert num_eval_envs is None, "Unused hyperparameter in IsaacLab."
    assert num_record_envs is None, "Unused hyperparameter in IsaacLab."
    train_env = make_isaaclab_env(
        env_name=env_name,
        num_envs=num_train_envs,
        seed=seed,
    )
    # One SimulationApp per process:
    # https://github.com/isaac-sim/IsaacLab/discussions/1241
    eval_env = train_env
    record_env = train_env

    return train_env, eval_env, record_env
