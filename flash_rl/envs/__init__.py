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
    # NOTE: IsaacLab/IsaacSim only supports one SimulationApp instance per process by design.
    # See https://github.com/isaac-sim/IsaacLab/discussions/1241
    eval_env = train_env
    record_env = train_env

    return train_env, eval_env, record_env
