"""Isaac Lab / unitree_rl_lab environment adapter for FlashSAC.

Bridges ManagerBased (and Direct) Isaac Lab tasks into the Gymnasium VectorEnv
API that ``train.py`` expects.

Key contracts
-------------
Observation
    - Isaac Lab returns a dict with at least ``policy`` (and often ``critic`` for
      privileged info).
    - This wrapper concatenates ``policy || critic`` into one vector for the
      buffer / critic. The actor policy dimension is reported separately via
      ``infos["actor_observation_size"]`` so FlashSAC can slice when
      ``asymmetric_observation=true`` (required for sim2sim / real deploy).

Action
    - The agent outputs roughly in ``[-1, 1]`` (tanh for FlashSAC).
    - Before ``env.step``, we clamp to ``[-1, 1]`` and multiply by
      ``ACTION_BOUNDS[env_name]`` (usually ``1.0``).
    - Isaac Lab's action manager then applies task-specific scale / offset
      (e.g. JointPositionAction). Those scale/offset values are what
      ``deploy.yaml`` captures for unitree controllers — not ACTION_BOUNDS.

Episode horizon (RSL-RL style)
    - ``random_start_init=True`` (training): randomize ``episode_length_buf`` so
      parallel envs do not timeout together.
    - ``random_start_init=False`` (eval / play): full episodes from a clean start.

Process limit
    - One ``SimulationApp`` per process. train / eval / record therefore share
      the same env instance (see ``flash_rl.envs.create_envs``).
"""

from __future__ import annotations

from typing import Any, Union, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import F32NDArray, NDArray

# Pre-step action scale applied by this wrapper (agent output * bounds → env).
# Isaac Lab does not expose a uniform "action range" on the gym API, so we keep
# a task table (same idea as FastTD3). Locomotion tasks use 1.0 so the network
# stays in ~[-1, 1] and the env action manager owns physical scale/offset.
# Unknown task names fall back to 1.0 with a warning in make_isaaclab_env().
ACTION_BOUNDS = {
    # --- Official Isaac Lab tasks ---
    "Isaac-Repose-Cube-Shadow-Direct-v0": 1.0,
    "Isaac-Repose-Cube-Allegro-Direct-v0": 1.0,
    "Isaac-Velocity-Flat-G1-v0": 1.0,
    "Isaac-Velocity-Rough-G1-v0": 1.0,
    "Isaac-Velocity-Flat-H1-v0": 1.0,
    "Isaac-Velocity-Rough-H1-v0": 1.0,
    "Isaac-Lift-Cube-Franka-v0": 3.0,
    "Isaac-Open-Drawer-Franka-v0": 3.0,
    "Isaac-Velocity-Flat-Anymal-C-v0": 1.0,
    "Isaac-Velocity-Rough-Anymal-C-v0": 1.0,
    "Isaac-Velocity-Flat-Anymal-D-v0": 1.0,
    "Isaac-Velocity-Rough-Anymal-D-v0": 1.0,
    # --- unitree_rl_lab (deploy-oriented); bounds=1.0 matches RSL policy range ---
    "Unitree-G1-29dof-Velocity": 1.0,
    "Unitree-G1-29dof-Velocity-Rough": 1.0,
    "Unitree-G1-29dof-Velocity-POMDP1": 1.0,
    "Unitree-G1-29dof-Velocity-POMDP2": 1.0,
    "Unitree-H1-Velocity": 1.0,
    "Unitree-Go2-Velocity": 1.0,
}


def _register_task_packages() -> None:
    """Import gym task registries before parse_env_cfg / gym.make.

    - ``isaaclab_tasks``: official Isaac-* ids (usually already importable).
    - ``unitree_rl_lab.tasks``: Unitree-* ids; optional — only needed when
      training or exporting those tasks. Missing package is silent so pure
      Isaac Lab installs still work.
    """
    try:
        import isaaclab_tasks  # noqa: F401
    except ImportError:
        pass
    try:
        import unitree_rl_lab.tasks  # noqa: F401
    except ImportError:
        # Optional: only required for Unitree-* task ids.
        pass


def recursive_to_numpy(
    data: Union[torch.Tensor, dict[str, Any], list[Any], tuple[Any, ...], NDArray],
) -> Union[NDArray, dict[str, Any], list[Any], tuple[Any, ...]]:
    """Move nested torch structures to numpy (FlashSAC buffers prefer numpy)."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {k: recursive_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_numpy(v) for v in data)
    else:
        return data


class IsaacLabVectorEnv(
    VectorEnv[Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray]]
):
    """Gymnasium-style vector env over a single Isaac Lab ``gym.make`` instance.

    Isaac Lab already vectorizes over ``num_envs`` on GPU; we do not spawn
    multiple processes. Observation layout and action bounds are described in
    the module docstring.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        seed: int,
        device: str,
        action_bounds: float,
        to_numpy: bool = True,
        headless: bool = True,
    ):
        # Starts Omniverse / Isaac Sim (one AppLauncher per process).
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=headless, device=device, enable_cameras=not headless)
        self.simulation_app = app_launcher.app

        # Ensure gym registry contains Isaac-* and (if installed) Unitree-*.
        _register_task_packages()

        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(
            env_name,
            device=device,
            num_envs=num_envs,
        )
        env_cfg.seed = seed
        self.seed = seed
        self.device = device
        self.envs = gym.make(env_name, cfg=env_cfg, render_mode=None)

        self.num_envs = cast(Any, self.envs.unwrapped).num_envs
        self.max_episode_steps = cast(Any, self.envs.unwrapped).max_episode_length
        self.to_numpy = to_numpy

        # --- Observation spaces ---
        # policy: what deploy / real robot can measure
        # critic: privileged sim terms (optional); concatenated for training buffer
        self.obs_size = cast(Any, self.envs.unwrapped).single_observation_space["policy"].shape
        self.asymmetric_obs = "critic" in cast(Any, self.envs.unwrapped).single_observation_space
        if self.asymmetric_obs:
            # Concat shape = policy + critic. FlashSAC agent uses infos to know
            # the policy slice length when asymmetric_observation=true.
            self.critic_obs_size = cast(Any, self.envs.unwrapped).single_observation_space["critic"].shape
            # Bounds unused; only shape/dtype matter for buffer allocation.
            self.single_observation_space = gym.spaces.Box(
                low=0.0, high=0.0, shape=self.obs_size + self.critic_obs_size, dtype=np.float32
            )
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        else:
            self.critic_obs_size = 0
            self.single_observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=self.obs_size, dtype=np.float32)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        # --- Action space (agent-facing; physical scale lives in env managers) ---
        self.action_bounds = action_bounds
        self.action_size = cast(Any, self.envs.unwrapped).single_action_space.shape
        self.single_action_space = gym.spaces.Box(
            low=-1.0 * self.action_bounds, high=1.0 * self.action_bounds, shape=self.action_size, dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        random_start_init: bool = True,
    ) -> tuple[Union[torch.Tensor, F32NDArray], dict[str, Any]]:
        """Reset all envs; optionally decorrelate episode horizons.

        Args:
            random_start_init: If True (training default), randomize
                ``episode_length_buf`` so timeouts are staggered (RSL-RL style).
                If False (eval / play / video), start from a full episode.
        """
        del seed, options  # Gymnasium API compatibility; Isaac Lab cfg seed is set at make time.
        obs_dict, infos = self.envs.reset()
        obs = obs_dict["policy"]
        if self.asymmetric_obs:
            critic_obs = obs_dict["critic"]
            obs = torch.cat((obs, critic_obs), dim=-1)
        else:
            critic_obs = None
        # Decorrelate horizons: Isaac uses
        #   time_out = episode_length_buf >= max_episode_length - 1
        # Randomizing the buffer spreads mass resets across steps.
        if random_start_init:
            cast(Any, self.envs.unwrapped).episode_length_buf = torch.randint_like(
                cast(Any, self.envs.unwrapped).episode_length_buf, high=int(self.max_episode_steps)
            )
        if self.to_numpy:
            obs = obs.cpu().numpy()
            infos = recursive_to_numpy(infos)  # type: ignore
        # actor_observation_size: policy-only shape so agent can slice concat obs.
        infos.update({"actor_observation_size": self.obs_size, "asymmetric_obs": self.asymmetric_obs})
        return obs, infos

    def step(self, actions: Union[torch.Tensor, F32NDArray]) -> tuple[
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        dict[str, Any],
    ]:
        """Step with clamp * bounds, then pack policy||critic obs and final_obs."""
        if isinstance(actions, torch.Tensor):
            torch_actions = actions.to(self.device)
        else:
            torch_actions = torch.from_numpy(actions).to(self.device)

        # Agent → env interface: keep actions in [-bounds, bounds] before managers.
        if self.action_bounds is not None:
            torch_actions = torch.clamp(torch_actions, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminations, truncations, infos = cast(Any, self.envs.step(torch_actions))
        obs = obs_dict["policy"]
        if self.asymmetric_obs:
            critic_obs = obs_dict["critic"]
            obs = torch.cat((obs, critic_obs), dim=-1)
        else:
            critic_obs = None
        infos = {"time_outs": truncations, "observations": {"critic": critic_obs}}
        # Isaac Lab does not expose terminal raw obs cleanly for all tasks;
        # use current obs as final_obs for n-step / bootstrap bookkeeping.
        # See https://github.com/isaac-sim/IsaacLab/issues/1362
        infos["final_obs"] = obs

        if self.to_numpy:
            obs = obs.cpu().numpy()
            rew = rew.cpu().numpy()
            terminations = terminations.cpu().numpy()
            truncations = truncations.cpu().numpy()
            infos = recursive_to_numpy(infos)
        return obs, rew, terminations, truncations, infos

    def close(self, **kwargs: Any) -> None:
        # Intentionally no-op: closing SimulationApp mid-process is brittle;
        # process exit tears down the app.
        del kwargs
        return

    def render(self) -> None:
        raise NotImplementedError("We don't support rendering for IsaacLab environments")


def make_isaaclab_env(
    env_name: str,
    num_envs: int,
    seed: int,
    headless: bool = True,
) -> IsaacLabVectorEnv:
    """Factory used by train / play / export.

    Looks up ``ACTION_BOUNDS`` (default 1.0). Prefer CUDA when available.
    """
    if env_name not in ACTION_BOUNDS:
        print(f"Action bounds not defined for {env_name}; using default value 1.0.")
    action_bounds = ACTION_BOUNDS.get(env_name, 1.0)
    env = IsaacLabVectorEnv(
        env_name=env_name,
        num_envs=num_envs,
        seed=seed,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        action_bounds=action_bounds,
        to_numpy=True,
        headless=headless,
    )
    return env
