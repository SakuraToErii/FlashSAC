"""Load FlashSAC ``actor.pt`` and export a deterministic policy for deploy.

Deploy contract (unitree_rl_lab / g1_ctrl / MuJoCo sim2sim)
------------------------------------------------------------
Input
    Flat float32 observation of size ``input_dim`` — must be the **policy**
    group only. That requires training with ``asymmetric_observation=true``
    so the actor was never fed privileged critic features.

Output
    Action in roughly ``[-1, 1]`` as ``tanh(mean)``. Same as FlashSAC eval
    with ``temperature=0`` (no exploration noise). Downstream controllers
    apply JointPositionAction scale/offset from ``deploy.yaml``.

Checkpoint format
    ``Network.save`` writes::

        {
          "network_state_dict": ...,
          "optimizer_state_dict": ...,
          "scheduler_state_dict": ...,
          "update_step": ...,
        }

    Architecture (blocks / dims) is inferred from weight shapes so export
    does not need the original Hydra config.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from flash_rl.agents.flashSAC.network import FlashSACActor


def _unwrap_state_dict(ckpt: dict[str, Any] | Any) -> dict[str, torch.Tensor]:
    """Accept FlashSAC Network checkpoints or a raw state_dict."""
    if isinstance(ckpt, dict) and "network_state_dict" in ckpt:
        return ckpt["network_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt  # type: ignore[return-value]
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def infer_actor_dims(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """Infer FlashSACActor sizes from parameter tensor shapes / key names.

    - ``embedder.w.w.weight``: [hidden_dim, input_dim]
    - ``predictor.mean_w.w.weight``: [action_dim, hidden_dim]
    - ``encoder.{i}.w1.w.weight``: block count = max(i) + 1
    """
    embed_w = state_dict["embedder.w.w.weight"]
    hidden_dim, input_dim = int(embed_w.shape[0]), int(embed_w.shape[1])
    mean_w = state_dict["predictor.mean_w.w.weight"]
    action_dim = int(mean_w.shape[0])
    block_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("encoder.") and key.endswith(".w1.w.weight")
    }
    if not block_ids:
        raise ValueError("Could not find encoder blocks in actor state_dict")
    num_blocks = max(block_ids) + 1
    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "action_dim": action_dim,
        "num_blocks": num_blocks,
    }


def load_flashsac_actor(
    actor_path: str,
    device: str | torch.device = "cpu",
) -> tuple[FlashSACActor, dict[str, int]]:
    """Rebuild ``FlashSACActor``, load weights, set eval mode."""
    ckpt = torch.load(actor_path, map_location=device, weights_only=False)
    state_dict = _unwrap_state_dict(ckpt)
    dims = infer_actor_dims(state_dict)
    actor = FlashSACActor(
        num_blocks=dims["num_blocks"],
        input_dim=dims["input_dim"],
        hidden_dim=dims["hidden_dim"],
        action_dim=dims["action_dim"],
    )
    actor.load_state_dict(state_dict)
    actor.to(device)
    actor.eval()
    return actor, dims


class DeterministicFlashSACActor(nn.Module):
    """Deploy wrapper: ``obs -> tanh(mean)`` with BatchNorm in eval mode.

    Matches ``_sample_flashsac_actions(..., temperature=0.0)`` in the agent.
    Std / entropy sampling is discarded for ONNX and controllers.
    """

    def __init__(self, actor: FlashSACActor):
        super().__init__()
        self.actor = actor

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # training=False freezes UnitBatchNorm running stats (inference path).
        mean, _std = self.actor.get_mean_and_std(observations, training=False)
        return torch.tanh(mean)


def export_actor_onnx(
    actor: FlashSACActor,
    output_path: str,
    input_dim: int,
    opset_version: int = 18,
    device: str | torch.device = "cpu",
) -> str:
    """Trace deterministic actor to ONNX (inputs: ``obs``, outputs: ``actions``).

    Uses the TorchScript ONNX exporter (``dynamo=False``) for broader
    runtime compatibility (e.g. unitree onnxruntime builds). Opset 18 matches
    Isaac Lab / unitree export conventions.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    policy = DeterministicFlashSACActor(actor).to(device)
    policy.eval()
    dummy = torch.zeros(1, input_dim, dtype=torch.float32, device=device)
    with torch.no_grad():
        torch.onnx.export(
            policy,
            dummy,
            output_path,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=opset_version,
            dynamo=False,
        )
    return output_path


def export_actor_torchscript(
    actor: FlashSACActor,
    output_path: str,
    input_dim: int,
    device: str | torch.device = "cpu",
) -> str:
    """Optional TorchScript export (same deterministic forward as ONNX)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    policy = DeterministicFlashSACActor(actor).to(device)
    policy.eval()
    dummy = torch.zeros(1, input_dim, dtype=torch.float32, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(policy, dummy)
    traced.save(output_path)
    return output_path
