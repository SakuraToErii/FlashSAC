"""Export FlashSAC actor.pt to policy.onnx (+ deploy.yaml for unitree_rl_lab).

Examples
--------
ONNX only (no Isaac Sim)::

    uv run python export_policy.py \\
        --checkpoint_path models/.../step24400

ONNX + deploy.yaml (requires Isaac Sim + unitree_rl_lab for Unitree tasks)::

    uv run python export_policy.py \\
        --checkpoint_path models/.../step24400 \\
        --env_name Unitree-G1-29dof-Velocity
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _export_onnx_only(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from flash_rl.export.actor_export import (
        export_actor_onnx,
        export_actor_torchscript,
        load_flashsac_actor,
    )

    actor_path = os.path.join(args.checkpoint_path, "actor.pt")
    if not os.path.isfile(actor_path):
        raise FileNotFoundError(f"actor.pt not found at {actor_path}")

    output_dir = args.output_dir or os.path.join(args.checkpoint_path, "exported")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)
    actor, dims = load_flashsac_actor(actor_path, device=device)
    print(f"[export] actor dims: {dims}")

    onnx_path = os.path.join(output_dir, "policy.onnx")
    export_actor_onnx(
        actor,
        output_path=onnx_path,
        input_dim=dims["input_dim"],
        opset_version=args.opset,
        device=device,
    )
    print(f"[export] wrote {onnx_path}")

    if args.torchscript:
        pt_path = os.path.join(output_dir, "policy.pt")
        export_actor_torchscript(actor, pt_path, input_dim=dims["input_dim"], device=device)
        print(f"[export] wrote {pt_path}")

    meta = {
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "actor_path": os.path.abspath(actor_path),
        "policy_onnx": os.path.abspath(onnx_path),
        **dims,
        "action": "tanh(mean)  # FlashSAC deterministic eval (temperature=0)",
        "asymmetric_observation_required": True,
        "note": "ONNX input is policy obs only when trained with asymmetric_observation=true",
    }
    meta_path = os.path.join(output_dir, "policy_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] wrote {meta_path}")
    return meta


def _export_deploy_yaml(args: argparse.Namespace, output_dir: str) -> str:
    """Launch Isaac Sim briefly and write params/deploy.yaml via unitree_rl_lab."""
    try:
        from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
    except ImportError as exc:
        raise ImportError(
            "unitree_rl_lab is required for deploy.yaml export. "
            "Install unitree_rl_lab into this environment, or pass --skip_deploy."
        ) from exc

    from flash_rl.envs.isaaclab import make_isaaclab_env

    env_name = args.env_name
    if env_name is None:
        raise ValueError("--env_name is required when exporting deploy.yaml (or pass --skip_deploy)")

    env = make_isaaclab_env(
        env_name=env_name,
        num_envs=args.num_envs,
        seed=args.seed,
        headless=True,
    )
    try:
        # export_deploy_cfg writes <log_dir>/params/deploy.yaml
        export_deploy_cfg(env.envs.unwrapped, output_dir)
        deploy_path = os.path.join(output_dir, "params", "deploy.yaml")
        if not os.path.isfile(deploy_path):
            raise RuntimeError(f"export_deploy_cfg finished but {deploy_path} is missing")
        print(f"[export] wrote {deploy_path}")
        return deploy_path
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FlashSAC actor.pt to ONNX (+ deploy.yaml)")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Directory containing actor.pt (e.g. models/.../step24400)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <checkpoint_path>/exported)",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="IsaacLab / unitree task id for deploy.yaml (e.g. Unitree-G1-29dof-Velocity)",
    )
    parser.add_argument(
        "--skip_deploy",
        action="store_true",
        help="Only export ONNX; do not launch Isaac Sim for deploy.yaml",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Envs for deploy.yaml extraction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--torchscript",
        action="store_true",
        help="Also export TorchScript policy.pt",
    )
    args = parser.parse_args()

    meta = _export_onnx_only(args)
    output_dir = args.output_dir or os.path.join(args.checkpoint_path, "exported")

    if not args.skip_deploy:
        if args.env_name is None:
            print(
                "[export] --env_name not set; skipping deploy.yaml. "
                "Pass --env_name Unitree-G1-29dof-Velocity or --skip_deploy.",
                file=sys.stderr,
            )
        else:
            deploy_path = _export_deploy_yaml(args, output_dir)
            meta["deploy_yaml"] = os.path.abspath(deploy_path)
            meta["env_name"] = args.env_name
            meta_path = os.path.join(output_dir, "policy_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"[export] updated {meta_path}")

    print("[export] done")


if __name__ == "__main__":
    main()
