# FlashSAC (IsaacLab)

FlashSAC training stack for **IsaacLab** / **unitree_rl_lab**, with export for sim2sim and real deploy.

Target stack:

| Component | Version |
|---|---|
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 |
| PyTorch | 2.9.1 |

Paper: [FlashSAC](https://arxiv.org/abs/2604.04539) · [Project page](https://holiday-robot.github.io/FlashSAC/)

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Pin Python 3.11

```bash
uv python pin 3.11.14
```

### 3. Install dependencies

```bash
uv sync --extra isaaclab
```

For Unitree tasks / `deploy.yaml` export, also install [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) into the same environment (editable install via their script).

```bash
uv sync --extra isaaclab --dev   # optional lint tools
```

## Training

Defaults target **unitree deploy**:

- env: `Unitree-G1-29dof-Velocity`
- `agent.asymmetric_observation=true` (actor uses policy obs only)

```bash
uv run python train.py
```

Override env / seed:

```bash
uv run python train.py \
    --overrides env.env_name='Unitree-G1-29dof-Velocity-Rough' \
    --overrides seed=1000
```

Official IsaacLab tasks still work:

```bash
uv run python train.py --overrides env.env_name='Isaac-Velocity-Flat-G1-v0'
```

Batch scripts:

```bash
bash scripts/run_unitree.sh    # unitree_rl_lab tasks
bash scripts/run_isaaclab.sh   # official IsaacLab locomotion set
```

### Logging

Set `logger_type` in `configs/flashSAC_base.yaml` to `wandb` or `tensorboard`. TensorBoard logs go to `runs/`:

```bash
tensorboard --logdir runs
```

## Checkpointing

```bash
uv run python train.py \
    --overrides save_checkpoint_per_interaction_step=24400 \
    --overrides save_buffer_per_interaction_step=24400
```

Resume:

```bash
uv run python train.py \
    --overrides agent_load_path='models/.../step24400' \
    --overrides buffer_load_path='models/.../step24400'
```

## Visualization (Isaac Sim)

```bash
uv run python play_isaaclab.py \
    --checkpoint_path 'models/.../step24400' \
    --num_envs 16 \
    --num_episodes 10 \
    --overrides env.env_name='Unitree-G1-29dof-Velocity' \
    --overrides agent.buffer_max_length=1
```

## Export for sim2sim / real deploy

Exports **deterministic** policy `tanh(mean)` (same as eval `temperature=0`).

ONNX only (no simulator):

```bash
uv run python export_policy.py \
    --checkpoint_path models/.../step24400 \
    --skip_deploy
```

ONNX + unitree `deploy.yaml` (needs Isaac Sim + unitree_rl_lab):

```bash
uv run python export_policy.py \
    --checkpoint_path models/.../step24400 \
    --env_name Unitree-G1-29dof-Velocity
```

Outputs under `<checkpoint>/exported/` by default:

| File | Role |
|---|---|
| `policy.onnx` | Deploy / MuJoCo / `g1_ctrl` |
| `params/deploy.yaml` | Obs layout, action scale/offset, joint map, PD |
| `policy_meta.json` | Input/action dims and paths |

Copy into unitree_rl_lab deploy layout (example):

```text
unitree_rl_lab/deploy/robots/g1_29dof/config/policy/<name>/
  exported/policy.onnx
  params/deploy.yaml
```

Then follow unitree_rl_lab sim2sim (`unitree_mujoco` + `g1_ctrl`) or sim2real.

**Requirements for a valid deploy policy:**

1. Train with `agent.asymmetric_observation=true` (default).
2. Train on the same task you pass to `--env_name` when exporting `deploy.yaml`.
3. ONNX input = policy observation only; action decoding uses `deploy.yaml` (scale/offset).

## Project layout

```
flash_rl/
  agents/           # FlashSAC agent
  buffers/
  common/
  envs/isaaclab.py  # IsaacLab + unitree task registration
  export/           # actor.pt → ONNX helpers
  evaluation.py
configs/
scripts/run_unitree.sh
scripts/run_isaaclab.sh
train.py
play_isaaclab.py
export_policy.py
```

## Development

```bash
uv sync --extra isaaclab --dev
./bin/lint
```

## Citation

```bibtex
@article{kim2026flashsac,
  title={FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control},
  author={Kim, Donghu and Lee, Youngdo and Park, Minho and Kim, Kinam and Nahendra, I Made Aswin and Seno, Takuma and Min, Sehee and Palenicek, Daniel and Vogt, Florian and Kragic, Danica and Peters, Jan and Choo, Jaegul and Lee, Hojoon},
  journal={arXiv preprint arXiv:2604.04539},
  year={2026}
}
```
