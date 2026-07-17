# FlashSAC (IsaacLab)

FlashSAC training stack for **IsaacLab** only.

Target stack:

| Component | Version |
|---|---|
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 |
| PyTorch | 2.9.1 |

Paper: [FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control](https://arxiv.org/abs/2604.04539) · [Project page](https://holiday-robot.github.io/FlashSAC/)

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

Core package + IsaacLab:

```bash
uv sync --extra isaaclab
```

Dev tools (lint/typecheck):

```bash
uv sync --extra isaaclab --dev
```

## Training

Default config is IsaacLab (`Isaac-Velocity-Flat-G1-v0`, 1024 envs, GPU buffer, AMP):

```bash
uv run python train.py
```

Override env / seed:

```bash
uv run python train.py \
    --overrides env.env_name='Isaac-Velocity-Rough-G1-v0' \
    --overrides seed=1000
```

Batch benchmark:

```bash
bash scripts/run_isaaclab.sh
```

### Logging

Set `logger_type` in `configs/flashSAC_base.yaml` to `wandb` or `tensorboard`. TensorBoard logs go to `runs/`:

```bash
tensorboard --logdir runs
```

## Checkpointing

Save at intervals:

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

## Visualization

```bash
uv run python play_isaaclab.py \
    --checkpoint_path 'models/.../step24400' \
    --num_envs 16 \
    --num_episodes 10 \
    --overrides env.env_name='Isaac-Velocity-Flat-G1-v0' \
    --overrides agent.asymmetric_observation=true \
    --overrides agent.buffer_max_length=1
```

## Project layout

```
flash_rl/
  agents/          # FlashSAC agent
  buffers/         # Replay buffers
  common/          # Logger
  envs/isaaclab.py # IsaacLab Gymnasium wrapper
  evaluation.py
configs/           # Hydra configs (default: isaaclab)
scripts/run_isaaclab.sh
train.py
play_isaaclab.py
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
