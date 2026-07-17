# FlashSAC (IsaacLab)

FlashSAC 的 IsaacLab 训练与部署导出：离策略 SAC，面向高维机器人控制。

论文: [arXiv](https://arxiv.org/abs/2604.04539) · [项目页](https://holiday-robot.github.io/FlashSAC/)

默认目标栈：**本地 Isaac Sim 5.1 + Isaac Lab 2.3 + Python 3.11**（用已有 Lab 的 uv 环境，不走 pip 装 isaacsim）。

---

## 环境（与 unitree_rl_lab 同一套用法）

假定你已经像训 unitree 一样配好：

- Isaac Lab：`~/projects/IsaacLab`，`uv` 管理的 `.venv`
- 本地 Isaac Sim，Lab 里 `_isaac_sim` 链过去
- activate 后 venv 已带好 Sim 运行时（LD 路径等）——**不必**用 `isaaclab.sh -p`
- `unitree_rl_lab` 已装进该 venv

### 1. 激活 Lab 环境，装 FlashSAC 依赖

```bash
source ~/projects/IsaacLab/.venv/bin/activate
cd /path/to/FlashSAC

# 只装本仓库依赖；不要在本仓库 uv sync --extra isaaclab
uv pip install -e . \
  hydra-core omegaconf tqdm numpy "wandb==0.23" tensorboard pillow \
  "gymnasium>=1.1.1" "onnx>=1.16.0" "onnxruntime>=1.18.0"
```

保持 Lab 自带的 `torch`（常见 2.7.x）。

### 2. 启动（推荐，对齐你在 unitree 下的习惯）

```bash
source ~/projects/IsaacLab/.venv/bin/activate
cd /path/to/FlashSAC

# 与 unitree 下 uv python scripts/rsl_rl/train.py 同类：直接用当前 venv 的解释器
python train.py
# 或
uv run --python "$VIRTUAL_ENV/bin/python" train.py
```

说明：

- **不要**在 FlashSAC 目录裸跑 `uv run python`（未指定 `--python` 时，uv 可能另起本仓库 `.venv`，里面没有本地 Sim / unitree）。
- 已 activate 时用 `python train.py` 最简单。
- `isaaclab.sh -p` 是另一条入口，可用但不是必须。

---

## 训练

默认：`Unitree-G1-29dof-Velocity`，`asymmetric_observation=true`。

```bash
source ~/projects/IsaacLab/.venv/bin/activate
cd /path/to/FlashSAC

python train.py

python train.py \
  --overrides env.env_name='Unitree-G1-29dof-Velocity-Rough' \
  --overrides seed=1000 \
  --overrides num_train_envs=4096 \
  --overrides num_env_steps=50000896

python train.py --overrides env.env_name='Isaac-Velocity-Flat-G1-v0'
```

批量（先 activate Lab venv，再跑；脚本用 `$VIRTUAL_ENV/bin/python`）：

```bash
bash scripts/run_unitree.sh     # Unitree 任务，默认 4096 envs + ~50M steps
bash scripts/run_isaaclab.sh    # 官方 IsaacLab 任务集
```

日志：`configs/flashSAC_base.yaml` 的 `logger_type`。  
断点：`agent_load_path` / `buffer_load_path`。

可视化：

```bash
python play_isaaclab.py \
  --checkpoint_path models/.../step24400 \
  --num_envs 16 \
  --overrides env.env_name='Unitree-G1-29dof-Velocity' \
  --overrides agent.buffer_max_length=1
```

---

## 导出（actor.pt → policy.onnx + deploy.yaml）

```bash
# 仅 ONNX（可不启 Sim）
python export_policy.py \
  --checkpoint_path models/.../step24400 \
  --skip_deploy

# ONNX + deploy.yaml（需 Sim + unitree task）
python export_policy.py \
  --checkpoint_path models/.../step24400 \
  --env_name Unitree-G1-29dof-Velocity
```

输出默认在 `<checkpoint>/exported/`：`policy.onnx`、`params/deploy.yaml`、`policy_meta.json`。

---

## Sim2sim / Sim2real（unitree_rl_lab）

把导出目录放到例如：

```text
unitree_rl_lab/deploy/robots/g1_29dof/config/policy/<name>/
  exported/policy.onnx
  params/deploy.yaml
```

再按 unitree 文档：`unitree_mujoco` + `g1_ctrl`（sim2sim），或 `./g1_ctrl --network eth0`（真机）。

---

## 布局

```
train.py / play_isaaclab.py / export_policy.py
configs/
flash_rl/envs/    # IsaacLab wrapper
flash_rl/export/
scripts/
```
