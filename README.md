# FlashSAC (IsaacLab)

FlashSAC 的 IsaacLab 训练与部署导出：离策略 SAC，面向高维机器人控制。

论文: [arXiv](https://arxiv.org/abs/2604.04539) · [项目页](https://holiday-robot.github.io/FlashSAC/)

默认目标栈：**本地 Isaac Sim 5.1 + Isaac Lab 2.3 + Python 3.11**（用已有 Lab 环境，不走 pip 安装 isaacsim）。

---

## 环境（本地已有 IsaacLab / IsaacSim）

假定：

- Isaac Lab：`~/projects/IsaacLab`（uv 管理 `.venv`）
- Isaac Sim：本地安装，Lab 内 `_isaac_sim` 指向该目录
- `unitree_rl_lab` 已装进 Lab 的 venv（可选，Unitree 任务与 `deploy.yaml` 需要）

### 1. 激活 Lab 环境并挂上 FlashSAC 依赖

```bash
source ~/projects/IsaacLab/.venv/bin/activate
cd /path/to/FlashSAC

# 只装本仓库依赖，不要 uv sync --extra isaaclab（会拉 pip isaacsim）
uv pip install -e . \
  hydra-core omegaconf tqdm numpy "wandb==0.23" tensorboard pillow \
  "gymnasium>=1.1.1" "onnx>=1.16.0" "onnxruntime>=1.18.0"
```

保持 Lab 自带的 `torch`（常见 2.7.x），不要强行升到 2.9。

### 2. 启动方式

`isaaclab.sh` 通过 `CONDA_PREFIX` 选 python。uv venv 可这样用：

```bash
source ~/projects/IsaacLab/.venv/bin/activate
export CONDA_PREFIX="$VIRTUAL_ENV"

cd /path/to/FlashSAC
~/projects/IsaacLab/isaaclab.sh -p train.py
```

或已正确配置 Sim 环境变量后，直接：

```bash
python train.py
```

---

## 训练

默认：`Unitree-G1-29dof-Velocity`，`asymmetric_observation=true`（actor 只吃 policy obs，便于部署）。

```bash
# 单次
~/projects/IsaacLab/isaaclab.sh -p train.py

# 换任务 / seed
~/projects/IsaacLab/isaaclab.sh -p train.py \
  --overrides env.env_name='Unitree-G1-29dof-Velocity-Rough' \
  --overrides seed=1000

# 官方 IsaacLab 任务
~/projects/IsaacLab/isaaclab.sh -p train.py \
  --overrides env.env_name='Isaac-Velocity-Flat-G1-v0'
```

批量（需先 activate + `CONDA_PREFIX`，并把脚本里的 `uv run python` 改成 `python`，或手动循环 `isaaclab.sh -p train.py`）：

```bash
# Unitree 任务
bash scripts/run_unitree.sh
# 官方 IsaacLab 任务集
bash scripts/run_isaaclab.sh
```

日志：`configs/flashSAC_base.yaml` 里 `logger_type: tensorboard | wandb`。  
断点：`agent_load_path` / `buffer_load_path` 指向 `models/.../stepN`。

Isaac 内可视化：

```bash
~/projects/IsaacLab/isaaclab.sh -p play_isaaclab.py \
  --checkpoint_path models/.../step24400 \
  --num_envs 16 \
  --overrides env.env_name='Unitree-G1-29dof-Velocity' \
  --overrides agent.buffer_max_length=1
```

---

## 导出（actor.pt → policy.onnx + deploy.yaml）

确定性策略：`tanh(mean)`（与 eval temperature=0 一致）。

```bash
# 仅 ONNX
python export_policy.py \
  --checkpoint_path models/.../step24400 \
  --skip_deploy

# ONNX + deploy.yaml（需 unitree 任务 + 短时起 Sim）
~/projects/IsaacLab/isaaclab.sh -p export_policy.py \
  --checkpoint_path models/.../step24400 \
  --env_name Unitree-G1-29dof-Velocity
```

默认输出 `<checkpoint>/exported/`：

| 文件 | 用途 |
|---|---|
| `policy.onnx` | sim2sim / 真机推理 |
| `params/deploy.yaml` | obs 拼装、action scale/offset、关节映射、PD |
| `policy_meta.json` | 维度与路径记录 |

前提：训练时 `asymmetric_observation=true`；导出 `--env_name` 与训练 task 一致。

---

## Sim2sim / Sim2real（unitree_rl_lab）

1. 将导出结果放进 unitree 部署目录，例如：

```text
unitree_rl_lab/deploy/robots/g1_29dof/config/policy/<name>/
  exported/policy.onnx
  params/deploy.yaml
```

2. **Sim2sim**：按 unitree_rl_lab 文档起 `unitree_mujoco`，再跑 `deploy/robots/g1_29dof/build/g1_ctrl`（站立 → 触地 → 跑 policy）。

3. **Sim2real**：关板上冲突控制后：

```bash
./g1_ctrl --network eth0
```

细节与手柄流程见 [unitree_rl_lab Deploy](https://github.com/unitreerobotics/unitree_rl_lab#deploy)。

---

## 布局

```
train.py / play_isaaclab.py / export_policy.py
configs/          # 默认 env=isaaclab → Unitree-G1-29dof-Velocity
flash_rl/envs/    # IsaacLab wrapper（可注册 unitree tasks）
flash_rl/export/  # actor → ONNX
scripts/          # 批量训练
```
