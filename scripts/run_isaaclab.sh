#!/bin/bash
##################################################################################
# Batch-train FlashSAC on official Isaac Lab task ids (paper-style suite).
#
# 与 run_unitree.sh 的区别
#   - 任务来自 isaaclab_tasks 的 Isaac-*（非 unitree_rl_lab / 非 deploy 契约）
#   - 观测一般无 history_length=5，buffer 显存压力通常小于 unitree 任务
#   - 用于复现论文 GPU 仿真设定 / 验证 FlashSAC 栈，而不是 g1_ctrl 部署
#
# 启动（与 unitree 脚本相同习惯）:
#   source ~/projects/IsaacLab/.venv/bin/activate
#   cd /path/to/FlashSAC
#   bash scripts/run_isaaclab.sh
#
# 使用 $VIRTUAL_ENV/bin/python，不要在本仓库裸 uv run（会落到无 Sim 的 .venv）。
#
# ---------------------------------------------------------------------------
# 并行环境数 × 每交互更新次数 与 UTD（updates-to-data）
#
# 定义（与 train.py / Hydra 一致）:
#   一次 interaction 收集 num_train_envs 条 transition
#   并做 updates_per_interaction_step 次梯度更新
#   UTD = updates_per_interaction_step / num_train_envs
#
# 论文 FlashSAC GPU 默认比例:
#   num_train_envs=1024, updates_per_interaction_step=2
#   → UTD = 2/1024
#
# 若改为 4096 envs 且保持同一 UTD:
#   updates_per_interaction_step=8  →  8/4096 = 2/1024
#
# 其它默认:
#   num_env_steps ≈ 50M
#   num_eval_episodes = num_train_envs（evaluate 要求整除 num_envs）
#   asymmetric_observation=true（G1 官方仅有 policy 组时无影响；有 critic 的任务会切 policy）
#   logger_type=wandb, entity_name=null
#
# Replay buffer (agent.buffer_max_length, device=cuda):
#   论文默认 10_000_000。
#   Isaac 官方 locomotion 多无 hist5，同 N 下通常比 unitree 更省显存。
#   单卡 4090 可先试 10M；OOM 再降到 2M/1M。
#   本脚本默认 10M（对齐论文）；与 run_unitree.sh 的 2M 不同。
# ---------------------------------------------------------------------------
##################################################################################

set -euo pipefail

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
else
  echo "[run_isaaclab] 请先: source ~/projects/IsaacLab/.venv/bin/activate" >&2
  echo "[run_isaaclab] 当前未检测到 VIRTUAL_ENV，将回退到 PATH 中的 python。" >&2
  PYTHON="python"
fi

# 按需注释/取消注释。建议先跑 G1 Flat 验证栈，再开全量。
env_names=(
    "Isaac-Velocity-Flat-G1-v0"
    # "Isaac-Velocity-Rough-G1-v0"
    # "Isaac-Velocity-Flat-H1-v0"
    # "Isaac-Velocity-Rough-H1-v0"
    # "Isaac-Velocity-Flat-Anymal-C-v0"
    # "Isaac-Velocity-Rough-Anymal-C-v0"
    # "Isaac-Velocity-Flat-Anymal-D-v0"
    # "Isaac-Velocity-Rough-Anymal-D-v0"
    # "Isaac-Lift-Cube-Franka-v0"
    # "Isaac-Open-Drawer-Franka-v0"
    # "Isaac-Repose-Cube-Allegro-Direct-v0"
    # "Isaac-Repose-Cube-Shadow-Direct-v0"
)
seeds=( 42 )

# --- 规模与 UTD（见文件头注释）---
# 论文默认: 1024 envs, updates=2 → UTD = 2/1024
NUM_TRAIN_ENVS=1024
UPDATES_PER_INTERACTION_STEP=2

NUM_ENV_STEPS=50000896           # 50_000_896 ≈ 50M env steps
NUM_EVAL_EPISODES=${NUM_TRAIN_ENVS}

# 论文 10M；OOM 时改为 2000000 / 1000000
BUFFER_MAX_LENGTH=10000000

# 可选：收探索（unitree 步态上常有用；Isaac 官方任务一般保持默认即可）
# TEMP_TARGET_SIGMA=0.15
# ACTOR_NOISE_ZETA_MAX=16

echo "[run_isaaclab] python=${PYTHON}"
echo "[run_isaaclab] envs=${NUM_TRAIN_ENVS}, updates/interaction=${UPDATES_PER_INTERACTION_STEP}, UTD=${UPDATES_PER_INTERACTION_STEP}/${NUM_TRAIN_ENVS}"
echo "[run_isaaclab] buffer_max_length=${BUFFER_MAX_LENGTH} (cuda)"

for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "$env_name, $seed (env_steps=${NUM_ENV_STEPS})"
        "${PYTHON}" train.py \
            --config_name flashSAC_base \
            --overrides seed=${seed} \
            --overrides env.env_name=${env_name} \
            --overrides num_train_envs=${NUM_TRAIN_ENVS} \
            --overrides updates_per_interaction_step=${UPDATES_PER_INTERACTION_STEP} \
            --overrides num_env_steps=${NUM_ENV_STEPS} \
            --overrides num_eval_episodes=${NUM_EVAL_EPISODES} \
            --overrides agent.buffer_max_length=${BUFFER_MAX_LENGTH} \
            --overrides agent.asymmetric_observation=true \
            --overrides logger_type=wandb \
            --overrides project_name=FlashSAC \
            --overrides entity_name=null \
            --overrides group_name=isaaclab \
            --overrides exp_name=flashsac
    done
done
