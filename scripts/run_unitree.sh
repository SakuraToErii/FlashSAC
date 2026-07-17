#!/bin/bash
##################################################################################
# Batch-train FlashSAC on unitree_rl_lab tasks (deploy-oriented).
#
# 启动:
#   source ~/projects/IsaacLab/.venv/bin/activate
#   cd /path/to/FlashSAC
#   bash scripts/run_unitree.sh
#
# 使用 $VIRTUAL_ENV/bin/python（activate 后的 Lab 环境），不要裸 uv run。
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
# 本脚本默认:
#   num_train_envs=4096, updates_per_interaction_step=8
#   → UTD = 8/4096 = 2/1024（与论文相同）
#
# 其它:
#   num_env_steps ≈ 50M（FlashSAC 默认总环境步）
#   num_eval_episodes = num_train_envs（须整除 num_envs）
#   asymmetric_observation=true
#   logger_type=wandb, entity_name=null（用本地 wandb login 的默认账号）
#
# Replay buffer (agent.buffer_max_length, device=cuda):
#   论文默认 10_000_000。
#   TorchUniformBuffer 预分配 (N, obs_dim) 的 obs + next_obs。
#   unitree policy history_length=5 时 obs_dim≈480（仅 policy 前缀量级）；
#   若 buffer 存 policy∥critic 会更大。
#   粗算 float32、仅 obs+next_obs:
#     N=10M, dim=480 → ~38 GB  → 单卡 4090(24G) 不够
#     N=2M,  dim=480 → ~7.7 GB → 与仿真并存通常可行
#     N=1M,  dim=480 → ~3.8 GB → 更保守
#   本脚本默认 2M（仍放 GPU，不改 buffer_device_type）。
# ---------------------------------------------------------------------------
##################################################################################

set -euo pipefail

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
else
  echo "[run_unitree] 请先: source ~/projects/IsaacLab/.venv/bin/activate" >&2
  echo "[run_unitree] 当前未检测到 VIRTUAL_ENV，将回退到 PATH 中的 python。" >&2
  PYTHON="python"
fi

env_names=(
    "Unitree-G1-29dof-Velocity"
    # "Unitree-G1-29dof-Velocity-Rough"
)
seeds=( 42 )

# --- 规模与 UTD（见文件头注释）---
# 论文默认: NUM_TRAIN_ENVS=1024, UPDATES_PER_INTERACTION_STEP=2  →  2/1024
NUM_TRAIN_ENVS=1024
UPDATES_PER_INTERACTION_STEP=2   # 8/4096 = 2/1024

NUM_ENV_STEPS=50000896           # 50_000_896 ≈ 50M env steps
NUM_EVAL_EPISODES=${NUM_TRAIN_ENVS}

# 论文 10_000_000；单卡 4090 + unitree hist5 建议 1e6~2e6（见文件头）
BUFFER_MAX_LENGTH=2000000

echo "[run_unitree] python=${PYTHON}"
echo "[run_unitree] envs=${NUM_TRAIN_ENVS}, updates/interaction=${UPDATES_PER_INTERACTION_STEP}, UTD=${UPDATES_PER_INTERACTION_STEP}/${NUM_TRAIN_ENVS}"
echo "[run_unitree] buffer_max_length=${BUFFER_MAX_LENGTH} (cuda)"

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
            --overrides group_name=unitree \
            --overrides exp_name=flashsac
    done
done

