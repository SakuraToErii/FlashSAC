#!/bin/bash
##################################################################################
# Batch-train FlashSAC on unitree_rl_lab tasks (deploy-oriented).
#
# What this does
#   Nested loops over env_names × seeds, each calling train.py with
#   asymmetric_observation=true so the actor is deployable (policy obs only).
#
# Prerequisites
#   - Active env has Isaac Lab + local Isaac Sim + unitree_rl_lab.
#   - Prefer IsaacLab .venv, not FlashSAC's pip isaacsim extra.
#
# Local IsaacLab launch (recommended)
#   source ~/projects/IsaacLab/.venv/bin/activate
#   export CONDA_PREFIX="$VIRTUAL_ENV"
#   Then either replace "uv run python" below with:
#     ~/projects/IsaacLab/isaaclab.sh -p
#   or run the same overrides manually with isaaclab.sh -p train.py ...
#
# Note: bare "uv run python" uses FlashSAC's project .venv if present.
##################################################################################

env_names=(
    "Unitree-G1-29dof-Velocity"
    "Unitree-G1-29dof-Velocity-Rough"
)
seeds=( 0 1000 2000 )

for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "$env_name, $seed"
        uv run python train.py \
            --config_name flashSAC_base \
            --overrides seed=${seed} \
            --overrides env.env_name=${env_name} \
            --overrides agent.asymmetric_observation=true \
            --overrides group_name=unitree \
            --overrides exp_name=flashsac
    done
done
