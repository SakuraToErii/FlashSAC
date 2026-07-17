#!/bin/bash
##################################################################################
# unitree_rl_lab tasks with FlashSAC (asymmetric obs for deploy)
# Requires: uv sync --extra isaaclab, and unitree_rl_lab installed in the env.
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
