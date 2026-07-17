#!/bin/bash
##################################################################################
# Batch-train FlashSAC on official Isaac Lab task ids (paper-style suite).
#
# Difference vs run_unitree.sh
#   Task list is Isaac-* from isaaclab_tasks (hands, G1/H1/Anymal, Franka).
#   Obs/action layouts follow official Lab configs — not unitree deploy.yaml.
#
# Same launch caveats as run_unitree.sh: for a local binary Isaac Sim + Lab
# venv, use isaaclab.sh -p (or activated Lab python), not necessarily uv run.
#
# Defaults for buffer/AMP/envs live in configs/flashSAC_base.yaml and
# configs/agent/flashSAC.yaml; this script only overrides env_name and seed.
##################################################################################

env_names=(
    "Isaac-Repose-Cube-Shadow-Direct-v0"
    "Isaac-Repose-Cube-Allegro-Direct-v0"
    "Isaac-Velocity-Flat-G1-v0"
    "Isaac-Velocity-Rough-G1-v0"
    "Isaac-Velocity-Flat-H1-v0"
    "Isaac-Velocity-Rough-H1-v0"
    "Isaac-Lift-Cube-Franka-v0"
    "Isaac-Open-Drawer-Franka-v0"
    "Isaac-Velocity-Flat-Anymal-C-v0"
    "Isaac-Velocity-Rough-Anymal-C-v0"
    "Isaac-Velocity-Flat-Anymal-D-v0"
    "Isaac-Velocity-Rough-Anymal-D-v0"
)
seeds=( 0 1000 2000 3000 4000 )

for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "$env_name, $seed"
        uv run python train.py \
            --config_name flashSAC_base \
            --overrides seed=${seed} \
            --overrides env.env_name=${env_name} \
            --overrides agent.asymmetric_observation=true
    done
done
