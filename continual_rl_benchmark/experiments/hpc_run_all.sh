#!/bin/bash
# HPC Job Script for Continual RL Benchmark
# Submit with: sbatch hpc_run_all.sh

#SBATCH --job-name=continual_rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

# Set headless rendering for HPC
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Load modules (adjust for your HPC)
# module load python/3.10 cuda/11.8

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Run the benchmark
# Usage examples:
#   sbatch hpc_run_all.sh --method base --seeds 0 1 2
#   sbatch hpc_run_all.sh --method all --steps-per-task 400000

python experiments/hpc_run_all.py \
    --method "${1:-all}" \
    --env continual_world \
    --seeds ${SLURM_ARRAY_TASK_ID:-0} \
    --steps-per-task 400000 \
    --eval-interval 10000 \
    --eval-episodes 10 \
    --checkpoint-dir checkpoints \
    --output-dir results \
    "${@:2}"
