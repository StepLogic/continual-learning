#!/bin/bash
# SLURM Job Script for Continual RL Benchmark
# Submit with: sbatch slurm.sh
# Or array job: sbatch --array=0-4 slurm.sh

#SBATCH --job-name=continual_rl      # Job name
#SBATCH --partition=gpu              # Partition/Queue name
#SBATCH --mail-type=END,FAIL         # Mail events
#SBATCH --mail-user=egyaase@maine.edu # Where to send mail
#SBATCH --ntasks=1                   # Run on single node
#SBATCH --cpus-per-task=8            # Run with 8 threads
#SBATCH --mem=150gb                  # Job memory request
#SBATCH --time=96:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:l40:1             # Request 1 L40 GPU
#SBATCH --output=logs/rl_%j_%A_%a.log # Standard output and error log
cd FAME/Atari
bash run_baselines_cross_game.sh all
bash run_qr_dqn_cross_game.sh
