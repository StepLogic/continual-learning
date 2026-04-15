#!/bin/bash
# SLURM Job Script for Atari FAME Benchmark (3 games)
# Games: Breakout (1 mode), SpaceInvaders (10 modes), Freeway (8 modes)
# Timesteps: 1M per task (paper default)
#
# Submit:
#   sbatch slurm.sh
#
# Array job (multiple seeds):
#   sbatch --array=1-3 slurm.sh

#SBATCH --job-name=atari_fame        # Job name
#SBATCH --partition=gpu              # Partition/Queue name
#SBATCH --mail-type=END,FAIL         # Mail events
#SBATCH --mail-user=egyaase@maine.edu # Where to send mail
#SBATCH --ntasks=1                   # Run on single node
#SBATCH --cpus-per-task=4            # Run with 4 threads
#SBATCH --mem=32gb                   # Job memory request
#SBATCH --time=96:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --output=logs/fame_%j_%A_%a.log

# Headless rendering for HPC (Atari uses pygame/SDL, disable display)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

# Create directories
mkdir -p logs agents data_FAME runs

# ============================================
# Anaconda Environment Setup
# ============================================
ENV_NAME="continual-learning"

echo "============================================"
echo "Setting up Anaconda environment"
echo "============================================"

module load anaconda3
$INIT_CONDA

echo "Conda environment '${ENV_NAME}' found. Activating..."
conda activate ${ENV_NAME}

echo ""
echo "Environment verification:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "============================================"

bash FAME/Atari/run_fame_cross_game.sh

echo ""
echo "=========================================="
echo "BENCHMARK COMPLETE"
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "All games completed successfully!"
else
    echo "${FAILED} game(s) failed. Check logs for details."
    exit 1
fi

echo ""
echo "Benchmark completed at $(date)"