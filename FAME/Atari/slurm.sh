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

# Change to project directory
cd /media/kojogyaase/disk_two/Research/continual_learning/FAME/Atari

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

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' not found. Creating..."
    conda create -n ${ENV_NAME} python=3.10 -y
    conda activate ${ENV_NAME}

    echo "Installing PyTorch with CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    echo "Installing Atari and Gymnasium dependencies..."
    pip install gymnasium[atari]==0.28.1
    pip install ale-py==0.8.1
    pip install stable-baselines3==2.0.0
    pip install tensorboard==2.11.2
    pip install opencv-python==4.7.0.72
    pip install absl-py==1.4.0
    pip install tyro==0.5.10
    pip install tqdm>=4.65.0
    pip install numpy>=1.24.0
    pip install pandas>=2.0.0
    pip install matplotlib>=3.7.0
    pip install seaborn>=0.12.0
    pip install scipy
    pip install tabulate
    pip install colormaps

    echo "Installing AutoROM with accepted license..."
    pip install autorom[accept-rom-license]==0.4.2

    echo "Conda environment setup complete!"
else
    echo "Conda environment '${ENV_NAME}' found. Activating..."
    conda activate ${ENV_NAME}
fi

echo ""
echo "Environment verification:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "============================================"

# ============================================
# Configuration
# ============================================

# Seed from SLURM array task ID or default
SEED=${SLURM_ARRAY_TASK_ID:-1}

# Timesteps per task (1M = paper default)
TIMESTEPS=1000000

# ============================================
# Run FAME Benchmark on all 3 games
# ============================================

echo ""
echo "=========================================="
echo "FAME Benchmark: All 3 Games (HPC)"
echo "=========================================="
echo "Games: Breakout (1 mode), SpaceInvaders (10 modes), Freeway (8 modes)"
echo "Seed: ${SEED}"
echo "Timesteps per task: ${TIMESTEPS}"
echo "Timestamp: $(date)"
echo "=========================================="

FAILED=0

for env in "ALE/Breakout-v5" "ALE/SpaceInvaders-v5" "ALE/Freeway-v5"; do
    game_name=$(echo "$env" | sed 's/ALE\///;s/-v5//')

    echo ""
    echo "=========================================="
    echo "Running FAME on ${env}"
    echo "=========================================="
    echo "Seed: ${SEED}"
    echo "Timesteps per task: ${TIMESTEPS}"
    echo "Timestamp: $(date)"
    echo "=========================================="
    echo ""

    # Create game-specific data directory
    mkdir -p "data_FAME/envs/${game_name}"

    # Run FAME via run_ppo_FAME.py (handles all modes sequentially)
    python3 src/train/run_ppo_FAME.py \
        --model-type=FAME \
        --env-id="${env}" \
        --seed=${SEED} \
        --save-dir=agents \
        --total-timesteps=${TIMESTEPS} \
        --epoch_meta=200 \
        --buffer_path="data_FAME/${game_name}_buffer_"

    if [ $? -ne 0 ]; then
        echo "FAME on ${env}: FAILED"
        FAILED=$((FAILED + 1))
    else
        echo "FAME on ${env}: SUCCESS"
    fi
done

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