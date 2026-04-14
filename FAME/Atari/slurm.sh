#!/bin/bash
# SLURM Job Script for Atari FAME Benchmark
# Submit with: sbatch slurm.sh
# Or array job: sbatch --array=0-4 slurm.sh

#SBATCH --job-name=atari_fame        # Job name
#SBATCH --partition=gpu              # Partition/Queue name
#SBATCH --mail-type=END,FAIL         # Mail events
#SBATCH --mail-user=egyaase@maine.edu # Where to send mail
#SBATCH --ntasks=1                   # Run on single node
#SBATCH --cpus-per-task=4            # Run with 4 threads (RTX 4050 laptop GPU)
#SBATCH --mem=16gb                   # Job memory request (16GB fits 22GB system RAM)
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                 # Request 1 GPU (RTX 4050 6GB)
#SBATCH --output=logs/fame_%j_%A_%a.log # Standard output and error log

# Set headless rendering for HPC
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Change to project directory
cd /media/kojogyaase/disk_two/Research/continual_learning/FAME/Atari

# Create logs directory
mkdir -p logs

# ============================================
# Anaconda Environment Setup
# ============================================
ENV_NAME="continual-learning"

echo "============================================"
echo "Setting up Anaconda environment"
echo "============================================"

# Load anaconda module
module load anaconda3

# Initialize conda for this shell
$INIT_CONDA

# Check if environment exists, create if not
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' not found. Creating..."

    # Create conda environment with Python 3.10
    conda create -n ${ENV_NAME} python=3.10 -y

    # Activate environment
    conda activate ${ENV_NAME}

    # Install PyTorch with CUDA support
    echo "Installing PyTorch with CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Install Atari/Gymnasium dependencies
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

    # Install ROMs
    echo "Installing AutoROM with accepted license..."
    pip install autorom[accept-rom-license]==0.4.2

    echo "Conda environment setup complete!"
else
    echo "Conda environment '${ENV_NAME}' found. Activating..."
    conda activate ${ENV_NAME}
fi

# Verify environment
echo ""
echo "Environment verification:"
echo "  Python version: $(python --version)"
echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "============================================"

# ============================================
# Run Atari FAME Benchmark
# ============================================

# Get seed from SLURM array task ID or use default
SEED=${SLURM_ARRAY_TASK_ID:-1}

# Algorithm to run (can be passed as argument)
ALGORITHM=${1:-fame}

# Environment to run (can be passed as argument)
ENV=${2:-ALE/Freeway-v5}

# Total timesteps per task (use 100000 for quick tests, 1000000 for full runs)
TIMESTEPS=${3:-100000}

echo ""
echo "============================================"
echo "Starting Atari FAME Benchmark (HPC)"
echo "============================================"
echo "Algorithm: ${ALGORITHM}"
echo "Environment: ${ENV}"
echo "Seed: ${SEED}"
echo "Timesteps per task: ${TIMESTEPS}"
echo "MUJOCO_GL: ${MUJOCO_GL}"
echo "Timestamp: $(date)"
echo "============================================"
echo ""

# Run the benchmark
python run_experiments.py \
    --algorithm "${ALGORITHM}" \
    --env "${ENV}" \
    --seed ${SEED} \
    --timesteps ${TIMESTEPS} \
    --start-mode 0

echo ""
echo "============================================"
echo "Benchmark completed at $(date)"
echo "============================================"

# ============================================
# PIP Install Commands Reference
# ============================================
# If you need to manually install dependencies, run:
#
#   module load anaconda3
#   source $(conda info --base)/etc/profile.d/conda.sh
#   conda activate continual-learning
#
#   # PyTorch with CUDA
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#
#   # Atari/Gymnasium
#   pip install gymnasium[atari]==0.28.1 ale-py==0.8.1
#   pip install stable-baselines3==2.0.0
#   pip install tensorboard==2.11.2
#   pip install opencv-python==4.7.0.72
#
#   # ROMs
#   pip install autorom[accept-rom-license]==0.4.2
#
#   # Utilities
#   pip install absl-py==1.4.0 tyro==0.5.10 tqdm>=4.65.0
#   pip install numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 seaborn>=0.12.0
#
# ============================================
