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

# Set headless rendering for HPC
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Change to project directory
cd /media/kojogyaase/disk_two/Research/continual_learning/continual_rl_benchmark

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

    # Install core dependencies
    echo "Installing core dependencies..."
    pip install gymnasium>=0.29.0
    pip install numpy>=1.24.0
    pip install tensorboard>=2.14.0
    pip install tqdm>=4.65.0
    pip install hydra-core>=1.3.0
    pip install omegaconf>=2.3.0
    pip install pandas>=2.0.0
    pip install matplotlib>=3.7.0
    pip install seaborn>=0.12.0
    pip install pytest>=7.0.0

    # Install Mujoco
    echo "Installing Mujoco..."
    pip install mujoco>=3.0.0
    pip install mujoco-py>=2.1.0

    # Install MetaWorld for environments
    echo "Installing MetaWorld environments..."
    pip install git+https://github.com/Farama-Foundation/MetaWorld.git

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
# Run Continual RL Benchmark
# ============================================

# Get seed from SLURM array task ID or use default
SEED=${SLURM_ARRAY_TASK_ID:-0}

# Method to run (can be passed as argument)
METHOD=${1:-all}

# HPC Configuration: 5M steps per task, 1M replay buffer
STEPS_PER_TASK=5000000       # 5M steps per task for thorough training
TOTAL_STEPS=50000000         # 10 tasks * 5M = 50M total steps

echo ""
echo "============================================"
echo "Starting Continual RL Benchmark (HPC)"
echo "============================================"
echo "Method: ${METHOD}"
echo "Seed: ${SEED}"
echo "Steps per task: ${STEPS_PER_TASK} (5M)"
echo "Total steps: ${TOTAL_STEPS} (50M)"
echo "Replay buffer: 1,000,000 transitions"
echo "MUJOCO_GL: ${MUJOCO_GL}"
echo "Timestamp: $(date)"
echo "============================================"
echo ""

# Run the HPC benchmark
python experiments/hpc_run_all.py \
    --method "${METHOD}" \
    --env continual_world \
    --seed ${SEED} \
    --steps-per-task ${STEPS_PER_TASK} \
    --eval-interval 25000 \
    --eval-episodes 10 \
    --checkpoint-dir checkpoints \
    --output-dir results \
    --render false

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
#   # Core dependencies
#   pip install gymnasium>=0.29.0 numpy>=1.24.0 tensorboard>=2.14.0 tqdm>=4.65.0
#   pip install hydra-core>=1.3.0 omegaconf>=2.3.0 pandas>=2.0.0
#   pip install matplotlib>=3.7.0 seaborn>=0.12.0 pytest>=7.0.0
#
#   # Mujoco
#   pip install mujoco>=3.0.0 mujoco-py>=2.1.0
#
#   # MetaWorld environments
#   pip install git+https://github.com/Farama-Foundation/MetaWorld.git
#
# ============================================
