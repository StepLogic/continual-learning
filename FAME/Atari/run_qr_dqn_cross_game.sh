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

#SBATCH --job-name=atari_baselines        # Job name
#SBATCH --partition=gpu              # Partition/Queue name
#SBATCH --mail-type=END,FAIL         # Mail events
#SBATCH --mail-user=egyaase@maine.edu # Where to send mail
#SBATCH --ntasks=1                   # Run on single node
#SBATCH --cpus-per-task=4            # Run with 4 threads
#SBATCH --mem=32gb                   # Job memory request
#SBATCH --time=96:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:a30:1             # Request 1 GPU
#SBATCH --output=logs/fame_%j_%A_%a.log

# ============================================================================
# QR-DQN Cross-Game Benchmark: Train ONE model on all games, evaluate across all
# ============================================================================
#
# Training: Single QR-DQN agent trained sequentially on:
#   1. Breakout (mode 0)
#   2. Freeway (modes 0-7)
#   3. SpaceInvaders (modes 0-9)
#
# Evaluation: Test final model on ALL games' modes
#   - Breakout: mode 0
#   - Freeway: modes 0-7
#   - SpaceInvaders: modes 0-9
#
# Usage: ./run_qr_dqn_cross_game.sh [seed]
#   seed: optional, default=1
# ============================================================================

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


set -e

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="/home/egyaase/continual-learning/FAME/Atari"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Directories
# mkdir -p checkpoints/qr_dqn results/qr_dqn

# Configuration
SEED=${1:-1}
STEPS_PER_TASK=${STEPS_PER_TASK:-1000000}
EVAL_EPISODES=${EVAL_EPISODES:-10}

# Games and modes (same as FAME)
declare -A GAME_MODES
GAME_MODES["ALE/Breakout-v5"]="0"
GAME_MODES["ALE/Freeway-v5"]="0 1 2 3 4 5 6 7"
GAME_MODES["ALE/SpaceInvaders-v5"]="0 1 2 3 4 5 6 7 8 9"

ALL_GAMES=("ALE/Breakout-v5" "ALE/Freeway-v5" "ALE/SpaceInvaders-v5")

# Build --games argument for training (games only, no modes)
TRAIN_GAMES=()
for g in "${ALL_GAMES[@]}"; do
    TRAIN_GAMES+=("$g")
done

echo ""
echo "=============================================="
echo "QR-DQN Cross-Game Benchmark"
echo "=============================================="
echo "Seed: ${SEED}"
echo "Steps per task: ${STEPS_PER_TASK}"
echo "Eval episodes: ${EVAL_EPISODES}"
echo "=============================================="
echo ""

# ============================================================================
# PHASE 1: TRAINING
# ============================================================================
echo "=============================================="
echo "PHASE 1: TRAINING (Cross-Game)"
echo "=============================================="
echo ""

python3 run_qr_dqn_continual.py \
    --games "${TRAIN_GAMES[@]}" \
    --steps-per-task ${STEPS_PER_TASK} \
    --eval-interval 25000 \
    --eval-episodes ${EVAL_EPISODES} \
    --seed ${SEED} \
    --checkpoint-dir checkpoints/qr_dqn \
    --results-dir results/qr_dqn

echo ""
echo "Training complete!"
echo ""

# ============================================================================
# PHASE 2: CROSS-GAME EVALUATION
# ============================================================================
echo ""
echo "=============================================="
echo "PHASE 2: CROSS-GAME EVALUATION"
echo "=============================================="
echo ""

# Find the highest task checkpoint
FINAL_TASK_IDX=$(( ${#ALL_GAMES[@]} - 1 ))
MODEL_PATH="checkpoints/qr_dqn/qr_dqn_task${FINAL_TASK_IDX}_seed${SEED}.pkl"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Checkpoint not found at ${MODEL_PATH}"
    exit 1
fi

echo "Evaluating checkpoint: ${MODEL_PATH}"
echo ""

# Output CSV (written by eval_qr_dqn_game.py)
EVAL_CSV="FAME/Atari/eval_qr_dqn_cross_game_seed${SEED}.csv"
rm -f "$EVAL_CSV"

TOTAL_EVAL=0

for TEST_GAME in "${ALL_GAMES[@]}"; do
    TEST_GAME_NAME=$(echo "$TEST_GAME" | sed 's/ALE\///;s/-v5//')
    MODES=(${GAME_MODES[$TEST_GAME]})

    echo "Testing on ${TEST_GAME} (${#MODES[@]} modes)..."

    for TEST_MODE in ${MODES[@]}; do
        echo -n "  Mode ${TEST_MODE}: "

        python3 experiments/eval_qr_dqn_game.py \
            --load "$MODEL_PATH" \
            --game "$TEST_GAME" \
            --mode "$TEST_MODE" \
            --num-episodes ${EVAL_EPISODES} \
            --seed ${SEED} \
            --csv "$EVAL_CSV"

        # Compute mean from CSV for this (game, mode)
        AVG_RET=$(python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$EVAL_CSV')
    df = df[df['environment'] == '$TEST_GAME']
    df = df[df['test mode'] == $TEST_MODE]
    if len(df) == 0:
        sys.exit(1)
    print(f'{df['ep ret'].mean():.2f}')
except Exception:
    sys.exit(1)
")

        if [ -n "$AVG_RET" ]; then
            echo "${AVG_RET}"
            TOTAL_EVAL=$((TOTAL_EVAL + 1))
        else
            echo "FAILED"
        fi
    done
done

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "=============================================="
echo "Total evaluations: ${TOTAL_EVAL}"
echo "Results saved to: ${EVAL_CSV}"
echo ""

# ============================================================================
# PHASE 3: SUMMARY
# ============================================================================
echo "=============================================="
echo "RESULTS SUMMARY"
echo "=============================================="

if command -v python3 &> /dev/null; then
    python3 << 'PYEOF'
import pandas as pd

eval_csv = "${EVAL_CSV}"
df = pd.read_csv(eval_csv)

print("\nAverage Return by Test Game:")
print("-" * 40)
for game in df['environment'].unique():
    game_data = df[df['environment'] == game]['ep ret']
    mean_ret = game_data.mean()
    std_ret = game_data.std()
    game_name = str(game).replace("ALE/", "").replace("-v5", "")
    print(f"  {game_name:15}: {mean_ret:7.2f} (+/- {std_ret:.2f})")

print("\nDetailed Results (Mean Return per Mode):")
print("-" * 40)
summary = df.groupby(['environment', 'test mode'])['ep ret'].mean().reset_index()
for _, row in summary.iterrows():
    game_name = str(row['environment']).replace("ALE/", "").replace("-v5", "")
    print(f"  {game_name} mode {int(row['test mode']):2d}: {row['ep ret']:7.2f}")
PYEOF
fi

echo ""
echo "Benchmark complete!"