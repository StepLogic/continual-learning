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

set -e

SCRIPT_DIR="~/continual_learning/FAME/Atari"
cd "$SCRIPT_DIR"

# Directories
mkdir -p agents logs data_FAME

# Configuration
SEED=${1:-1}
TIMESTEPS=1000000

echo ""
echo "=============================================="
echo "FAME Cross-Game Benchmark"
echo "=============================================="
echo "Seed: ${SEED}"
echo "Timesteps per task: ${TIMESTEPS}"
echo "=============================================="
echo ""

# ============================================================================
# PHASE 1: TRAINING
# ============================================================================
echo "=============================================="
echo "PHASE 1: TRAINING (Cross-Game)"
echo "=============================================="
echo ""

python3 run_baselines_cross_game.py \
    --model-type=FAME \
    --seed=${SEED} \
    --save-dir=agents \
    --total-timesteps=${TIMESTEPS} \
    --epoch_meta=200 \
    --use_ttest=1

echo ""
echo "Training complete!"
echo ""

# ============================================================================
# PHASE 2: EVALUATION
# ============================================================================
echo ""
echo "=============================================="
echo "PHASE 2: CROSS-GAME EVALUATION"
echo "=============================================="
echo ""

# Output CSV
EVAL_CSV="eval_cross_game_seed${SEED}.csv"
echo "train_game,train_mode,test_game,test_mode,avg_return" > "$EVAL_CSV"

# Model to evaluate (single cross-game model)
MODEL_DIR="agents/cross_game__FAME__run_baselines_cross_game__${SEED}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model not found at ${MODEL_DIR}"
    exit 1
fi

echo "Evaluating model: ${MODEL_DIR}"
echo ""

# Games and modes
declare -A GAME_MODES
GAME_MODES["ALE/Breakout-v5"]="0"
GAME_MODES["ALE/Freeway-v5"]="0 1 2 3 4 5 6 7"
GAME_MODES["ALE/SpaceInvaders-v5"]="0 1 2 3 4 5 6 7 8 9"

ALL_GAMES=("ALE/Breakout-v5" "ALE/Freeway-v5" "ALE/SpaceInvaders-v5")

TOTAL_EVAL=0

for TEST_GAME in "${ALL_GAMES[@]}"; do
    TEST_GAME_NAME=$(echo "$TEST_GAME" | sed 's/ALE\///;s/-v5//')
    MODES=(${GAME_MODES[$TEST_GAME]})

    echo "Testing on ${TEST_GAME} (${#MODES[@]} modes)..."

    for TEST_MODE in ${MODES[@]}; do
        echo -n "  Mode ${TEST_MODE}: "

        OUTPUT=$(python3 test_agent.py \
            --load "$MODEL_DIR" \
            --test-env "$TEST_GAME" \
            --test-mode "$TEST_MODE" \
            --num-episodes 10 \
            --max-timesteps 1000 \
            2>&1)

        AVG_RET=$(echo "$OUTPUT" | grep "Avg. episodic return:" | sed 's/.*: //')

        if [ -n "$AVG_RET" ]; then
            echo "${AVG_RET}"
            # For cross-game model, train_game is "cross_game" (all games)
            echo "cross_game,all,${TEST_GAME},${TEST_MODE},${AVG_RET}" >> "$EVAL_CSV"
            TOTAL_EVAL=$((TOTAL_EVAL + 1))
        else
            echo "FAILED"
            echo "cross_game,all,${TEST_GAME},${TEST_MODE},NaN" >> "$EVAL_CSV"
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
    python3 << EOF
import pandas as pd

df = pd.read_csv("${EVAL_CSV}")

print("\nAverage Return by Test Game:")
print("-" * 40)
for game in df['test_game'].unique():
    game_data = df[df['test_game'] == game]['avg_return']
    mean_ret = game_data.mean()
    std_ret = game_data.std()
    game_name = str(game).replace("ALE/", "").replace("-v5", "")
    print(f"  {game_name:15}: {mean_ret:7.2f} (+/- {std_ret:.2f})")

print("\nDetailed Results:")
print("-" * 40)
for _, row in df.iterrows():
    game_name = row['test_game'].replace("ALE/", "").replace("-v5", "")
    print(f"  {game_name} mode {int(row['test_mode']):2d}: {row['avg_return']:7.2f}")
EOF
fi

echo ""
echo "Benchmark complete!"
