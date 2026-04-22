#!/bin/bash
# ============================================================================
# QR-DQN Cross-Game Evaluation Only
# ============================================================================
#
# Evaluate an existing QR-DQN checkpoint on ALL games and modes.
#
# Usage: ./eval_qr_dqn_cross_game.sh [seed]
#   seed: optional, default=1
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Configuration
SEED=${1:-1}
EVAL_EPISODES=${EVAL_EPISODES:-10}

# Games and modes (same as FAME)
declare -A GAME_MODES
GAME_MODES["ALE/Breakout-v5"]="0"
GAME_MODES["ALE/Freeway-v5"]="0 1 2 3 4 5 6 7"
GAME_MODES["ALE/SpaceInvaders-v5"]="0 1 2 3 4 5 6 7 8 9"

ALL_GAMES=("ALE/Breakout-v5" "ALE/Freeway-v5" "ALE/SpaceInvaders-v5")

# Find the highest task checkpoint
CHECKPOINT_DIR="checkpoints/qr_dqn"
FINAL_TASK_IDX=0
for f in "$CHECKPOINT_DIR"/qr_dqn_task*_seed${SEED}.pkl; do
    [ -f "$f" ] || continue
    task_idx=$(echo "$f" | sed -n 's/.*task\([0-9]*\)_seed.*/\1/p')
    if [ "$task_idx" -gt "$FINAL_TASK_IDX" ]; then
        FINAL_TASK_IDX=$task_idx
    fi
done

MODEL_PATH="${CHECKPOINT_DIR}/qr_dqn_task${FINAL_TASK_IDX}_seed${SEED}.pkl"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: No checkpoint found for seed ${SEED} in ${CHECKPOINT_DIR}"
    exit 1
fi

echo ""
echo "=============================================="
echo "QR-DQN Cross-Game Evaluation"
echo "=============================================="
echo "Seed: ${SEED}"
echo "Eval episodes: ${EVAL_EPISODES}"
echo "Checkpoint: ${MODEL_PATH}"
echo "=============================================="
echo ""

# Output CSV (written by eval_qr_dqn_game.py)
EVAL_CSV="results/qr_dqn/eval_cross_game_seed${SEED}.csv"
mkdir -p "results/qr_dqn"
rm -f "$EVAL_CSV"

echo ""
echo "=============================================="
echo "CROSS-GAME EVALUATION"
echo "=============================================="
echo ""

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
# SUMMARY
# ============================================================================
echo "=============================================="
echo "RESULTS SUMMARY"
echo "=============================================="

if command -v python3 &> /dev/null; then
    python3 << 'EOF'
import pandas as pd

df = pd.read_csv("${EVAL_CSV}")

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
EOF
fi

echo ""
echo "Benchmark complete!"
