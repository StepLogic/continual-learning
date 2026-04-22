#!/bin/bash
# ============================================================================
# FAME Cross-Game Benchmark: Train ONE model on all games, evaluate across all
# ============================================================================
#
# Training: Single FAME agent trained sequentially on:
#   1. Breakout (mode 0)
#   2. Freeway (modes 0-7)
#   3. SpaceInvaders (modes 0-9)
#
# Evaluation: Test final model on ALL games' modes
#   - Breakout: mode 0
#   - Freeway: modes 0-7
#   - SpaceInvaders: modes 0-9
#
# Usage: ./run_fame_cross_game.sh [seed]
#   seed: optional, default=1
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

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
# PHASE 2: EVALUATION
# ============================================================================
echo ""
echo "=============================================="
echo "PHASE 2: CROSS-GAME EVALUATION"
echo "=============================================="
echo ""

# Model to evaluate (single cross-game model)
MODEL_DIR="agents/cross_game__FAME__run_baselines_cross_game__${SEED}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model not found at ${MODEL_DIR}"
    exit 1
fi

echo "Evaluating model: ${MODEL_DIR}"
echo ""

# Output CSV (written by test_agent.py)
EVAL_CSV="eval_cross_game_seed${SEED}.csv"
# test_agent.py writes: algorithm,environment,train mode,test mode,seed,ep ret

# Games and modes
declare -A GAME_MODES
GAME_MODES["ALE/Breakout-v5"]="0"
GAME_MODES["ALE/Freeway-v5"]="0 1 2 3 4 5 6 7"
GAME_MODES["ALE/SpaceInvaders-v5"]="0 1 2 3 4 5 6 7 8 9"

ALL_GAMES=("ALE/Breakout-v5" "ALE/Freeway-v5" "ALE/SpaceInvaders-v5")

TOTAL_EVAL=0

# Clear CSV (first call writes header)
rm -f "$EVAL_CSV"

for TEST_GAME in "${ALL_GAMES[@]}"; do
    TEST_GAME_NAME=$(echo "$TEST_GAME" | sed 's/ALE\///;s/-v5//')
    MODES=(${GAME_MODES[$TEST_GAME]})

    echo "Testing on ${TEST_GAME} (${#MODES[@]} modes)..."

    for TEST_MODE in ${MODES[@]}; do
        echo -n "  Mode ${TEST_MODE}: "

        python3 test_agent.py \
            --load "$MODEL_DIR" \
            --test-env "$TEST_GAME" \
            --test-mode "$TEST_MODE" \
            --num-episodes 10 \
            --max-timesteps 1000 \
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
    python3 << EOF
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
