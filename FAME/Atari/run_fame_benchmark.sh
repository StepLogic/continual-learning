#!/bin/bash
# ============================================================================
# FAME Benchmark: Train on all games, Evaluate across all games
# ============================================================================
#
# Training: Trains FAME on all modes of each game
#   - Breakout: mode 0 (1 mode)
#   - Freeway: modes 0-7 (8 modes)
#   - SpaceInvaders: modes 0-9 (10 modes)
#
# Evaluation: Tests each trained model on ALL games' modes
#   - Within-game transfer: train Freeway-3, test Freeway 0-7
#   - Cross-game transfer: train Freeway-3, test SpaceInvaders 0-9, Breakout 0
#
# Usage: ./run_fame_benchmark.sh
# ============================================================================

set -e

# Directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p agents logs data_FAME

# Output CSV
EVAL_CSV="eval_cross_game_matrix.csv"

# Games and modes
declare -A GAME_MODES
GAME_MODES["ALE/Breakout-v5"]=1
GAME_MODES["ALE/Freeway-v5"]=8
GAME_MODES["ALE/SpaceInvaders-v5"]=10

ALL_GAMES=("ALE/Breakout-v5" "ALE/Freeway-v5" "ALE/SpaceInvaders-v5")
SEEDS=(1 2 3)
TIMESTEPS=1000000

# ============================================================================
# PHASE 1: TRAINING
# ============================================================================
echo ""
echo "=============================================="
echo "PHASE 1: TRAINING FAME ON ALL GAMES"
echo "=============================================="
echo "Timesteps per task: ${TIMESTEPS}"
echo "Seeds: ${SEEDS[*]}"
echo ""

for GAME in "${ALL_GAMES[@]}"; do
    GAME_NAME=$(echo "$GAME" | sed 's/ALE\///;s/-v5//')
    N_MODES=${GAME_MODES[$GAME]}

    echo "----------------------------------------------"
    echo "Training on ${GAME} (${N_MODES} modes)"
    echo "----------------------------------------------"

    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "=== Seed ${SEED} ==="

        # Create game-specific data directory
        mkdir -p "data_FAME/envs/${GAME_NAME}"

        # Run FAME training (handles all modes sequentially)
        python3 src/train/run_ppo_FAME.py \
            --model-type=FAME \
            --env-id="${GAME}" \
            --seed=${SEED} \
            --save-dir=agents \
            --total-timesteps=${TIMESTEPS} \
            --epoch_meta=200 \
            --buffer_path="data_FAME/${GAME_NAME}_buffer_"

        echo "Seed ${SEED} complete: ${N_MODES} models saved"
    done

    echo ""
    echo "Training on ${GAME} complete!"
done

echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "=============================================="
echo ""

# ============================================================================
# PHASE 2: EVALUATION (Cross-Game Transfer Matrix)
# ============================================================================
echo ""
echo "=============================================="
echo "PHASE 2: CROSS-GAME EVALUATION"
echo "=============================================="
echo ""

# Write CSV header
echo "train_game,train_mode,test_game,test_mode,seed,avg_return" > "$EVAL_CSV"

TOTAL_EVAL=0

# For each trained model, test on ALL games' modes
for TRAIN_GAME in "${ALL_GAMES[@]}"; do
    TRAIN_GAME_NAME=$(echo "$TRAIN_GAME" | sed 's/ALE\///;s/-v5//')
    N_TRAIN_MODES=${GAME_MODES[$TRAIN_GAME]}

    for SEED in "${SEEDS[@]}"; do
        for TRAIN_MODE in $(seq 0 $((N_TRAIN_MODES - 1))); do
            # Construct model path
            MODEL_DIR="agents/ALE-${TRAIN_GAME_NAME}-v5_${TRAIN_MODE}__FAME__run_ppo_FAME__${SEED}"

            if [ ! -d "$MODEL_DIR" ]; then
                echo "WARNING: Model not found: ${MODEL_DIR}, skipping..."
                continue
            fi

            echo ""
            echo "Evaluating: ${MODEL_DIR}"

            # Test on ALL games' modes
            for TEST_GAME in "${ALL_GAMES[@]}"; do
                TEST_GAME_NAME=$(echo "$TEST_GAME" | sed 's/ALE\///;s/-v5//')
                N_TEST_MODES=${GAME_MODES[$TEST_GAME]}

                for TEST_MODE in $(seq 0 $((N_TEST_MODES - 1))); do
                    echo -n "  -> ${TEST_GAME} mode ${TEST_MODE}: "

                    # Run evaluation
                    OUTPUT=$(python3 src/evaluate/test_agent.py \
                        --load "$MODEL_DIR" \
                        --test-env "$TEST_GAME" \
                        --test-mode "$TEST_MODE" \
                        --num-episodes 10 \
                        --max-timesteps 1000 \
                        2>&1)

                    # Parse average return
                    AVG_RET=$(echo "$OUTPUT" | grep "Avg. episodic return:" | sed 's/.*: //')

                    if [ -n "$AVG_RET" ]; then
                        echo "${AVG_RET}"
                        echo "${TRAIN_GAME},${TRAIN_MODE},${TEST_GAME},${TEST_MODE},${SEED},${AVG_RET}" >> "$EVAL_CSV"
                        TOTAL_EVAL=$((TOTAL_EVAL + 1))
                    else
                        echo "FAILED"
                        echo "${TRAIN_GAME},${TRAIN_MODE},${TEST_GAME},${TEST_MODE},${SEED},NaN" >> "$EVAL_CSV"
                    fi
                done
            done
        done
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
echo "TRANSFER MATRIX SUMMARY"
echo "=============================================="

if command -v python3 &> /dev/null; then
    python3 << 'EOF'
import pandas as pd

df = pd.read_csv("eval_cross_game_matrix.csv")

# Pivot to show transfer matrix (avg across seeds)
df_pivot = df.pivot_table(
    index=['train_game', 'train_mode'],
    columns=['test_game', 'test_mode'],
    values='avg_return'
).round(2)

print("\nTransfer Matrix (Average Return Across 3 Seeds):")
print("=" * 60)
print(df_pivot.to_string())

# Within-game vs cross-game summary
print("\n" + "=" * 60)
print("Within-Game vs Cross-Game Transfer:")
print("=" * 60)

for tg in df['train_game'].unique():
    tg_name = tg.replace("ALE/", "").replace("-v5", "")
    within = df[(df['train_game'] == tg) & (df['test_game'] == tg)]['avg_return'].mean()

    for test_g in df['test_game'].unique():
        if test_g != tg:
            test_g_name = test_g.replace("ALE/", "").replace("-v5", "")
            cross = df[(df['train_game'] == tg) & (df['test_game'] == test_g)]['avg_return'].mean()
            print(f"Train {tg_name:15} -> Test {test_g_name:15}: {cross:7.2f} (vs within-game: {within:.2f})")
EOF
fi

echo ""
echo "Benchmark complete!"
