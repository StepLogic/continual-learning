#!/bin/bash
# ============================================================================
# Cross-Game Continual Learning Benchmark for PPO Baselines
# ============================================================================
#
# Trains ONE agent on all games, then evaluates across all game/mode combos:
#   Training: Breakout(0) → Freeway(0-7) → SpaceInvaders(0-9)
#   Evaluation: All games × all modes
#
# Usage: ./run_cross_game.sh <baseline> [seed] [timesteps]
#   baseline:  cnn-simple | cnn-simple-ft | dino-simple | cnn-componet | prog-net | packnet
#   seed:      optional, default=1
#   timesteps: optional, default=1000000
#
# Examples:
#   ./run_cross_game.sh packnet
#   ./run_cross_game.sh cnn-simple 1 1000000
#   ./run_cross_game.sh prog-net 2
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
BASELINE=${1:?"Usage: $0 <baseline> [seed] [timesteps]"}
SEED=${2:-1}
TIMESTEPS=${3:-1000000}

# Validate baseline
case "$BASELINE" in
    cnn-simple|cnn-simple-ft|dino-simple|cnn-componet|prog-net|packnet)
        ;;
    *)
        echo "ERROR: Unknown baseline '$BASELINE'"
        echo "Valid options: cnn-simple, cnn-simple-ft, dino-simple, cnn-componet, prog-net, packnet"
        exit 1
        ;;
esac

# Directories
mkdir -p agents logs

# Game configuration
declare -A GAME_MODES
GAME_MODES["ALE/Breakout-v5"]="0"
GAME_MODES["ALE/Freeway-v5"]="0 1 2 3 4 5 6 7"
GAME_MODES["ALE/SpaceInvaders-v5"]="0 1 2 3 4 5 6 7 8 9"

ALL_GAMES=("ALE/Breakout-v5" "ALE/Freeway-v5" "ALE/SpaceInvaders-v5")

# Task ID mapping: (game, mode) -> global task_id
# Breakout 0=task0, Freeway 0-7=task1-8, SpaceInvaders 0-9=task9-18

echo ""
echo "=============================================="
echo "Cross-Game Benchmark: ${BASELINE}"
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

EXTRA_ARGS=""
if [ "$BASELINE" = "dino-simple" ]; then
    EXTRA_ARGS="--num-envs 1"
fi

python3 run_ppo_cross_game.py \
    --model-type "${BASELINE}" \
    --seed "${SEED}" \
    --save-dir agents \
    --total-timesteps "${TIMESTEPS}" \
    ${EXTRA_ARGS}

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
EVAL_CSV="eval_cross_game_${BASELINE}_seed${SEED}.csv"
echo "algorithm,environment,train_mode,test_mode,seed,ep_ret" > "$EVAL_CSV"

# Model directory
MODEL_BASE="agents/cross_game__${BASELINE}__run_ppo_cross_game__${SEED}"

if [ ! -d "$MODEL_BASE" ]; then
    echo "ERROR: Model not found at ${MODEL_BASE}"
    exit 1
fi

echo "Evaluating model: ${MODEL_BASE}"
echo ""

TOTAL_EVAL=0

for TEST_GAME in "${ALL_GAMES[@]}"; do
    TEST_GAME_NAME=$(echo "$TEST_GAME" | sed 's/ALE\///;s/-v5//')
    MODES=(${GAME_MODES[$TEST_GAME]})

    echo "Testing on ${TEST_GAME} (${#MODES[@]} modes)..."

    for TEST_MODE in "${MODES[@]}"; do
        echo -n "  Mode ${TEST_MODE}: "

        # Determine the task_id for this (game, mode)
        TASK_ID=$(python3 -c "
GAME_SEQUENCE = ['ALE/Breakout-v5', 'ALE/Freeway-v5', 'ALE/SpaceInvaders-v5']
GAME_MODES = {'ALE/Breakout-v5': [0], 'ALE/Freeway-v5': list(range(8)), 'ALE/SpaceInvaders-v5': list(range(10))}
test_game = '${TEST_GAME}'
test_mode = ${TEST_MODE}
task_id = 0
for g in GAME_SEQUENCE:
    if g == test_game:
        task_id += test_mode
        break
    task_id += len(GAME_MODES[g])
print(task_id)
")

        CHECKPOINT_DIR="${MODEL_BASE}/task_${TASK_ID}"

        # Build eval command based on baseline (all use --cross-game for ActionPadWrapper)
        EVAL_CMD="python3 test_agent.py --load ${CHECKPOINT_DIR} --algorithm ${BASELINE} --test-env ${TEST_GAME} --test-mode ${TEST_MODE} --cross-game --num-episodes 10 --max-timesteps 1000"

        # For prog-net and cnn-componet, pass prev-units
        if [ "$BASELINE" = "prog-net" ] || [ "$BASELINE" = "cnn-componet" ]; then
            PREV_UNITS=""
            for prev in $(seq 0 $((TASK_ID - 1))); do
                PREV_UNITS="${PREV_UNITS} ${MODEL_BASE}/task_${prev}"
            done
            if [ -n "$PREV_UNITS" ]; then
                EVAL_CMD="${EVAL_CMD} --prev-units ${PREV_UNITS}"
            fi
        fi

        # For packnet, load from the final checkpoint (has all masks)
        # and set the correct task view + load per-task actor/critic
        if [ "$BASELINE" = "packnet" ]; then
            FINAL_CHECKPOINT="${MODEL_BASE}/task_18"
            TASK_CHECKPOINT="${MODEL_BASE}/task_${TASK_ID}"
            EVAL_CMD="python3 test_agent.py --load ${FINAL_CHECKPOINT} --algorithm ${BASELINE} --mode ${TASK_ID} --test-env ${TEST_GAME} --test-mode ${TEST_MODE} --task-checkpoint ${TASK_CHECKPOINT} --cross-game --num-episodes 10 --max-timesteps 1000"
        fi

        OUTPUT=$(${EVAL_CMD} 2>&1)

        AVG_RET=$(echo "$OUTPUT" | grep "Avg. episodic return:" | sed 's/.*: //')

        if [ -n "$AVG_RET" ]; then
            echo "${AVG_RET}"
            echo "${BASELINE},${TEST_GAME},${TASK_ID},${TEST_MODE},${SEED},${AVG_RET}" >> "$EVAL_CSV"
            TOTAL_EVAL=$((TOTAL_EVAL + 1))
        else
            echo "FAILED"
            echo "${BASELINE},${TEST_GAME},${TASK_ID},${TEST_MODE},${SEED},NaN" >> "$EVAL_CSV"
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
echo "RESULTS SUMMARY: ${BASELINE}"
echo "=============================================="

if command -v python3 &> /dev/null; then
    python3 << PYEOF
import pandas as pd

df = pd.read_csv("${EVAL_CSV}")

print("\nAverage Return by Test Game:")
print("-" * 40)
for game in df['environment'].unique():
    game_data = df[df['environment'] == game]['ep_ret']
    mean_ret = game_data.mean()
    std_ret = game_data.std()
    game_name = str(game).replace("ALE/", "").replace("-v5", "")
    print(f"  {game_name:15}: {mean_ret:7.2f} (+/- {std_ret:.2f})")

print("\nDetailed Results (Mean Return per Mode):")
print("-" * 40)
summary = df.groupby(['environment', 'test_mode'])['ep_ret'].mean().reset_index()
for _, row in summary.iterrows():
    game_name = str(row['environment']).replace("ALE/", "").replace("-v5", "")
    print(f"  {game_name} mode {int(row['test_mode']):2d}: {row['ep_ret']:7.2f}")
PYEOF
fi

echo ""
echo "Benchmark complete!"