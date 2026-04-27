#!/bin/bash
# ============================================================================
# Cross-Game Continual Learning Benchmark for ALL FAME Paper Baselines
# ============================================================================
#
# Trains a single agent on all games sequentially, then evaluates across all
# game/mode combos:
#   Training: Breakout(0) -> Freeway(0-7) -> SpaceInvaders(0-9)
#   Evaluation: All games x all modes
#
# FAME paper baselines (Table 1):
#   cnn-simple    — Reset / from-scratch
#   cnn-simple-ft — Fine-tuning
#   cnn-componet  — CompoNet (compositional network)
#   prog-net      — ProgressiveNet (progressive neural networks)
#   packnet       — PackNet (pruning-based)
#   fame          — FAME (fast + meta dual-learner)
#
# Usage:
#   ./run_baselines_cross_game.sh <baseline> [seed] [timesteps]
#   ./run_baselines_cross_game.sh all [seed] [timesteps]   # Run all baselines
#
# Examples:
#   ./run_baselines_cross_game.sh packnet
#   ./run_baselines_cross_game.sh fame 1 1000000
#   ./run_baselines_cross_game.sh all 1 1000000
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

SCRIPT_DIR="/home/egyaase/continual-learning/FAME/Atari"
cd "$SCRIPT_DIR"

# Configuration
BASELINE=${1:?"Usage: $0 <baseline|all> [seed] [timesteps]
Valid baselines: cnn-simple, cnn-simple-ft, cnn-componet, prog-net, packnet, fame, all"}
SEED=${2:-1}
TIMESTEPS=${3:-1000000}

ALL_BASELINES=("cnn-simple" "cnn-simple-ft" "cnn-componet" "prog-net" "packnet" "fame")

# Validate baseline
if [ "$BASELINE" != "all" ]; then
    case "$BASELINE" in
        cnn-simple|cnn-simple-ft|cnn-componet|prog-net|packnet|fame)
            ;;
        *)
            echo "ERROR: Unknown baseline '$BASELINE'"
            echo "Valid options: cnn-simple, cnn-simple-ft, cnn-componet, prog-net, packnet, fame, all"
            exit 1
            ;;
    esac
fi

# Directories
# mkdir -p agents logs

# Game configuration
declare -A GAME_MODES
GAME_MODES["ALE/Breakout-v5"]="0"
GAME_MODES["ALE/Freeway-v5"]="0 1 2 3 4 5 6 7"
GAME_MODES["ALE/SpaceInvaders-v5"]="0 1 2 3 4 5 6 7 8 9"

ALL_GAMES=("ALE/Breakout-v5" "ALE/Freeway-v5" "ALE/SpaceInvaders-v5")

# Task ID mapping
TASK_ID_FOR() {
    python3 -c "
GAME_SEQUENCE = ['ALE/Breakout-v5', 'ALE/Freeway-v5', 'ALE/SpaceInvaders-v5']
GAME_MODES = {'ALE/Breakout-v5': [0], 'ALE/Freeway-v5': list(range(8)), 'ALE/SpaceInvaders-v5': list(range(10))}
test_game = '$1'
test_mode = $2
task_id = 0
for g in GAME_SEQUENCE:
    if g == test_game:
        task_id += test_mode
        break
    task_id += len(GAME_MODES[g])
print(task_id)
"
}

run_single_baseline() {
    local BL=$1
    local SD=$2
    local TS=$3

    echo ""
    echo "=============================================="
    echo "Cross-Game Benchmark: ${BL}"
    echo "=============================================="
    echo "Seed: ${SD}"
    echo "Timesteps per task: ${TS}"
    echo "=============================================="
    echo ""

    # ========================================================================
    # PHASE 1: TRAINING
    # ========================================================================
    echo "=============================================="
    echo "PHASE 1: TRAINING (Cross-Game)"
    echo "=============================================="
    echo ""

    python3 run_baselines_cross_game.py \
        --model-type "${BL}" \
        --seed "${SD}" \
        --save-dir agents \
        --total-timesteps "${TS}"

    echo ""
    echo "Training complete!"
    echo ""

    # ========================================================================
    # PHASE 2: EVALUATION
    # ========================================================================
    echo ""
    echo "=============================================="
    echo "PHASE 2: CROSS-GAME EVALUATION"
    echo "=============================================="
    echo ""

    # Output CSV
    EVAL_CSV="eval_cross_game_${BL}_seed${SD}.csv"
    echo "algorithm,environment,train_mode,test_mode,seed,ep_ret" > "$EVAL_CSV"

    # Model directory
    MODEL_BASE="agents/cross_game__${BL}__run_baselines_cross_game__${SD}"

    if [ ! -d "$MODEL_BASE" ]; then
        echo "ERROR: Model not found at ${MODEL_BASE}"
        echo "Skipping evaluation for ${BL}."
        return 1
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

            # Compute task_id for checkpoint lookup
            TASK_ID=$(TASK_ID_FOR "$TEST_GAME" "$TEST_MODE")

            # Determine checkpoint directory based on baseline type
            CHECKPOINT_DIR="${MODEL_BASE}/task_${TASK_ID}"

            # Build eval command (all use --cross-game for ActionPadWrapper)
            EVAL_CMD="python3 test_agent.py --load ${CHECKPOINT_DIR} --algorithm ${BL} --test-env ${TEST_GAME} --test-mode ${TEST_MODE} --cross-game --num-episodes 10 --max-timesteps 1000"

            # For prog-net and cnn-componet, pass prev-units
            if [ "$BL" = "prog-net" ] || [ "$BL" = "cnn-componet" ]; then
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
            if [ "$BL" = "packnet" ]; then
                FINAL_CHECKPOINT="${MODEL_BASE}/task_18"
                TASK_CHECKPOINT="${MODEL_BASE}/task_${TASK_ID}"
                EVAL_CMD="python3 test_agent.py --load ${FINAL_CHECKPOINT} --algorithm ${BL} --mode ${TASK_ID} --test-env ${TEST_GAME} --test-mode ${TEST_MODE} --task-checkpoint ${TASK_CHECKPOINT} --cross-game --num-episodes 10 --max-timesteps 1000"
            fi

            # For FAME, load the fast agent
            if [ "$BL" = "fame" ]; then
                EVAL_CMD="python3 test_agent.py --load ${CHECKPOINT_DIR} --algorithm FAME --test-env ${TEST_GAME} --test-mode ${TEST_MODE} --cross-game --num-episodes 10 --max-timesteps 1000"
            fi

            OUTPUT=$(${EVAL_CMD} 2>&1)

            AVG_RET=$(echo "$OUTPUT" | grep "Avg. episodic return:" | sed 's/.*: //')

            if [ -n "$AVG_RET" ]; then
                echo "${AVG_RET}"
                echo "${BL},${TEST_GAME},${TASK_ID},${TEST_MODE},${SD},${AVG_RET}" >> "$EVAL_CSV"
                TOTAL_EVAL=$((TOTAL_EVAL + 1))
            else
                echo "FAILED"
                echo "${BL},${TEST_GAME},${TASK_ID},${TEST_MODE},${SD},NaN" >> "$EVAL_CSV"
            fi
        done
    done

    echo ""
    echo "=============================================="
    echo "EVALUATION COMPLETE: ${BL}"
    echo "=============================================="
    echo "Total evaluations: ${TOTAL_EVAL}"
    echo "Results saved to: ${EVAL_CSV}"
    echo ""

    # ========================================================================
    # PHASE 3: SUMMARY
    # ========================================================================
    echo "=============================================="
    echo "RESULTS SUMMARY: ${BL}"
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
    echo "Benchmark ${BL} complete!"
}

# ============================================================================
# Run requested baseline(s)
# ============================================================================
if [ "$BASELINE" = "all" ]; then
    echo ""
    echo "################################################"
    echo "# Running ALL FAME paper baselines"
    echo "################################################"
    echo "Baselines: ${ALL_BASELINES[*]}"
    echo ""

    for BL in "${ALL_BASELINES[@]}"; do
        echo ""
        echo "################################################"
        echo "# Running baseline: ${BL}"
        echo "################################################"
        echo ""
        run_single_baseline "$BL" "$SEED" "$TIMESTEPS" || echo "WARNING: ${BL} failed, continuing..."
        echo ""
        echo "Baseline ${BL} complete."
        echo ""
    done

    # ========================================================================
    # COMBINED SUMMARY
    # ========================================================================
    echo ""
    echo "=============================================="
    echo "COMBINED RESULTS SUMMARY"
    echo "=============================================="

    if command -v python3 &> /dev/null; then
        python3 << 'PYEOF'
import pandas as pd
import glob
import os

csv_files = sorted(glob.glob("eval_cross_game_*_seed*.csv"))
if not csv_files:
    print("No result files found.")
else:
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        print("\nAverage Return by Algorithm and Game:")
        print("-" * 60)
        for algo in all_df['algorithm'].unique():
            algo_df = all_df[all_df['algorithm'] == algo]
            print(f"\n  {algo}:")
            for game in algo_df['environment'].unique():
                game_data = algo_df[algo_df['environment'] == game]['ep_ret']
                mean_ret = game_data.mean()
                std_ret = game_data.std()
                game_name = str(game).replace("ALE/", "").replace("-v5", "")
                print(f"    {game_name:15}: {mean_ret:7.2f} (+/- {std_ret:.2f})")
    else:
        print("No valid result files found.")
PYEOF
    fi

    echo ""
    echo "All baselines complete!"
else
    run_single_baseline "$BASELINE" "$SEED" "$TIMESTEPS"
fi