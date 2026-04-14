#!/bin/bash
# ============================================================================
# Atari Continual Learning Benchmark
# Reproduces experiments from: "Principled Fast and Meta Knowledge Learners
# for Continual Reinforcement Learning" (ICLR 2026)
# https://openreview.net/forum?id=loNTDX3wTn
# ============================================================================
#
# Usage:
#   ./benchmark_experiments.sh [OPTIONS]
#
# Options:
#   --env=ENV            Game (default: Breakout, Freeway, and SpaceInvaders)
#   --seeds=1,2,3        Comma-separated seeds (default: 1,2,3)
#   --timesteps=N        Timesteps per task (default: 1000000)
#   --algorithms=LIST    Comma-separated list (default: all)
#   --skip-training      Skip training, run evaluation only
#   --skip-eval          Skip evaluation, run training only
#   --skip-process       Skip results processing
#   --quick              Quick test: 10000 steps, 1 seed, Freeway only
#   -h, --help           Show this help
#
# Algorithms:
#   cnn-simple     From-scratch baseline (standard CNN + MLP)
#   cnn-simple-ft Fine-tuning baseline
#   cnn-componet  Compositional network with attention
#   prog-net      Progressive neural networks
#   packnet       Pruning-based continual learning
#   fame          Fast Adaptive Meta-Emergence (FAME)
#   dino-simple   DINO vision transformer encoder (optional, slow)
#
# Games:
#   ALE/Breakout-v5       1 game mode
#   ALE/SpaceInvaders-v5 10 game modes
# ============================================================================

set -uo pipefail  # allow command failures (we handle them), but catch undefined vars

# ---- Parse CLI arguments ----
ENVS=""
SEEDS="1,2,3"
TIMESTEPS=1000000
ALGORITHMS=""
SKIP_TRAINING=0
SKIP_EVAL=0
SKIP_PROCESS=0
QUICK=0

show_help() {
    sed -n '2,/^# =====/p' "$0" | sed 's/^# //;s/^#//'
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --env=*)       ENVS="${arg#*=}" ;;
        --seeds=*)      SEEDS="${arg#*=}" ;;
        --timesteps=*)  TIMESTEPS="${arg#*=}" ;;
        --algorithms=*) ALGORITHMS="${arg#*=}" ;;
        --skip-training) SKIP_TRAINING=1 ;;
        --skip-eval)     SKIP_EVAL=1 ;;
        --skip-process)  SKIP_PROCESS=1 ;;
        --quick)         QUICK=1 ;;
        -h|--help)       show_help ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ---- Quick mode overrides ----
if [ "$QUICK" -eq 1 ]; then
    ENVS="ALE/Freeway-v5"
    SEEDS="1"
    TIMESTEPS=10000
    echo "=== QUICK TEST MODE ==="
fi

# ---- Defaults ----
if [ -z "$ENVS" ]; then
    ENVS="ALE/Breakout-v5,ALE/Freeway-v5,ALE/SpaceInvaders-v5"
fi

if [ -z "$ALGORITHMS" ]; then
    ALGORITHMS="cnn-simple,cnn-simple-ft,cnn-componet,prog-net,packnet,fame"
fi

# ---- Task structure per game ----
declare -A GAME_MODES
GAME_MODES["ALE/Breakout-v5"]=1
GAME_MODES["ALE/Freeway-v5"]=8
GAME_MODES["ALE/SpaceInvaders-v5"]=10

# ---- Tracking ----
LOG_FILE="benchmark_$(date +%Y%m%d_%H%M%S).log"
ERROR_FILE="benchmark_errors.log"
TOTAL_TRAIN=0
TOTAL_TRAIN_OK=0
TOTAL_EVAL=0
TOTAL_EVAL_OK=0

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$ERROR_FILE" "$LOG_FILE"; }

# ---- Get script directory ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

log "=========================================="
log "Atari Continual Learning Benchmark"
log "=========================================="
log "Environments: $ENVS"
log "Seeds: $SEEDS"
log "Timesteps: $TIMESTEPS"
log "Algorithms: $ALGORITHMS"
log "=========================================="

# ---- Helper: convert env to run-name prefix ----
env_to_prefix() {
    # ALE/Freeway-v5 -> ALE-Freeway-v5
    echo "${1//\//-}"
}

# ---- Helper: build agent directory path ----
agent_dir() {
    local env_prefix="$1"
    local mode="$2"
    local algo="$3"
    local seed="$4"
    # FAME uses run_ppo_FAME as exp_name; others use run_ppo
    local exp_name="run_ppo"
    if [ "$algo" = "FAME" ]; then
        exp_name="run_ppo_FAME"
    fi
    echo "agents/${env_prefix}_${mode}__${algo}__${exp_name}__${seed}"
}

# ---- Helper: build prev-units args for continual methods ----
prev_units_args() {
    local env_prefix="$1"
    local algo="$2"
    local current_mode="$3"
    local seed="$4"
    local exp_name="run_ppo"
    if [ "$algo" = "FAME" ]; then
        exp_name="run_ppo_FAME"
    fi

    if [ "$current_mode" -eq 0 ]; then
        echo ""
        return
    fi

    local args=""

    if [ "$algo" = "cnn-simple-ft" ] || [ "$algo" = "packnet" ]; then
        # single previous module: the agent from the previous mode
        local prev_mode=$((current_mode - 1))
        args="--prev-units agents/${env_prefix}_${prev_mode}__${algo}__${exp_name}__${seed}"
    elif [ "$algo" = "cnn-componet" ] || [ "$algo" = "prog-net" ]; then
        # all previous modules
        args="--prev-units"
        for m in $(seq 0 $((current_mode - 1))); do
            args="$args agents/${env_prefix}_${m}__${algo}__${exp_name}__${seed}"
        done
    fi

    echo "$args"
}

# ============================================================================
# Phase 1: Training
# ============================================================================
run_training() {
    if [ "$SKIP_TRAINING" -eq 1 ]; then
        log "Skipping training (--skip-training)"
        return
    fi

    log ""
    log "=========================================="
    log "PHASE 1: TRAINING"
    log "=========================================="

    IFS=',' read -ra ENV_LIST <<< "$ENVS"
    IFS=',' read -ra ALGO_LIST <<< "$ALGORITHMS"
    IFS=',' read -ra SEED_LIST <<< "$SEEDS"

    for env in "${ENV_LIST[@]}"; do
        local n_modes=${GAME_MODES[$env]}
        local env_prefix=$(env_to_prefix "$env")
        local game_name=$(echo "$env" | sed 's/ALE\///;s/-v5//')

        for algo in "${ALGO_LIST[@]}"; do
            # Map CLI algorithm name to model_type used by run_experiments.py
            local model_type="$algo"
            if [ "$algo" = "cnn-simple-ft" ]; then
                model_type="finetune"
            fi

            for seed in "${SEED_LIST[@]}"; do
                log ""
                log "--- $algo | $env | seed=$seed ---"

                for mode in $(seq 0 $((n_modes - 1))); do
                    local dir=$(agent_dir "$env_prefix" "$mode" "$algo" "$seed")

                    # Resume: skip if already trained
                    if [ -d "$dir" ]; then
                        log "  Mode $mode: already exists at $dir, skipping"
                        continue
                    fi

                    TOTAL_TRAIN=$((TOTAL_TRAIN + 1))
                    log "  Mode $mode: training -> $dir"

                    # Build the training command
                    local script="run_experiments.py"
                    local cli_algo="$model_type"

                    if [ "$algo" = "FAME" ]; then
                        script="run_ppo_FAME.py"
                        cli_algo="FAME"
                    fi

                    local cmd="python3 $script --algorithm=$cli_algo --env=$env --seed=$seed --start-mode=$mode --timesteps=$TIMESTEPS"

                    # Algorithm-specific args
                    if [ "$algo" = "cnn-componet" ]; then
                        cmd="$cmd --componet-finetune-encoder"
                    fi
                    if [ "$algo" = "packnet" ]; then
                        cmd="$cmd --total-task-num=$n_modes"
                    fi
                    if [ "$algo" = "dino-simple" ]; then
                        cmd="$cmd --dino-size=s --num-envs=2"
                    fi

                    # Previous units for continual methods
                    local prev_args=$(prev_units_args "$env_prefix" "$algo" "$mode" "$seed")
                    if [ -n "$prev_args" ]; then
                        cmd="$cmd $prev_args"
                    fi

                    # FAME buffer path
                    if [ "$algo" = "FAME" ]; then
                        cmd="$cmd --buffer-path data_FAME/${game_name}_buffer_"
                    fi

                    # Run
                    log "    CMD: $cmd"
                    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
                        TOTAL_TRAIN_OK=$((TOTAL_TRAIN_OK + 1))
                        log "  Mode $mode: DONE"
                    else
                        log_error "  Mode $mode: FAILED (exit $?)"
                    fi
                done
            done
        done
    done

    log ""
    log "Training complete: $TOTAL_TRAIN_OK/$TOTAL_TRAIN succeeded"
}

# ============================================================================
# Phase 2: Evaluation
# ============================================================================
run_evaluation() {
    if [ "$SKIP_EVAL" -eq 1 ]; then
        log "Skipping evaluation (--skip-eval)"
        return
    fi

    log ""
    log "=========================================="
    log "PHASE 2: EVALUATION"
    log "=========================================="

    IFS=',' read -ra ENV_LIST <<< "$ENVS"
    IFS=',' read -ra ALGO_LIST <<< "$ALGORITHMS"
    IFS=',' read -ra SEED_LIST <<< "$SEEDS"

    for env in "${ENV_LIST[@]}"; do
        local n_modes=${GAME_MODES[$env]}
        local env_prefix=$(env_to_prefix "$env")
        local eval_csv="eval_results_$(echo "$env" | sed 's/ALE\///;s/-v5//').csv"

        # Write CSV header
        if [ ! -f "$eval_csv" ]; then
            echo "algorithm,environment,train_mode,test_mode,seed,ep_ret" > "$eval_csv"
        fi

        for algo in "${ALGO_LIST[@]}"; do
            for seed in "${SEED_LIST[@]}"; do
                for train_mode in $(seq 0 $((n_modes - 1))); do
                    local dir=$(agent_dir "$env_prefix" "$train_mode" "$algo" "$seed")

                    if [ ! -d "$dir" ]; then
                        log "  $dir not found, skipping evaluation"
                        continue
                    fi

                    for test_mode in $(seq 0 $((n_modes - 1))); do
                        TOTAL_EVAL=$((TOTAL_EVAL + 1))

                        # Skip if already evaluated
                        if grep -q "^${algo},${env},${train_mode},${test_mode},${seed}," "$eval_csv" 2>/dev/null; then
                            log "  $algo t=$train_mode e=$test_mode s=$seed: already in CSV, skipping"
                            TOTAL_EVAL_OK=$((TOTAL_EVAL_OK + 1))
                            continue
                        fi

                        log "  $algo t=$train_mode e=$test_mode s=$seed: evaluating..."

                        # Run test_agent.py, capture output
                        local output
                        output=$(python3 test_agent.py \
                            --load "$dir" \
                            --mode "$test_mode" \
                            --seed "$seed" \
                            --num-episodes 10 \
                            --max-timesteps 1000 \
                            2>&1)
                        local rc=$?

                        if [ $rc -eq 0 ]; then
                            # Parse average return from output
                            local avg_ret
                            avg_ret=$(echo "$output" | grep "Avg. episodic return:" | sed 's/.*: //')
                            if [ -z "$avg_ret" ]; then
                                avg_ret="NaN"
                            fi
                            echo "${algo},${env},${train_mode},${test_mode},${seed},${avg_ret}" >> "$eval_csv"
                            TOTAL_EVAL_OK=$((TOTAL_EVAL_OK + 1))
                            log "    -> avg_return=$avg_ret"
                        else
                            log_error "  $algo t=$train_mode e=$test_mode s=$seed: FAILED (exit $rc)"
                            echo "${algo},${env},${train_mode},${test_mode},${seed},NaN" >> "$eval_csv"
                        fi
                    done
                done
            done
        done

        log "Evaluation for $env complete: $TOTAL_EVAL_OK/$TOTAL_EVAL entries"
        log "Results saved to $eval_csv"
    done
}

# ============================================================================
# Phase 3: Results Processing
# ============================================================================
run_processing() {
    if [ "$SKIP_PROCESS" -eq 1 ]; then
        log "Skipping results processing (--skip-process)"
        return
    fi

    log ""
    log "=========================================="
    log "PHASE 3: RESULTS PROCESSING"
    log "=========================================="

    IFS=',' read -ra ENV_LIST <<< "$ENVS"

    for env in "${ENV_LIST[@]}"; do
        local game_name=$(echo "$env" | sed 's/ALE\///;s/-v5//')
        local n_modes=${GAME_MODES[$env]}
        local data_dir="data_FAME/envs/${game_name}"

        # Step 3a: Extract TensorBoard event data to per-task CSVs
        log "Extracting TensorBoard data for $game_name..."
        mkdir -p "$data_dir"

        # Run process_results_pre.py (it handles its own iteration over modes/seeds)
        log "Running process_results_pre.py for $game_name..."
        python3 process_results_pre.py 2>&1 | tail -5 || \
            log_error "  process_results_pre.py failed for $game_name"

        # Step 3b: Compute continual RL metrics
        local eval_csv="eval_results_${game_name}.csv"
        if [ -f "$eval_csv" ]; then
            log "Computing continual RL metrics for $game_name..."
            python3 process_results.py \
                --data-dir "$data_dir" \
                --eval-results "$eval_csv" 2>&1 || \
                log_error "process_results.py failed for $game_name"
        else
            log "No evaluation results found at $eval_csv, skipping metric computation"
        fi
    done
}

# ============================================================================
# Main
# ============================================================================
run_training
run_evaluation
run_processing

log ""
log "=========================================="
log "BENCHMARK COMPLETE"
log "=========================================="
log "Training:  $TOTAL_TRAIN_OK/$TOTAL_TRAIN tasks succeeded"
log "Eval:      $TOTAL_EVAL_OK/$TOTAL_EVAL entries succeeded"
if [ "$TOTAL_TRAIN" -ne "$TOTAL_TRAIN_OK" ] || [ "$TOTAL_EVAL" -ne "$TOTAL_EVAL_OK" ]; then
    log "Some tasks failed. See $ERROR_FILE for details."
    exit 1
fi
log "All tasks completed successfully!"
exit 0