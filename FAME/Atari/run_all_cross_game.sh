#!/bin/bash
# ============================================================================
# Run ALL baselines through cross-game continual learning benchmark
# ============================================================================
#
# Usage: ./run_all_cross_game.sh [seed] [timesteps]
#   seed:      optional, default=1
#   timesteps: optional, default=1000000
#
# This runs each baseline sequentially:
#   packnet, cnn-simple, cnn-simple-ft, dino-simple, cnn-componet, prog-net
#
# Results are saved to: eval_cross_game_<baseline>_seed<seed>.csv
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SEED=${1:-1}
TIMESTEPS=${2:-1000000}

BASELINES=("packnet" "cnn-simple" "cnn-simple-ft" "dino-simple" "cnn-componet" "prog-net")

echo ""
echo "=============================================="
echo "Cross-Game Benchmark: ALL BASELINES"
echo "=============================================="
echo "Seed: ${SEED}"
echo "Timesteps per task: ${TIMESTEPS}"
echo "Baselines: ${BASELINES[*]}"
echo "=============================================="
echo ""

for BASELINE in "${BASELINES[@]}"; do
    echo ""
    echo "################################################"
    echo "# Running baseline: ${BASELINE}"
    echo "################################################"
    echo ""

    ./run_cross_game.sh "$BASELINE" "$SEED" "$TIMESTEPS" || echo "WARNING: ${BASELINE} failed, continuing..."

    echo ""
    echo "Baseline ${BASELINE} complete."
    echo ""
done

# ============================================================================
# COMBINED SUMMARY
# ============================================================================
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