#!/usr/bin/env python3
"""
Evaluate all trained agents and generate results summary.

Usage:
    python3 evaluate_all.py [--agents-dir=agents] [--env=ALE/Freeway-v5] [--output=results.csv]
"""

import os
import re
import subprocess
import glob
import argparse
import pandas as pd
from task_utils import parse_name_info


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all trained agents")
    parser.add_argument("--agents-dir", type=str, default="agents", help="Directory containing trained models")
    parser.add_argument("--env", type=str, default=None, help="Filter by environment (optional)")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Output CSV file")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes per evaluation")
    parser.add_argument("--max-timesteps", type=int, default=1000, help="Max timesteps per episode")
    return parser.parse_args()


def evaluate_model(cmd, num_episodes, max_timesteps):
    """Evaluate a single model via test_agent.py subprocess and return the results."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr

        # Parse average episodic return from output
        avg_return = None
        for line in output.split("\n"):
            if "Avg. episodic return:" in line:
                try:
                    avg_return = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass

        return {
            "success": result.returncode == 0,
            "avg_return": avg_return,
            "error": result.stderr if result.returncode != 0 else None
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "avg_return": None, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "avg_return": None, "error": str(e)}


# Cross-game task mapping (mirrors run_baselines_cross_game.py)
GAME_SEQUENCE = ["ALE/Breakout-v5", "ALE/Freeway-v5", "ALE/SpaceInvaders-v5"]
GAME_MODES = {
    "ALE/Breakout-v5": [0],
    "ALE/Freeway-v5": list(range(8)),
    "ALE/SpaceInvaders-v5": list(range(10)),
}
TOTAL_TASKS = sum(len(m) for m in GAME_MODES.values())


def task_id_to_game_mode(task_id):
    """Map task_id to (game, mode) based on the training sequence."""
    for game in GAME_SEQUENCE:
        modes = GAME_MODES[game]
        if task_id < len(modes):
            return game, modes[task_id]
        task_id -= len(modes)
    return None, None


def get_max_task_dir(model_path):
    """Find the highest task_N directory under model_path."""
    max_id = -1
    max_dir = None
    for entry in glob.glob(f"{model_path}/task_*"):
        if not os.path.isdir(entry):
            continue
        m = re.match(r"^task_(\d+)$", os.path.basename(entry))
        if m:
            tid = int(m.group(1))
            if tid > max_id:
                max_id = tid
                max_dir = entry
    return max_dir, max_id


def main():
    args = parse_args()

    # Find all trained models
    if not os.path.exists(args.agents_dir):
        print(f"Error: Agents directory '{args.agents_dir}' not found.")
        print("Run benchmark_experiments.sh first to train models.")
        return

    model_dirs = glob.glob(f"{args.agents_dir}/*")

    if not model_dirs:
        print(f"No models found in '{args.agents_dir}'")
        return

    print(f"Found {len(model_dirs)} models to evaluate")
    print("=" * 60)

    results = []

    for model_path in sorted(model_dirs):
        # Parse model name to get metadata
        model_name = os.path.basename(model_path)
        try:
            parsed = parse_name_info(model_name)
            if len(parsed) == 4:
                env_name, mode, algorithm, seed = parsed
            elif len(parsed) == 2:
                algorithm, seed = parsed
                env_name, mode = None, None
            else:
                raise ValueError("Unexpected parse_name_info return length")
        except (ValueError, IndexError) as e:
            print(f"Could not parse model name: {model_name}")
            continue

        # Cross-game models have task_N subdirectories
        task_pattern = re.compile(r"^task_(\d+)$")
        task_entries = sorted(glob.glob(f"{model_path}/task_*"))
        task_dirs = []
        task_ids = []
        for td in task_entries:
            if not os.path.isdir(td):
                continue
            m = task_pattern.match(os.path.basename(td))
            if m:
                task_dirs.append(td)
                task_ids.append(int(m.group(1)))

        if not task_dirs:
            # Single-checkpoint model (legacy structure)
            task_dirs = [model_path]
            task_ids = [None]

        print(f"\nEvaluating: {model_name}")
        print(f"  Algorithm: {algorithm}, Seed: {seed}, Tasks: {len(task_dirs)}")

        for task_dir, task_id in zip(task_dirs, task_ids):
            if task_id is not None:
                game, mode_val = task_id_to_game_mode(task_id)
                if game is None:
                    continue
            else:
                game, mode_val = env_name, mode

            # Skip if we cannot determine the evaluation environment
            if game is None or mode_val is None:
                print(f"  Skipping {task_dir}: cannot determine game/mode")
                continue

            # Filter by environment if specified
            if args.env and game != args.env:
                continue

            # Build test_agent.py command
            cmd = [
                "python3", "test_agent.py",
                "--load", task_dir,
                "--algorithm", algorithm if algorithm != "fame" else "FAME",
                "--seed", str(seed),
                "--test-env", str(game),
                "--test-mode", str(mode_val),
                "--cross-game",
                "--num-episodes", str(args.num_episodes),
                "--max-timesteps", str(args.max_timesteps)
            ]

            # CompoNet and ProgressiveNet need previous units
            if algorithm in ("cnn-componet", "prog-net") and task_id is not None and task_id > 0:
                prev_units = [f"{model_path}/task_{p}" for p in range(task_id)]
                cmd += ["--prev-units"] + prev_units

            # PackNet needs special handling
            if algorithm == "packnet" and task_id is not None:
                final_dir, final_task_id = get_max_task_dir(model_path)
                if final_dir is None:
                    print(f"  Skipping packnet task {task_id}: no checkpoints found")
                    continue
                cmd = [
                    "python3", "test_agent.py",
                    "--load", final_dir,
                    "--algorithm", algorithm,
                    "--seed", str(seed),
                    "--mode", str(task_id),
                    "--test-env", str(game),
                    "--test-mode", str(mode_val),
                    "--task-checkpoint", task_dir,
                    "--cross-game",
                    "--num-episodes", str(args.num_episodes),
                    "--max-timesteps", str(args.max_timesteps)
                ]

            eval_result = evaluate_model(cmd, args.num_episodes, args.max_timesteps)

            if eval_result["success"]:
                print(f"  ✓ task_{task_id} ({game} mode {mode_val}): Avg return {eval_result['avg_return']}")
            else:
                print(f"  ✗ task_{task_id} ({game} mode {mode_val}): Failed: {eval_result['error']}")

            results.append({
                "model_path": task_dir,
                "model_name": model_name,
                "algorithm": algorithm,
                "environment": game,
                "mode": mode_val,
                "seed": seed,
                "task_id": task_id,
                "avg_return": eval_result["avg_return"],
                "success": eval_result["success"],
                "error": eval_result["error"]
            })

    # Create summary DataFrame
    df = pd.DataFrame(results)

    # Save results
    if not df.empty:
        df.to_csv(args.output, index=False)
        print(f"\n{'=' * 60}")
        print(f"Results saved to: {args.output}")
    else:
        print(f"\n{'=' * 60}")
        print("No results to save.")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY BY ALGORITHM")
    print("=" * 60)

    if df.empty or "algorithm" not in df.columns:
        print("No evaluation results to summarize.")
        return

    summary = df.groupby("algorithm").agg({
        "avg_return": ["mean", "std", "count"],
        "success": "sum"
    }).round(2)

    print(summary.to_string())

    # Print success rate
    print("\n" + "=" * 60)
    print("SUCCESS RATE BY ALGORITHM")
    print("=" * 60)

    success_rate = df.groupby("algorithm")["success"].agg(["sum", "count"])
    success_rate["rate"] = (success_rate["sum"] / success_rate["count"] * 100).round(1)
    print(success_rate.to_string())


if __name__ == "__main__":
    main()
