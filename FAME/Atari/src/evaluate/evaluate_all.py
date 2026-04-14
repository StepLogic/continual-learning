#!/usr/bin/env python3
"""
Evaluate all trained agents and generate results summary.

Usage:
    python3 evaluate_all.py [--agents-dir=agents] [--env=ALE/Freeway-v5] [--output=results.csv]
"""

import os
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


def evaluate_model(model_path, num_episodes, max_timesteps):
    """Evaluate a single model and return the results."""
    cmd = [
        "python3", "test_agent.py",
        "--load", model_path,
        "--num-episodes", str(num_episodes),
        "--max-timesteps", str(max_timesteps)
    ]

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
            env_name, mode, algorithm, seed = parse_name_info(model_name)
        except (ValueError, IndexError) as e:
            print(f"Could not parse model name: {model_name}")
            continue

        # Filter by environment if specified
        if args.env and env_name != args.env:
            continue

        print(f"\nEvaluating: {model_name}")
        print(f"  Environment: {env_name}, Mode: {mode}, Algorithm: {algorithm}, Seed: {seed}")

        eval_result = evaluate_model(
            model_path,
            args.num_episodes,
            args.max_timesteps
        )

        if eval_result["success"]:
            print(f"  ✓ Success! Avg return: {eval_result['avg_return']}")
        else:
            print(f"  ✗ Failed: {eval_result['error']}")

        results.append({
            "model_path": model_path,
            "model_name": model_name,
            "algorithm": algorithm,
            "environment": env_name,
            "mode": mode,
            "seed": seed,
            "avg_return": eval_result["avg_return"],
            "success": eval_result["success"],
            "error": eval_result["error"]
        })

    # Create summary DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv(args.output, index=False)
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {args.output}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY BY ALGORITHM")
    print("=" * 60)

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
