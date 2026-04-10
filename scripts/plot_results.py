"""
Plotting utilities for Continual RL Benchmark.

Generates comparison plots for all methods.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Color palette for methods
METHOD_COLORS = {
    "base": "#1f77b4",
    "fast_trac": "#ff7f0e",
    "parseval": "#2ca02c",
    "dual_learner": "#d62728",
    "world_model": "#9467bd",
    "dual_actor": "#8c564b",
}

METHOD_LABELS = {
    "base": "SAC (Baseline)",
    "fast_trac": "Fast TRAC (NeurIPS 2024)",
    "parseval": "Parseval Reg. (NeurIPS 2024)",
    "dual_learner": "Dual-Learner (ICLR 2026)",
    "world_model": "World Model + MPC (ICML 2025)",
    "dual_actor": "Dual Actor (ICLR 2025)",
}


def load_results(results_dir: str) -> Dict:
    """Load aggregate results from JSON file."""
    results_files = list(Path(results_dir).glob("aggregate_results_*.json"))
    if not results_files:
        raise FileNotFoundError(f"No results found in {results_dir}")

    # Load most recent results
    results_file = sorted(results_files)[-1]
    with open(results_file, "r") as f:
        return json.load(f)


def plot_learning_curves(
    results: Dict,
    output_dir: str,
    metric: str = "average_return",
):
    """Plot learning curves for all methods."""
    plt.figure(figsize=(12, 8))

    for method, seeds in results.items():
        if method == "metadata":
            continue

        # Collect metric values across seeds
        values = []
        for seed, metrics in seeds.items():
            if "error" not in metrics:
                values.append(metrics.get(metric, 0))

        if not values:
            continue

        mean_val = np.mean(values)
        std_val = np.std(values) if len(values) > 1 else 0

        plt.errorbar(
            METHOD_LABELS.get(method, method),
            mean_val,
            yerr=std_val,
            capsize=5,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "gray"),
            marker="o",
            markersize=8,
        )

    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Continual RL Benchmark - {metric.replace('_', ' ').title()}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{metric}_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_forgetting_comparison(results: Dict, output_dir: str):
    """Plot forgetting measure comparison."""
    plot_learning_curves(results, output_dir, metric="forgetting_measure")


def plot_transfer_comparison(results: Dict, output_dir: str):
    """Plot forward transfer comparison."""
    plot_learning_curves(results, output_dir, metric="forward_transfer")


def plot_training_curves(csv_dir: str, output_dir: str):
    """Plot training curves over steps."""
    plt.figure(figsize=(14, 8))

    csv_files = list(Path(csv_dir).glob("**/metrics.csv"))

    # Group by method
    method_curves = {}

    for csv_file in csv_files:
        # Parse method name from path
        parts = str(csv_file).split(os.sep)
        method = None
        for part in parts:
            if part in METHOD_COLORS:
                method = part
                break

        if method is None:
            continue

        # Load CSV
        import pandas as pd
        df = pd.read_csv(csv_file)

        if "return" not in df.columns:
            continue

        # Smooth returns
        window = 10
        df["smoothed_return"] = df["return"].rolling(window=window, min_periods=1).mean()

        if method not in method_curves:
            method_curves[method] = []

        method_curves[method].append(df[["step", "smoothed_return"]])

    # Plot with confidence intervals
    for method, curves in method_curves.items():
        # Find max step for alignment
        max_step = max(c["step"].max() for c in curves)

        # Interpolate to common x-axis
        x_common = np.linspace(0, max_step, 100)
        y_values = []

        for curve in curves:
            y_interp = np.interp(
                x_common,
                curve["step"].values,
                curve["smoothed_return"].values,
            )
            y_values.append(y_interp)

        y_mean = np.mean(y_values, axis=0)
        y_std = np.std(y_values, axis=0)

        plt.plot(
            x_common, y_mean,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "gray"),
            linewidth=2,
        )
        plt.fill_between(
            x_common,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.2,
            color=METHOD_COLORS.get(method, "gray"),
        )

    plt.xlabel("Training Steps")
    plt.ylabel("Smoothed Episode Return")
    plt.title("Continual RL Training Curves")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def generate_all_plots(results_dir: str, output_dir: str):
    """Generate all standard plots."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        results = load_results(results_dir)

        # Metric comparisons
        plot_learning_curves(results, output_dir, "average_return")
        plot_learning_curves(results, output_dir, "forward_transfer")
        plot_learning_curves(results, output_dir, "backward_transfer")
        plot_learning_curves(results, output_dir, "forgetting_measure")

        print(f"\nAll plots saved to: {output_dir}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run experiments first to generate results.")


def main():
    parser = argparse.ArgumentParser(description="Plot Continual RL Results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="all",
        choices=["all", "return", "forgetting", "transfer", "curves"],
        help="Type of plot to generate",
    )

    args = parser.parse_args()

    if args.plot_type == "all":
        generate_all_plots(args.results_dir, args.output_dir)
    elif args.plot_type == "curves":
        plot_training_curves(
            os.path.join(args.results_dir, "csv"),
            args.output_dir,
        )
    else:
        results = load_results(args.results_dir)
        if args.plot_type == "return":
            plot_learning_curves(results, args.output_dir, "average_return")
        elif args.plot_type == "forgetting":
            plot_forgetting_comparison(results, args.output_dir)
        elif args.plot_type == "transfer":
            plot_transfer_comparison(results, args.output_dir)


if __name__ == "__main__":
    main()
