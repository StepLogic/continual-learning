"""
Generate summary report for Continual RL Benchmark.

Creates a markdown report with all results and analysis.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np


METHOD_NAMES = {
    "base": "SAC (Baseline)",
    "fast_trac": "Fast TRAC",
    "parseval": "Parseval Regularization",
    "dual_learner": "Dual-Learner",
    "world_model": "World Model + MPC",
    "dual_actor": "Dual Actor",
}

METHOD_VENUES = {
    "base": "-",
    "fast_trac": "NeurIPS 2024",
    "parseval": "NeurIPS 2024",
    "dual_learner": "ICLR 2026",
    "world_model": "ICML 2025",
    "dual_actor": "ICLR 2025",
}


def load_results(results_dir: str) -> Dict:
    """Load aggregate results."""
    results_files = list(Path(results_dir).glob("aggregate_results_*.json"))
    if not results_files:
        return None

    with open(sorted(results_files)[-1], "r") as f:
        return json.load(f)


def compute_rankings(results: Dict) -> List[tuple]:
    """Compute method rankings by average return."""
    rankings = []

    for method, seeds in results.items():
        if method == "metadata":
            continue

        returns = []
        for seed, metrics in seeds.items():
            if "error" not in metrics:
                returns.append(metrics.get("average_return", 0))

        if returns:
            mean_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0
            rankings.append((method, mean_return, std_return))

    # Sort by mean return (descending)
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


def compute_forgetting_rankings(results: Dict) -> List[tuple]:
    """Compute rankings by forgetting measure (lower is better)."""
    rankings = []

    for method, seeds in results.items():
        if method == "metadata":
            continue

        forgetting = []
        for seed, metrics in seeds.items():
            if "error" not in metrics:
                forgetting.append(metrics.get("forgetting_measure", 0))

        if forgetting:
            mean_forgetting = np.mean(forgetting)
            std_forgetting = np.std(forgetting) if len(forgetting) > 1 else 0
            rankings.append((method, mean_forgetting, std_forgetting))

    # Sort by forgetting (ascending - lower is better)
    rankings.sort(key=lambda x: x[1])
    return rankings


def generate_report(results: Dict, output_path: str):
    """Generate markdown report."""

    report = []
    report.append("# Continual RL Benchmark Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Summary table
    report.append("## Summary")
    report.append("")
    report.append("| Method | Venue | Avg Return | Std | Forgetting | Rank |")
    report.append("|--------|-------|------------|-----|------------|------|")

    rankings = compute_rankings(results)
    forgetting_rankings = compute_forgetting_rankings(results)

    # Create forgetting lookup
    forgetting_lookup = {m: (f, s) for m, f, s in forgetting_rankings}

    for rank, (method, mean_ret, std_ret) in enumerate(rankings, 1):
        name = METHOD_NAMES.get(method, method)
        venue = METHOD_VENUES.get(method, "-")
        forgetting, forget_std = forgetting_lookup.get(method, (0, 0))

        report.append(
            f"| {rank}. **{name}** | {venue} | {mean_ret:.2f} | {std_ret:.2f} | "
            f"{forgetting:.2f} ({forget_std:.2f}) | {rank} |"
        )

    report.append("")

    # Detailed results per method
    report.append("## Detailed Results")
    report.append("")

    for method, seeds in results.items():
        if method == "metadata":
            continue

        name = METHOD_NAMES.get(method, method)
        venue = METHOD_VENUES.get(method, "-")

        report.append(f"### {name} ({venue})")
        report.append("")

        # Per-seed results
        report.append("| Seed | Avg Return | Forward Transfer | Backward Transfer | Forgetting |")
        report.append("|------|------------|------------------|-------------------|------------|")

        for seed, metrics in sorted(seeds.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            if "error" in metrics:
                report.append(f"| {seed} | ERROR: {metrics['error']} | - | - | - |")
            else:
                avg_ret = metrics.get("average_return", 0)
                fwd_trans = metrics.get("forward_transfer", 0)
                bwd_trans = metrics.get("backward_transfer", 0)
                forgetting = metrics.get("forgetting_measure", 0)

                report.append(
                    f"| {seed} | {avg_ret:.2f} | {fwd_trans:.2f} | {bwd_trans:.2f} | {forgetting:.2f} |"
                )

        report.append("")

    # Key findings
    report.append("## Key Findings")
    report.append("")

    if rankings:
        best_method = rankings[0]
        report.append(
            f"1. **Best Performance:** {METHOD_NAMES.get(best_method[0], best_method[0])} "
            f"achieved the highest average return ({best_method[1]:.2f} ± {best_method[2]:.2f})"
        )

    if forgetting_rankings:
        best_forgetting = forgetting_rankings[0]
        report.append(
            f"2. **Least Forgetting:** {METHOD_NAMES.get(best_forgetting[0], best_forgetting[0])} "
            f"showed the least catastrophic forgetting ({best_forgetting[1]:.2f})"
        )

    # Check for positive transfer
    positive_transfer = []
    for method, seeds in results.items():
        if method == "metadata":
            continue
        for seed, metrics in seeds.items():
            if "error" not in metrics and metrics.get("forward_transfer", 0) > 0:
                positive_transfer.append(method)
                break

    if positive_transfer:
        report.append(
            f"3. **Positive Transfer:** Methods showing positive forward transfer: "
            f"{', '.join(set(METHOD_NAMES.get(m, m) for m in positive_transfer))}"
        )

    report.append("")

    # Method descriptions
    report.append("## Method Descriptions")
    report.append("")
    report.append("""
### Fast TRAC (NeurIPS 2024)
Parameter-free optimizer based on online convex optimization principles.
Adapts learning rates per-parameter to address plasticity loss without manual tuning.

### Parseval Regularization (NeurIPS 2024)
Maintains orthogonality of weight matrices during training to preserve plasticity
when learning new tasks. Uses QR decomposition for explicit orthogonalization.

### Dual-Learner (ICLR 2026)
Inspired by hippocampus-cortex interaction:
- Fast learner: Rapid adaptation to new tasks
- Meta learner: Slow knowledge integration
Uses gating mechanism to combine outputs.

### World Model + MPC (ICML 2025)
Learns online world dynamics model and plans via Model Predictive Control.
Uses imagined transitions for data augmentation to handle forgetting.

### Dual Actor (ICLR 2025)
Two-policy architecture:
- Current actor: Periodically resets for plasticity
- Distilled actor: Accumulates knowledge via behavior cloning
Balances stability-plasticity tradeoff.
""")

    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated by Continual RL Benchmark*")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Continual RL Report")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/reports/benchmark_report.md",
        help="Output path for report",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = load_results(args.results_dir)
    if results is None:
        print("No results found. Run experiments first.")
        return

    generate_report(results, args.output)


if __name__ == "__main__":
    main()
