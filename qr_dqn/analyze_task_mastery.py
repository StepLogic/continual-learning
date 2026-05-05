"""
Analyze per-task mastery metrics from continual learning experiments.

Usage:
    python qr_dqn/analyze_task_mastery.py results/qr_dqn_per_task_mastery_seed42.json

Outputs a markdown report with:
- Mastery status per task
- Sample efficiency comparison
- Stability analysis
- Recommendations for budget allocation
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_mastery_metrics(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def format_report(data: Dict) -> str:
    tasks = data["tasks"]
    overall = data["overall"]

    lines = []
    lines.append("# Per-Task Mastery Analysis Report")
    lines.append("")
    lines.append(f"**Total tasks:** {overall['num_tasks']}")
    lines.append(f"**Tasks mastered:** {overall['tasks_mastered']} / {overall['num_tasks']}")
    lines.append(f"**Avg normalized score:** {overall['avg_normalized_score']:.3f}")
    lines.append(f"**Avg sample efficiency:** {overall['avg_sample_efficiency']:.3f}")
    lines.append("")

    # Per-task table
    lines.append("## Per-Task Breakdown")
    lines.append("")
    lines.append("| Task | Mastered | Step | Episodes | Mean Return | Best | CV | Stability | Plateau | Norm. Score | Sample Eff. |")
    lines.append("|------|----------|------|----------|-------------|------|-----|-----------|---------|-------------|-------------|")

    for t in tasks:
        mastered = "Yes" if t["mastery_achieved"] else "No"
        lines.append(
            f"| {t['task_name']} | {mastered} | {t['mastery_step'] or '--'} | "
            f"{t['total_episodes']} | {t['final_mean_return']:.1f} ± {t['final_std_return']:.1f} | "
            f"{t['best_return']:.1f} | {t['cv_recent']:.3f} | "
            f"{'Yes' if t['stability_achieved'] else 'No'} | "
            f"{'Yes' if t['plateau_achieved'] else 'No'} | "
            f"{t['normalized_score'] or '--':.3f} | {t['sample_efficiency'] or '--':.3f} |"
        )

    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    unmastered = [t for t in tasks if not t["mastery_achieved"]]
    if unmastered:
        lines.append(f"**Warning:** {len(unmastered)} task(s) did NOT achieve mastery:")
        for t in unmastered:
            lines.append(f"- {t['task_name']}: stopped at step {t['total_steps']}, CV={t['cv_recent']:.3f}")
        lines.append("")
        lines.append("**Suggestion:** Increase `steps_per_task` or relax mastery criteria for these tasks.")
        lines.append("")

    mastered = [t for t in tasks if t["mastery_achieved"]]
    if mastered:
        avg_mastery_step = np.mean([t["mastery_step"] for t in mastered if t["mastery_step"] is not None])
        max_mastery_step = max([t["mastery_step"] for t in mastered if t["mastery_step"] is not None])
        lines.append(f"**Optimization:** Mastered tasks converged on average at step {avg_mastery_step:.0f}.")
        lines.append(f"- Slowest to master: {max_mastery_step:.0f} steps")
        lines.append(f"- Current fixed budget: check your config (likely 500k)")
        if avg_mastery_step < 400_000:
            lines.append("")
            lines.append("**Opportunity:** You could save ~{:.0f}% training time by using mastery-based stopping instead of fixed steps.".format(
                100 * (1 - avg_mastery_step / 500_000)
            ))
        lines.append("")

    # Stability analysis
    high_cv = [t for t in tasks if t["cv_recent"] > 0.2]
    if high_cv:
        lines.append("**Instability detected:** High coefficient of variation (>0.2) in:")
        for t in high_cv:
            lines.append(f"- {t['task_name']}: CV={t['cv_recent']:.3f}")
        lines.append("")
        lines.append("**Interpretation:** High CV indicates the agent is still exploring or the environment is noisy. Consider longer stability windows or higher evaluation episodes.")
        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_task_mastery.py <mastery_metrics.json>")
        sys.exit(1)

    path = sys.argv[1]
    data = load_mastery_metrics(path)
    report = format_report(data)

    # Print to stdout
    print(report)

    # Save to file
    out_path = Path(path).with_suffix(".md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
