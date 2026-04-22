#!/usr/bin/env python3
"""Generate a LaTeX table from cross-game evaluation CSV.

Supports both the corrected 5-column format and the legacy 6-column format.
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def parse_csv(path: str):
    """Parse CSV, handling both 5-column and 6-column formats."""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)

        for row in reader:
            if len(row) != n_cols:
                # Skip malformed rows
                continue

            if n_cols == 5:
                # Corrected format: train_game,train_mode,test_game,test_mode,avg_return
                train_game, train_mode, test_game, test_mode, avg_return = row
            elif n_cols == 6:
                # Legacy format from test_agent.py: algorithm,environment,train_mode,test_mode,seed,ep_ret
                # We only care about test_game (environment), test_mode, and avg_return (ep_ret)
                algorithm, test_game, train_mode, test_mode, seed, avg_return = row
            else:
                continue

            try:
                avg_return = float(avg_return)
            except ValueError:
                continue

            rows.append({
                "test_game": test_game,
                "test_mode": int(test_mode) if test_mode.strip() else 0,
                "avg_return": avg_return,
            })
    return rows


def compute_stats(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["test_game"], r["test_mode"])].append(r["avg_return"])

    results = []
    for (game, mode), returns in sorted(groups.items()):
        mean = statistics.mean(returns)
        std = statistics.stdev(returns) if len(returns) > 1 else 0.0
        results.append({
            "game": game,
            "mode": mode,
            "mean": mean,
            "std": std,
            "n": len(returns),
        })
    return results


def format_game_name(raw: str) -> str:
    return raw.replace("ALE/", "").replace("-v5", "")


def generate_latex(stats, seed: int = 1):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Cross-Game Evaluation Results (seed=" + str(seed) + r")}",
        r"\label{tab:cross-game}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Environment & Mode & Mean Return & Std Dev \\",
        r"\midrule",
    ]

    current_game = None
    for s in stats:
        game_name = format_game_name(s["game"])
        if current_game != game_name:
            if current_game is not None:
                lines.append(r"\midrule")
            current_game = game_name
            lines.append(
                f"{game_name} & {s['mode']} & {s['mean']:.2f} & {s['std']:.2f} \\\\"
            )
        else:
            lines.append(
                f" & {s['mode']} & {s['mean']:.2f} & {s['std']:.2f} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from cross-game eval CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to eval CSV")
    parser.add_argument("--seed", type=int, default=1, help="Seed for caption")
    parser.add_argument("--output", type=str, default=None, help="Output .tex file (default: stdout)")
    args = parser.parse_args()

    rows = parse_csv(args.csv)
    if not rows:
        print("No valid rows found in CSV.")
        return

    stats = compute_stats(rows)
    latex = generate_latex(stats, seed=args.seed)

    if args.output:
        Path(args.output).write_text(latex)
        print(f"LaTeX table written to {args.output}")
    else:
        print(latex)


if __name__ == "__main__":
    main()
