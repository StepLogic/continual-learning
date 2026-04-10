"""
Continual RL Evaluation Metrics.

Implements standard metrics for evaluating continual learning:
- Average Return
- Forward Transfer (plasticity)
- Backward Transfer (forgetting)
- Learning Curve
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class ContinualMetrics:
    """
    Metrics tracker for continual RL experiments.

    Tracks per-task performance and computes continual learning metrics.
    """

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks

        # Per-task performance history
        self.task_returns = defaultdict(list)  # task_idx -> list of avg returns
        self.task_steps = defaultdict(list)    # task_idx -> list of step counts

        # Overall metrics
        self.all_returns = []
        self.all_steps = []

        # Task-specific tracking
        self.current_task_idx = 0
        self.episode_returns = []
        self.best_performance = defaultdict(float)  # Best performance per task

    def log_episode(self, episode_return: float, task_idx: int, global_step: int):
        """Log episode return."""
        self.episode_returns.append(episode_return)
        self.all_returns.append(episode_return)
        self.all_steps.append(global_step)

    def end_task(self, task_idx: int):
        """Called when switching to a new task."""
        # Compute average return for this task
        if self.episode_returns:
            avg_return = np.mean(self.episode_returns[-100:])  # Last 100 episodes
            self.task_returns[task_idx].append(avg_return)

            # Track best performance
            if avg_return > self.best_performance[task_idx]:
                self.best_performance[task_idx] = avg_return

        self.episode_returns = []

    def compute_average_return(self) -> float:
        """Compute overall average return."""
        if not self.all_returns:
            return 0.0
        return np.mean(self.all_returns)

    def compute_forward_transfer(self) -> float:
        """
        Compute forward transfer (plasticity).

        Measures how much prior learning helps on new tasks.
        Positive = positive transfer, Negative = negative transfer.
        """
        if len(self.task_returns) < 2:
            return 0.0

        # Compare first task performance to later tasks
        first_task_perf = np.mean(self.task_returns[0]) if self.task_returns[0] else 0

        later_perfs = []
        for idx in range(1, self.num_tasks):
            if self.task_returns[idx]:
                later_perfs.append(np.mean(self.task_returns[idx]))

        if not later_perfs:
            return 0.0

        avg_later_perf = np.mean(later_perfs)
        return avg_later_perf - first_task_perf

    def compute_backward_transfer(self) -> float:
        """
        Compute backward transfer (forgetting).

        Measures how much learning new tasks affects old tasks.
        Negative = forgetting occurred.
        """
        if len(self.task_returns) < 2:
            return 0.0

        forgetting_scores = []
        for task_idx in range(self.num_tasks - 1):
            if len(self.task_returns[task_idx]) >= 2:
                # Compare initial vs final performance on each task
                initial = self.task_returns[task_idx][0]
                final = self.task_returns[task_idx][-1]
                forgetting_scores.append(final - initial)

        if not forgetting_scores:
            return 0.0

        return np.mean(forgetting_scores)

    def compute_forgetting_measure(self) -> float:
        """
        Compute standard forgetting measure.

        FM = (1/T) * sum(max(R_t^k) - R_t^T)
        where R_t^k is performance on task t after learning task k.
        """
        forgetting = 0.0
        count = 0

        for task_idx in range(self.num_tasks - 1):
            if not self.task_returns[task_idx]:
                continue

            max_perf = max(self.task_returns[task_idx])
            final_perf = self.task_returns[task_idx][-1]
            forgetting += max_perf - final_perf
            count += 1

        if count == 0:
            return 0.0

        return forgetting / count

    def compute_learning_curve(self) -> np.ndarray:
        """
        Compute learning curve (performance over time).

        Returns smoothed average returns over time.
        """
        if len(self.all_returns) < 10:
            return np.array(self.all_returns)

        # Smooth with window of 10 episodes
        window = min(10, len(self.all_returns) // 5)
        if window < 2:
            return np.array(self.all_returns)

        smoothed = []
        for i in range(len(self.all_returns)):
            start = max(0, i - window)
            smoothed.append(np.mean(self.all_returns[start:i+1]))

        return np.array(smoothed)

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {
            "average_return": self.compute_average_return(),
            "forward_transfer": self.compute_forward_transfer(),
            "backward_transfer": self.compute_backward_transfer(),
            "forgetting_measure": self.compute_forgetting_measure(),
            "total_episodes": len(self.all_returns),
            "total_tasks_seen": len([t for t in self.task_returns if self.task_returns[t]]),
        }

    def get_task_performance(self, task_idx: int) -> Optional[float]:
        """Get average performance for a specific task."""
        if not self.task_returns[task_idx]:
            return None
        return np.mean(self.task_returns[task_idx])

    def save(self, path: str):
        """Save metrics to file."""
        import json
        data = {
            "task_returns": {str(k): v for k, v in self.task_returns.items()},
            "all_returns": self.all_returns,
            "all_steps": self.all_steps,
            "best_performance": dict(self.best_performance),
            "summary": self.get_summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load metrics from file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        self.task_returns = defaultdict(list, {int(k): v for k, v in data["task_returns"].items()})
        self.all_returns = data["all_returns"]
        self.all_steps = data["all_steps"]
        self.best_performance = {int(k): v for k, v in data["best_performance"].items()}
