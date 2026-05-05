"""
Task Mastery and Per-Task Learning Verification.

Provides stopping criteria and detailed metrics for determining when
an RL agent has sufficiently learned a task before moving to the next.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json


@dataclass
class TaskMasteryConfig:
    """Configuration for task mastery criteria."""
    # Minimum requirements
    min_episodes: int = 50           # Don't stop before this many episodes
    min_steps: int = 50_000          # Minimum steps per task

    # Performance threshold (optional)
    performance_threshold: Optional[float] = None  # Absolute return threshold
    threshold_percentile: Optional[float] = None   # Percentile of random baseline

    # Stability criterion
    stability_window: int = 20       # Episodes to check stability
    max_cv: float = 0.15           # Coefficient of variation threshold (lower = more stable)

    # Plateau detection
    plateau_window: int = 30       # Episodes to check for plateau
    plateau_tolerance: float = 0.02  # Fractional improvement threshold

    # Combined: require ALL active criteria to be met
    require_stability: bool = True
    require_plateau: bool = True
    require_threshold: bool = False


@dataclass
class TaskLearningMetrics:
    """Comprehensive metrics for a single task's learning progress."""
    task_idx: int
    task_name: str

    # Learning curve
    episode_returns: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    eval_returns: List[float] = field(default_factory=list)
    eval_steps: List[int] = field(default_factory=list)

    # Mastery status
    mastery_achieved: bool = False
    mastery_step: Optional[int] = None
    mastery_episode: Optional[int] = None

    # Key metrics
    final_mean_return: float = 0.0
    final_std_return: float = 0.0
    best_return: float = 0.0
    worst_return: float = 0.0

    # Stability metrics
    cv_recent: float = float('inf')          # Coefficient of variation (recent window)
    stability_achieved: bool = False

    # Plateau metrics
    plateau_achieved: bool = False
    plateau_step: Optional[int] = None

    # Sample efficiency
    steps_to_threshold: Optional[int] = None
    episodes_to_threshold: Optional[int] = None

    # Normalized scores
    normalized_score: Optional[float] = None   # (score - random) / (expert - random)
    sample_efficiency: Optional[float] = None  # AUC of normalized learning curve

    # Random baseline (for normalization)
    random_baseline: Optional[float] = None
    expert_baseline: Optional[float] = None


class TaskMasteryTracker:
    """
    Tracks whether an agent has mastered a task and decides when to stop training.

    Usage:
        tracker = TaskMasteryTracker(config, task_idx=0, task_name="Breakout")

        # Inside training loop:
        for step in range(max_steps):
            # ... train ...
            if done:
                tracker.log_episode(episode_return, step)

            if step % eval_interval == 0:
                tracker.log_eval(eval_return, step)

            if tracker.check_mastery(step):
                print(f"Task mastered at step {step}!")
                break
    """

    def __init__(self, config: TaskMasteryConfig, task_idx: int, task_name: str,
                 random_baseline: Optional[float] = None,
                 expert_baseline: Optional[float] = None):
        self.config = config
        self.task_idx = task_idx
        self.task_name = task_name
        self.random_baseline = random_baseline
        self.expert_baseline = expert_baseline

        # Internal buffers for efficient computation
        self._recent_returns = deque(maxlen=max(config.stability_window, config.plateau_window))
        self._all_returns = []
        self._all_steps = []
        self._eval_returns = []
        self._eval_steps = []

        self._mastery_achieved = False
        self._mastery_step = None
        self._mastery_episode = None
        self._plateau_step = None
        self._steps_to_threshold = None
        self._episodes_to_threshold = None

    def log_episode(self, episode_return: float, step: int):
        """Log an episode return."""
        self._recent_returns.append(episode_return)
        self._all_returns.append(episode_return)
        self._all_steps.append(step)

        # Check if we just crossed the threshold
        if (self.config.performance_threshold is not None and
            self._steps_to_threshold is None and
            episode_return >= self.config.performance_threshold):
            self._steps_to_threshold = step
            self._episodes_to_threshold = len(self._all_returns)

    def log_eval(self, eval_return: float, step: int):
        """Log a periodic evaluation return."""
        self._eval_returns.append(eval_return)
        self._eval_steps.append(step)

    def _check_stability(self) -> bool:
        """Check if recent returns are stable (low coefficient of variation)."""
        if len(self._recent_returns) < self.config.stability_window:
            return False

        recent = list(self._recent_returns)[-self.config.stability_window:]
        mean = np.mean(recent)
        std = np.std(recent)

        if mean == 0:
            return std == 0

        cv = std / abs(mean)
        return cv <= self.config.max_cv

    def _check_plateau(self, step: int) -> bool:
        """
        Check if performance has plateaued (no significant improvement).

        Uses a simple regression-based approach: if the slope of recent returns
        is not significantly positive, we've plateaued.
        """
        if len(self._all_returns) < self.config.plateau_window:
            return False

        # Compare first half to second half of plateau window
        recent = self._all_returns[-self.config.plateau_window:]
        mid = len(recent) // 2
        first_half = np.mean(recent[:mid])
        second_half = np.mean(recent[mid:])

        # Check if improvement is below tolerance
        if first_half == 0:
            improvement = abs(second_half - first_half)
        else:
            improvement = (second_half - first_half) / abs(first_half)

        plateau = improvement < self.config.plateau_tolerance

        if plateau and self._plateau_step is None:
            self._plateau_step = step

        return plateau

    def _check_threshold(self) -> bool:
        """Check if performance threshold is met."""
        if self.config.performance_threshold is None:
            return True  # Not required

        if len(self._recent_returns) < self.config.stability_window:
            return False

        recent_mean = np.mean(list(self._recent_returns)[-self.config.stability_window:])
        return recent_mean >= self.config.performance_threshold

    def check_mastery(self, step: int) -> bool:
        """
        Check if task mastery criteria are met.

        Returns True if ALL active criteria are satisfied:
        - Minimum episodes/steps reached
        - Stability (if required)
        - Plateau (if required)
        - Performance threshold (if required)
        """
        if self._mastery_achieved:
            return True

        # Minimum requirements
        if len(self._all_returns) < self.config.min_episodes:
            return False
        if step < self.config.min_steps:
            return False

        # Check active criteria
        stability = self._check_stability() if self.config.require_stability else True
        plateau = self._check_plateau(step) if self.config.require_plateau else True
        threshold = self._check_threshold() if self.config.require_threshold else True

        if stability and plateau and threshold:
            self._mastery_achieved = True
            self._mastery_step = step
            self._mastery_episode = len(self._all_returns)
            return True

        return False

    def compute_normalized_score(self) -> Optional[float]:
        """
        Compute normalized score: (score - random) / (expert - random).

        Returns None if baselines not provided.
        """
        if self.random_baseline is None or self.expert_baseline is None:
            return None

        if self.expert_baseline == self.random_baseline:
            return None

        final_score = np.mean(self._all_returns[-20:]) if len(self._all_returns) >= 20 else np.mean(self._all_returns)
        normalized = (final_score - self.random_baseline) / (self.expert_baseline - self.random_baseline)
        return float(np.clip(normalized, 0.0, 1.0))

    def compute_sample_efficiency(self) -> Optional[float]:
        """
        Compute sample efficiency as normalized area under the learning curve.

        Higher = learned faster. Returns None if baselines not provided.
        """
        if self.random_baseline is None or self.expert_baseline is None:
            return None

        if len(self._all_returns) < 2:
            return None

        # Normalize returns
        normalized = [(r - self.random_baseline) / (self.expert_baseline - self.random_baseline)
                      for r in self._all_returns]
        normalized = np.clip(normalized, 0.0, 1.0)

        # Compute area under curve using trapezoidal rule
        auc = np.trapezoid(normalized)
        # Normalize by maximum possible AUC (which would be len(returns) * 1.0)
        max_auc = len(normalized)
        return float(auc / max_auc) if max_auc > 0 else 0.0

    def get_metrics(self) -> TaskLearningMetrics:
        """Compile all metrics for this task."""
        if not self._all_returns:
            return TaskLearningMetrics(task_idx=self.task_idx, task_name=self.task_name)

        recent = list(self._recent_returns)[-self.config.stability_window:] if len(self._recent_returns) >= self.config.stability_window else list(self._recent_returns)

        return TaskLearningMetrics(
            task_idx=self.task_idx,
            task_name=self.task_name,
            episode_returns=list(self._all_returns),
            episode_steps=list(self._all_steps),
            eval_returns=list(self._eval_returns),
            eval_steps=list(self._eval_steps),
            mastery_achieved=self._mastery_achieved,
            mastery_step=self._mastery_step,
            mastery_episode=self._mastery_episode,
            final_mean_return=float(np.mean(self._all_returns[-20:])) if len(self._all_returns) >= 20 else float(np.mean(self._all_returns)),
            final_std_return=float(np.std(self._all_returns[-20:])) if len(self._all_returns) >= 20 else float(np.std(self._all_returns)),
            best_return=float(np.max(self._all_returns)),
            worst_return=float(np.min(self._all_returns)),
            cv_recent=float(np.std(recent) / abs(np.mean(recent))) if recent and np.mean(recent) != 0 else float('inf'),
            stability_achieved=self._check_stability(),
            plateau_achieved=self._plateau_step is not None,
            plateau_step=self._plateau_step,
            steps_to_threshold=self._steps_to_threshold,
            episodes_to_threshold=self._episodes_to_threshold,
            normalized_score=self.compute_normalized_score(),
            sample_efficiency=self.compute_sample_efficiency(),
            random_baseline=self.random_baseline,
            expert_baseline=self.expert_baseline,
        )

    def get_summary_dict(self) -> Dict:
        """Get a compact summary dict for JSON logging."""
        m = self.get_metrics()
        return {
            "task_idx": m.task_idx,
            "task_name": m.task_name,
            "mastery_achieved": m.mastery_achieved,
            "mastery_step": m.mastery_step,
            "mastery_episode": m.mastery_episode,
            "total_episodes": len(m.episode_returns),
            "total_steps": m.episode_steps[-1] if m.episode_steps else 0,
            "final_mean_return": m.final_mean_return,
            "final_std_return": m.final_std_return,
            "best_return": m.best_return,
            "cv_recent": m.cv_recent,
            "stability_achieved": m.stability_achieved,
            "plateau_achieved": m.plateau_achieved,
            "steps_to_threshold": m.steps_to_threshold,
            "normalized_score": m.normalized_score,
            "sample_efficiency": m.sample_efficiency,
        }


def save_per_task_metrics(all_trackers: List[TaskMasteryTracker], save_path: str):
    """Save comprehensive per-task metrics to JSON."""
    data = {
        "tasks": [t.get_summary_dict() for t in all_trackers],
        "overall": {
            "num_tasks": len(all_trackers),
            "tasks_mastered": sum(1 for t in all_trackers if t._mastery_achieved),
            "avg_normalized_score": np.mean([t.compute_normalized_score() or 0.0 for t in all_trackers]),
            "avg_sample_efficiency": np.mean([t.compute_sample_efficiency() or 0.0 for t in all_trackers]),
        }
    }
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Per-task metrics saved to {save_path}")
