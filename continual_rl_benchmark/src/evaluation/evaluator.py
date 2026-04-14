"""
Evaluator for Continual RL Agents.

Handles evaluation loops, logging, and result saving.
"""

import os
import csv
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Evaluator:
    """
    Evaluates agents and logs results.

    Supports TensorBoard, CSV, and JSON logging.
    """

    def __init__(
        self,
        log_dir: str = "results",
        experiment_name: str = "benchmark",
        seed: int = 0,
    ):
        self.log_dir = os.path.join(log_dir, experiment_name, f"seed_{seed}")
        self.seed = seed
        self.global_step = 0
        self.episode_count = 0

        # Create directories
        os.makedirs(os.path.join(self.log_dir, "tensorboard"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "csv"), exist_ok=True)

        # TensorBoard writer
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(os.path.join(self.log_dir, "tensorboard"))
        else:
            self.writer = None

        # CSV logger
        self.csv_path = os.path.join(self.log_dir, "csv", "metrics.csv")
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "episode", "return", "length",
                "task_idx", "task_name", "task_switched"
            ])

    def log_episode(
        self,
        episode_return: float,
        episode_length: int,
        task_idx: int,
        task_name: str,
        task_switched: bool,
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log episode metrics."""
        self.episode_count += 1

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar("episode/return", episode_return, self.global_step)
            self.writer.add_scalar("episode/length", episode_length, self.global_step)
            self.writer.add_scalar("episode/task_idx", task_idx, self.global_step)

            if additional_metrics:
                for key, value in additional_metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self.writer.add_scalar(f"agent/{key}", float(value), self.global_step)

        # Log to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.global_step,
                self.episode_count,
                episode_return,
                episode_length,
                task_idx,
                task_name,
                int(task_switched),
            ])

    def log_training_metrics(self, metrics: Dict[str, float]):
        """Log training metrics (losses, etc.)."""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"training/{key}", value, self.global_step)

    def log_evaluation(self, eval_return: float, eval_length: int):
        """Log evaluation metrics."""
        if self.writer:
            self.writer.add_scalar("evaluation/return", eval_return, self.global_step)
            self.writer.add_scalar("evaluation/length", eval_length, self.global_step)

    def log_continual_metrics(self, metrics: Dict[str, float]):
        """Log continual learning specific metrics."""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"continual/{key}", value, self.global_step)

    def step(self, increment: int = 1):
        """Increment global step."""
        self.global_step += increment
        if self.writer:
            self.writer.flush()

    def save_checkpoint(self, agent: Any, path: str):
        """Save agent checkpoint."""
        if hasattr(agent, "save"):
            agent.save(path)

    def save_task_checkpoint(self, agent: Any, agent_name: str, task_idx: int, base_dir: str = "checkpoints"):
        """
        Save checkpoint at task boundary.

        Saves to: checkpoints/agent_name/agent_name_task{idx}.pt
        """
        checkpoint_dir = os.path.join(base_dir, agent_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{agent_name}_task{task_idx}.pt")
        self.save_checkpoint(agent, checkpoint_path)
        return checkpoint_path

    def close(self):
        """Close evaluator and finalize logs."""
        if self.writer:
            self.writer.close()

    def get_log_dir(self) -> str:
        """Get log directory path."""
        return self.log_dir
