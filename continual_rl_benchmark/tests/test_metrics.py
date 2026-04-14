"""
Unit tests for evaluation metrics.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import ContinualMetrics


class TestContinualMetrics:
    @pytest.fixture
    def metrics(self):
        return ContinualMetrics(num_tasks=5)

    def test_initialization(self, metrics):
        assert metrics.num_tasks == 5
        assert len(metrics.task_returns) == 0

    def test_log_episode(self, metrics):
        metrics.log_episode(episode_return=100.0, task_idx=0, global_step=1000)
        assert len(metrics.all_returns) == 1
        assert metrics.all_returns[0] == 100.0

    def test_end_task(self, metrics):
        # Log several episodes
        for i in range(10):
            metrics.log_episode(episode_return=float(i * 10), task_idx=0, global_step=i * 100)

        metrics.end_task(0)
        assert len(metrics.task_returns[0]) == 1

    def test_compute_average_return(self, metrics):
        for i in range(10):
            metrics.log_episode(episode_return=100.0, task_idx=0, global_step=i * 100)

        avg = metrics.compute_average_return()
        assert avg == 100.0

    def test_compute_forward_transfer(self, metrics):
        # Simulate positive transfer
        metrics.task_returns[0] = [50.0]
        metrics.task_returns[1] = [75.0]
        metrics.task_returns[2] = [80.0]

        transfer = metrics.compute_forward_transfer()
        assert transfer > 0  # Positive transfer

    def test_compute_backward_transfer(self, metrics):
        # Simulate forgetting (negative backward transfer)
        metrics.task_returns[0] = [100.0, 80.0]  # Performance dropped

        transfer = metrics.compute_backward_transfer()
        assert transfer < 0  # Negative (forgetting)

    def test_compute_forgetting_measure(self, metrics):
        # Simulate forgetting
        metrics.task_returns[0] = [100.0, 90.0, 80.0]

        forgetting = metrics.compute_forgetting_measure()
        assert forgetting > 0  # Forgetting occurred

    def test_get_summary(self, metrics):
        for i in range(10):
            metrics.log_episode(episode_return=50.0, task_idx=0, global_step=i * 100)

        summary = metrics.get_summary()
        assert "average_return" in summary
        assert "forward_transfer" in summary
        assert "backward_transfer" in summary
        assert "forgetting_measure" in summary

    def test_save_and_load(self, metrics, tmp_path):
        # Log some data
        for i in range(5):
            metrics.log_episode(episode_return=float(i * 20), task_idx=0, global_step=i * 100)
        metrics.end_task(0)

        # Save
        save_path = tmp_path / "metrics.json"
        metrics.save(str(save_path))

        # Load into new metrics object
        new_metrics = ContinualMetrics(num_tasks=5)
        new_metrics.load(str(save_path))

        assert new_metrics.all_returns == metrics.all_returns
        assert new_metrics.task_returns[0] == metrics.task_returns[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
