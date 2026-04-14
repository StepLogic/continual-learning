"""
Unit tests for environment wrapper.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environments import ContinualEnv


class TestContinualEnv:
    @pytest.fixture
    def env(self):
        return ContinualEnv(env_name="continual_world", steps_per_task=100, seed=0)

    def test_initialization(self, env):
        assert env is not None
        assert env.current_task_idx == 0
        assert env.steps_in_task == 0

    def test_reset(self, env):
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert len(obs.shape) == 1

    def test_step(self, env):
        obs = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "task_idx" in info
        assert "task_switched" in info

    def test_task_switching(self, env):
        # Run until task switch
        obs = env.reset()
        initial_task = env.current_task_idx

        for _ in range(100):
            action = env.action_space.sample()
            obs, _, _, _, info = env.step(action)

        # Task should have switched
        assert env.current_task_idx != initial_task or env.get_num_tasks() == 1

    def test_get_task_info(self, env):
        tasks = env.get_task_list()
        assert len(tasks) > 0
        assert isinstance(tasks[0], str)

        num_tasks = env.get_num_tasks()
        assert num_tasks == len(tasks)

    def test_dummy_env_fallback(self):
        # Test with unknown env name (should use dummy)
        env = ContinualEnv(env_name="unknown_env", seed=0)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)

        action = env.action_space.sample()
        next_obs, reward, _, _, _ = env.step(action)
        assert isinstance(next_obs, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
