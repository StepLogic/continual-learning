"""
Continual Learning Environment Wrapper.

Uses MetaWorld environments for continual RL benchmarks.
"""

import time
import numpy as np
import gymnasium as gym
from typing import List, Dict, Optional, Tuple
from collections import deque
import metaworld

class ContinualEnv:
    """
    Continual learning environment wrapper using MetaWorld.

    Manages sequential task switching across MetaWorld tasks.
    """

    def __init__(
        self,
        env_name: str = "metaworld",
        task_list: Optional[List[str]] = None,
        steps_per_task: int = 100000,
        seed: int = 0,
    ):
        self.env_name = env_name
        self.steps_per_task = steps_per_task
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Default MetaWorld ML10 tasks (continual learning benchmark)
        if env_name == "metaworld" or env_name == "continual_world":
            self.task_list = task_list or [
                "reach-v3",
                "push-v3",
                "pick-place-v3",
                "door-open-v3",
                "drawer-close-v3",
                "button-press-topdown-v3",
                "peg-insert-side-v3",
                "window-open-v3",
                "sweep-v3",
                "basketball-v3",
            ]
        elif env_name == "procgen":
            self.task_list = task_list or [
                "coinrun", "dodgeball", "fruitbot", "heist", "leaper",
                "maze", "miner", "ninja", "plunder", "starfish",
            ]
        else:
            raise ValueError(f"Unknown environment: {env_name}")

        self.current_task_idx = 0
        self.steps_in_task = 0
        self.total_steps = 0

        # Performance tracking
        self.task_returns = deque(maxlen=100)
        self.task_lengths = deque(maxlen=100)

        # Current environment
        self.env = self._create_env(self.task_list[0])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _create_env(self, task_name: str) -> gym.Env:
        """Create MetaWorld environment."""
        if self.env_name in ["metaworld", "continual_world"]:
            # Use MetaWorld MT1 (single task) with proper gymnasium API
            env = gym.make(f"Meta-World/MT1", env_name=task_name, render_mode="human")
            return env
        elif self.env_name == "procgen":
            return gym.make(f"procgen:procgen-{task_name}-v0", render_mode="human")
        else:
            raise ValueError(f"Unknown environment: {task_name}")

    def reset(self) -> np.ndarray:
        """Reset current environment."""
        obs, _ = self.env.reset(seed=self.seed)
        return obs

    def render(self):
        """Render the environment."""
        self.env.render()

    def step(self, action: np.ndarray, auto_reset_if_done: bool = True) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step environment and check for task switch.

        Args:
            action: Action to take
            auto_reset_if_done: If True, automatically reset when episode ends
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if truncated:
            self.env.reset()
        done = terminated or truncated

        self.steps_in_task += 1
        self.total_steps += 1

        # Check for task switch
        task_switched = False
        if self.steps_in_task >= self.steps_per_task:
            task_switched = True
            self._switch_task()

        info["task_idx"] = self.current_task_idx
        info["task_name"] = self.get_current_task_name()
        info["task_switched"] = task_switched
        info["steps_in_task"] = self.steps_in_task
        info["episode_done"] = done

        return obs, reward, terminated, truncated, info

    def _switch_task(self):
        """Switch to next task in sequence."""
        self.current_task_idx = (self.current_task_idx + 1) % len(self.task_list)
        self.steps_in_task = 0

        # Close old environment - explicitly close Mujoco viewer
        self._close_env_viewer()

        # Create new environment for next task
        next_task = self.task_list[self.current_task_idx]
        self.env = self._create_env(next_task)
        # Reset the new environment so it's ready for step()
        self.env.reset()

        # Reset the new environment so it's ready for step()
        self._reset_new_env()

    def _reset_new_env(self):
        """Reset newly created environment after task switch."""
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.reset(seed=self.seed + self.current_task_idx)
            except Exception:
                pass

    def _close_env_viewer(self):
        """Thoroughly close environment and its viewer."""
        # Close the gym environment
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass

        # Force Mujoco viewer cleanup - check multiple possible attribute names
        for viewer_attr in ["viewer", "_viewer", "mujoco_renderer", "_renderer"]:
            if hasattr(self.env, viewer_attr):
                viewer = getattr(self.env, viewer_attr)
                if viewer is not None:
                    try:
                        if hasattr(viewer, "close"):
                            viewer.close()
                        if hasattr(viewer, "close_vulkan"):
                            viewer.close_vulkan()
                    except Exception:
                        pass
                    setattr(self.env, viewer_attr, None)

        # Check for mujoco-specific renderer in env.model
        if hasattr(self.env, "model") and self.env.model is not None:
            if hasattr(self.env.model, "viewer") and self.env.model.viewer is not None:
                try:
                    self.env.model.viewer.close()
                except Exception:
                    pass
                self.env.model.viewer = None

        # Small delay to ensure window is fully closed
        time.sleep(0.1)

    def get_current_task_name(self) -> str:
        """Get name of current task."""
        return self.task_list[self.current_task_idx]

    def get_num_tasks(self) -> int:
        """Get total number of tasks."""
        return len(self.task_list)

    def get_task_list(self) -> List[str]:
        """Get list of all tasks."""
        return self.task_list.copy()

    def is_task_switch(self) -> bool:
        """Check if current step is a task switch."""
        return self.steps_in_task == 0

    def close(self):
        """Close environment and cleanup Mujoco viewer."""
        self._close_env_viewer()
