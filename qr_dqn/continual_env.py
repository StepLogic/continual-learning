import numpy as np
from typing import List, Tuple, Dict
from qr_dqn.atari_wrapper import make_atari_env


class ContinualAtariEnv:
    """
    Continual learning environment wrapper for sequential Atari games.

    Manages task switching across a list of Atari games.
    Each game is a separate task. The environment automatically
    switches to the next game after `steps_per_task` steps.
    """

    def __init__(
        self,
        games: List[str],
        steps_per_task: int = 500_000,
        seed: int = 0,
    ):
        self.games = games
        self.steps_per_task = steps_per_task
        self.seed = seed
        self.current_task_idx = 0
        self.steps_in_task = 0
        self.total_steps = 0

        # Create first environment
        self.env = make_atari_env(self.games[0], seed=seed)

    def reset(self, seed=None):
        """Reset current environment."""
        reset_seed = seed if seed is not None else self.seed + self.current_task_idx
        obs, info = self.env.reset(seed=reset_seed)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step environment and check for task switch.

        Returns observation, reward, terminated, truncated, info.
        Info dict includes task_idx, task_name, and task_switched.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps_in_task += 1
        self.total_steps += 1

        # Check for task switch
        task_switched = False
        if self.steps_in_task >= self.steps_per_task:
            task_switched = True
            self._switch_task()

        info["task_idx"] = self.current_task_idx
        info["task_name"] = self.get_current_game()
        info["task_switched"] = task_switched
        info["steps_in_task"] = self.steps_in_task

        return obs, reward, terminated, truncated, info

    def _switch_task(self):
        """Switch to next task in sequence."""
        self.current_task_idx = (self.current_task_idx + 1) % len(self.games)
        self.steps_in_task = 0

        # Close old environment to prevent state leakage
        self.env.close()

        # Create new environment for next game
        next_game = self.games[self.current_task_idx]
        self.env = make_atari_env(next_game, seed=self.seed)
        self.env.reset(seed=self.seed + self.current_task_idx)

    def get_current_game(self) -> str:
        """Get name of current game/task."""
        return self.games[self.current_task_idx]

    def get_num_tasks(self) -> int:
        """Get total number of tasks."""
        return len(self.games)

    def get_task_list(self) -> List[str]:
        """Get list of all tasks."""
        return self.games.copy()

    def is_task_switch(self) -> bool:
        """Check if current step is a task switch."""
        return self.steps_in_task == 0

    def close(self):
        """Close environment."""
        self.env.close()
