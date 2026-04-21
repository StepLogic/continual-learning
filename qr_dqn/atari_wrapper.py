import ale_py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Register ALE environments so gym.make can find Atari game IDs.
gym.register_envs(ale_py)


class _TransposeObservation(gym.ObservationWrapper):
    """Transpose observation dimensions."""

    def __init__(self, env, axes):
        super().__init__(env)
        self.axes = axes
        old_space = self.env.observation_space
        new_shape = tuple(old_space.shape[i] for i in axes)
        self.observation_space = spaces.Box(
            low=np.transpose(old_space.low, axes),
            high=np.transpose(old_space.high, axes),
            dtype=old_space.dtype,
        )

    def observation(self, obs):
        return np.transpose(obs, self.axes)


def make_atari_env(game_id: str, seed: int = 42, full_action_space: bool = True):
    """Create an Atari environment with standard preprocessing.

    Applies: grayscale, resize to 84x84, frame skip (4), reward clipping, frame stacking (4).
    Output shape: (84, 84, 4), dtype uint8.
    """
    env = gym.make(game_id, render_mode=None, full_action_space=full_action_space)
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.MaxAndSkipObservation(env, skip=4)
    env = gym.wrappers.ClipReward(env, min_reward=-1, max_reward=1)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    # FrameStackObservation stacks along axis 0, giving (4, 84, 84).
    # Transpose to (84, 84, 4) to match the expected convention.
    env = _TransposeObservation(env, (1, 2, 0))
    obs, _ = env.reset(seed=seed)
    return env