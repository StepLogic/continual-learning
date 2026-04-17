import jax
import jax.numpy as jnp
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.pos = 0
        self.size = 0

        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def add(self, obs, action, reward, next_obs, done):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, batch_size: int):
        indices = jax.random.choice(rng, self.size, shape=(batch_size,), replace=False)
        indices = np.asarray(indices)
        return {
            "obs": jnp.array(self.observations[indices], dtype=jnp.float32),
            "actions": jnp.array(self.actions[indices]),
            "rewards": jnp.array(self.rewards[indices]),
            "next_obs": jnp.array(self.next_observations[indices], dtype=jnp.float32),
            "dones": jnp.array(self.dones[indices]),
        }