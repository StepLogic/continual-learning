import jax
import jax.numpy as jnp
import numpy as np
from collections import deque


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
        actual_batch = min(batch_size, self.size)
        indices = jax.random.choice(rng, self.size, shape=(actual_batch,), replace=False)
        indices = np.asarray(indices)
        return {
            "obs": jnp.array(self.observations[indices], dtype=jnp.float32),
            "actions": jnp.array(self.actions[indices]),
            "rewards": jnp.array(self.rewards[indices]),
            "next_obs": jnp.array(self.next_observations[indices], dtype=jnp.float32),
            "dones": jnp.array(self.dones[indices]),
        }


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple,
                 alpha: float = 0.6, epsilon: float = 1e-6):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.alpha = alpha
        self.epsilon = epsilon
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, batch_size: int, beta: float = 1.0):
        batch_size = min(batch_size, self.size)
        probs = self.priorities[:self.size] ** self.alpha
        probs_sum = probs.sum()
        probs = probs / probs_sum

        indices = jax.random.choice(
            rng, self.size, shape=(batch_size,), p=probs, replace=True,
        )
        indices = np.asarray(indices)

        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return {
            "obs": jnp.array(self.observations[indices], dtype=jnp.float32),
            "actions": jnp.array(self.actions[indices]),
            "rewards": jnp.array(self.rewards[indices]),
            "next_obs": jnp.array(self.next_observations[indices], dtype=jnp.float32),
            "dones": jnp.array(self.dones[indices]),
            "weights": jnp.array(weights, dtype=jnp.float32),
            "indices": indices,
        }

    def update_priorities(self, indices, priorities):
        priorities = np.asarray(priorities, dtype=np.float32) + self.epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))


class NStepBuffer:
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self._buffer = deque()

    def push(self, obs, action, reward, next_obs, terminated):
        self._buffer.append({
            "obs": obs, "action": action, "reward": reward,
            "next_obs": next_obs, "terminated": terminated,
        })
        if len(self._buffer) >= self.n:
            return self._pop_n_step()
        return None

    def _pop_n_step(self):
        first = self._buffer[0]
        obs = first["obs"]
        action = first["action"]
        n_step_reward = 0.0
        discount = 1.0
        final_next_obs = None
        final_terminated = False

        for i in range(len(self._buffer)):
            trans = self._buffer[i]
            n_step_reward += discount * trans["reward"]
            final_next_obs = trans["next_obs"]
            final_terminated = trans["terminated"]
            if trans["terminated"]:
                break
            discount *= self.gamma

        self._buffer.popleft()
        return obs, action, n_step_reward, final_next_obs, final_terminated

    def flush(self):
        results = []
        while self._buffer:
            results.append(self._pop_n_step())
        return results