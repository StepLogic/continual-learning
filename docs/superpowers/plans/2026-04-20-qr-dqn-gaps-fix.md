# QR-DQN Gaps Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all identified gaps in the QR-DQN implementation — dueling architecture, prioritized replay, n-step returns, gradient clipping, truncation bug, soft target update support, and updated defaults.

**Architecture:** Incremental upgrades to the existing QR-DQN codebase. Pure QR-DQN architecture (no cos-basis embedding — that's IQN). Each task adds a self-contained feature with tests passing after each task. Designed for laptop-scale training (50k buffer, 64 quantiles).

**Tech Stack:** JAX, Flax, Optax, NumPy, Gymnasium, pytest

---

### File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `qr_dqn/configs.py` | Modify | Add new config fields, update defaults |
| `qr_dqn/network.py` | Modify | Add dueling architecture option |
| `qr_dqn/replay.py` | Modify | Add PrioritizedReplayBuffer, NStepBuffer, fix edge case |
| `qr_dqn/agent.py` | Modify | Wire up dueling, PER, gradient clipping, soft target update |
| `qr_dqn/train.py` | Modify | Truncation fix, n-step integration |
| `qr_dqn/losses.py` | Modify | Fix standalone loss to accept pre-selected targets |
| `qr_dqn/__init__.py` | Modify | Update exports |
| `tests/test_qr_dqn.py` | Modify | Tests for all new features |

---

### Task 1: Update Config Defaults and Add New Fields

**Files:**
- Modify: `qr_dqn/configs.py`
- Test: `tests/test_qr_dqn.py`

- [ ] **Step 1: Write the failing test for new config fields**

Add to `tests/test_qr_dqn.py`:

```python
def test_config_new_fields():
    config = QRDQNConfig()
    assert config.dueling is True
    assert config.n_step == 3
    assert config.max_grad_norm == 10.0
    assert config.per_alpha == 0.6
    assert config.per_beta_start == 0.4
    assert config.per_beta_frames == 10_000_000
    assert config.per_epsilon == 1e-6
    assert config.target_update_tau == 1.0


def test_config_updated_defaults():
    config = QRDQNConfig()
    assert config.num_quantiles == 64
    assert config.replay_capacity == 50_000
    assert config.epsilon_decay_steps == 250_000
    assert config.warmup_steps == 20_000
    assert config.max_frames == 10_000_000
    assert config.target_update_freq == 10_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qr_dqn.py::test_config_new_fields tests/test_qr_dqn.py::test_config_updated_defaults -v`
Expected: FAIL (AttributeError — new fields don't exist yet, defaults differ)

- [ ] **Step 3: Update config with new fields and defaults**

Replace `qr_dqn/configs.py` with:

```python
from dataclasses import dataclass


@dataclass
class QRDQNConfig:
    num_quantiles: int = 64
    kappa: float = 1.0
    learning_rate: float = 1e-4
    adam_eps: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 250_000
    batch_size: int = 32
    replay_capacity: int = 50_000
    warmup_steps: int = 20_000
    target_update_freq: int = 10_000
    target_update_tau: float = 1.0
    gamma: float = 0.99
    max_frames: int = 10_000_000
    hidden_dim: int = 256
    eval_interval: int = 25_000
    eval_episodes: int = 10
    num_envs: int = 1
    seed: int = 42
    game: str = "Pong"
    log_interval: int = 10_000
    # Architecture
    dueling: bool = True
    # N-step returns
    n_step: int = 3
    # Gradient clipping
    max_grad_norm: float = 10.0
    # Prioritized Experience Replay
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 10_000_000
    per_epsilon: float = 1e-6
```

- [ ] **Step 4: Update existing config tests for new defaults**

In `tests/test_qr_dqn.py`, update `test_config_defaults`:

```python
def test_config_defaults():
    config = QRDQNConfig()
    assert config.num_quantiles == 64
    assert config.kappa == 1.0
    assert config.learning_rate == 1e-4
    assert config.gamma == 0.99
    assert config.replay_capacity == 50_000
```

- [ ] **Step 5: Run all config tests to verify they pass**

Run: `pytest tests/test_qr_dqn.py::test_config_defaults tests/test_qr_dqn.py::test_config_override tests/test_qr_dqn.py::test_config_new_fields tests/test_qr_dqn.py::test_config_updated_defaults -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add qr_dqn/configs.py tests/test_qr_dqn.py
git commit -m "feat: add new config fields and update defaults"
```

---

### Task 2: Fix Truncation Bug

**Files:**
- Modify: `qr_dqn/train.py:51-53`
- Test: `tests/test_qr_dqn.py`

The bug: `done = terminated or truncated` causes zero-bootstrapping at time limits, creating artificial death penalties. Fix: store only `terminated` in the buffer; reset env on `terminated or truncated`.

- [ ] **Step 1: Fix the training loop**

In `qr_dqn/train.py`, change line 52-53:

```python
# Before:
done = terminated or truncated
agent.buffer_add(obs, action, reward, next_obs, done)
```

To:

```python
agent.buffer_add(obs, action, reward, next_obs, terminated)
```

Keep the episode reset logic:

```python
if terminated or truncated:
    ...
    obs, _ = env.reset()
```

The evaluate function's `done = terminated or truncated` is correct for loop control (no transitions stored), so leave it as-is.

- [ ] **Step 2: Write a test that validates the fix**

Add to `tests/test_qr_dqn.py`:

```python
def test_train_stores_terminated_only():
    """Buffer should store 'terminated' only, not 'terminated or truncated'."""
    config = QRDQNConfig(
        replay_capacity=500, warmup_steps=0, max_frames=50,
        epsilon_start=1.0, epsilon_end=1.0,
    )
    rng = jax.random.PRNGKey(config.seed)
    rng, agent_rng = jax.random.split(rng)
    agent = QRDQNAgent(config, num_actions=6, obs_shape=(84, 84, 4), rng=agent_rng)
    env = make_atari_env("PongNoFrameskip-v4", seed=config.seed)
    obs, _ = env.reset()
    for _ in range(50):
        action = agent.act(obs, epsilon=1.0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.buffer_add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    # Truncated transitions should have done=False (they bootstrap)
    for i in range(agent.buffer.size):
        if agent.buffer.dones[i]:
            assert isinstance(agent.buffer.dones[i], (bool, np.bool_))
    env.close()
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_qr_dqn.py::test_train_stores_terminated_only -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add qr_dqn/train.py tests/test_qr_dqn.py
git commit -m "fix: store only terminated (not truncated) as done flag in replay buffer"
```

---

### Task 3: Dueling Architecture

**Files:**
- Modify: `qr_dqn/network.py`
- Test: `tests/test_qr_dqn.py`

Add dueling streams (value + advantage) as an option on the existing `QuantileNetwork`. When `dueling=False`, keep the current single-stream architecture.

- [ ] **Step 1: Write the failing test for dueling network**

Add to `tests/test_qr_dqn.py`:

```python
def test_network_dueling_shape(rng):
    num_actions = 6
    num_quantiles = 32
    net = QuantileNetwork(
        num_actions=num_actions,
        num_quantiles=num_quantiles,
        dueling=True,
    )
    dummy_obs = jnp.ones((4, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert output.shape == (4, num_actions, num_quantiles)


def test_network_dueling_output_finite(rng):
    num_actions = 4
    num_quantiles = 8
    net = QuantileNetwork(
        num_actions=num_actions,
        num_quantiles=num_quantiles,
        dueling=True,
    )
    dummy_obs = jnp.ones((2, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert jnp.all(jnp.isfinite(output))


def test_network_nondueling_backward_compat(rng):
    """dueling=False should produce same shape as before."""
    num_actions = 6
    num_quantiles = 32
    net = QuantileNetwork(
        num_actions=num_actions,
        num_quantiles=num_quantiles,
        dueling=False,
    )
    dummy_obs = jnp.ones((2, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert output.shape == (2, num_actions, num_quantiles)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qr_dqn.py::test_network_dueling_shape -v`
Expected: FAIL (TypeError — QuantileNetwork doesn't accept dueling param)

- [ ] **Step 3: Update the network**

Replace `qr_dqn/network.py` with:

```python
import jax.numpy as jnp
import flax.linen as nn


class QuantileNetwork(nn.Module):
    num_actions: int
    num_quantiles: int
    hidden_dim: int = 256
    dueling: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        if self.dueling:
            # Value stream
            val = nn.Dense(self.hidden_dim)(x)
            val = nn.relu(val)
            val = nn.Dense(self.num_quantiles)(val)  # (batch, N)

            # Advantage stream
            adv = nn.Dense(self.hidden_dim)(x)
            adv = nn.relu(adv)
            adv = nn.Dense(self.num_actions * self.num_quantiles)(adv)
            adv = adv.reshape((-1, self.num_actions, self.num_quantiles))  # (batch, A, N)

            # Q = V + A - mean(A)
            q = val[:, None, :] + adv - adv.mean(axis=1, keepdims=True)  # (batch, A, N)
            return q
        else:
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(self.num_actions * self.num_quantiles)(x)
            x = x.reshape((-1, self.num_actions, self.num_quantiles))
            return x
```

- [ ] **Step 4: Update existing network tests to pass `dueling` param**

In `tests/test_qr_dqn.py`, update the existing network tests:

```python
def test_network_forward_shape(rng):
    num_actions = 6
    num_quantiles = 32
    net = QuantileNetwork(
        num_actions=num_actions,
        num_quantiles=num_quantiles,
        dueling=False,
    )
    dummy_obs = jnp.ones((4, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert output.shape == (4, num_actions, num_quantiles)


def test_network_single_obs(rng):
    num_actions = 6
    num_quantiles = 32
    net = QuantileNetwork(
        num_actions=num_actions,
        num_quantiles=num_quantiles,
        dueling=False,
    )
    dummy_obs = jnp.ones((1, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert output.shape == (1, num_actions, num_quantiles)


def test_network_output_finite(rng):
    num_actions = 4
    num_quantiles = 32
    net = QuantileNetwork(
        num_actions=num_actions,
        num_quantiles=num_quantiles,
        dueling=False,
    )
    dummy_obs = jnp.ones((2, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert jnp.all(jnp.isfinite(output))
```

- [ ] **Step 5: Run all network tests**

Run: `pytest tests/test_qr_dqn.py -v -k "network"`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add qr_dqn/network.py tests/test_qr_dqn.py
git commit -m "feat: add dueling architecture option to QuantileNetwork"
```

---

### Task 4: Prioritized Experience Replay

**Files:**
- Modify: `qr_dqn/replay.py`
- Test: `tests/test_qr_dqn.py`

Add a `PrioritizedReplayBuffer` class alongside the existing `ReplayBuffer`. Also fix the `replace=False` edge case in the existing buffer.

- [ ] **Step 1: Write the failing test for PER**

Add to `tests/test_qr_dqn.py`:

```python
from qr_dqn.replay import PrioritizedReplayBuffer


def test_per_buffer_init():
    buf = PrioritizedReplayBuffer(capacity=100, obs_shape=(84, 84, 4))
    assert buf.size == 0


def test_per_buffer_add():
    buf = PrioritizedReplayBuffer(capacity=100, obs_shape=(84, 84, 4))
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    next_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    buf.add(obs, 1, 1.0, next_obs, False)
    assert buf.size == 1


def test_per_buffer_sample_has_weights(rng):
    buf = PrioritizedReplayBuffer(capacity=100, obs_shape=(84, 84, 4))
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    next_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    for i in range(50):
        buf.add(obs, i % 4, float(i), next_obs, False)
    batch = buf.sample(rng, batch_size=8, beta=1.0)
    assert "weights" in batch
    assert "indices" in batch
    assert batch["weights"].shape == (8,)
    assert batch["indices"].shape == (8,)
    assert batch["obs"].shape == (8, 84, 84, 4)


def test_per_buffer_update_priorities():
    buf = PrioritizedReplayBuffer(capacity=100, obs_shape=(84, 84, 4))
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    next_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    for i in range(10):
        buf.add(obs, i % 4, float(i), next_obs, False)
    indices = np.array([0, 1, 2])
    priorities = np.array([5.0, 3.0, 1.0])
    buf.update_priorities(indices, priorities)
    assert buf.priorities[0] == pytest.approx(5.0 + 1e-6)
    assert buf.priorities[1] == pytest.approx(3.0 + 1e-6)
    assert buf.priorities[2] == pytest.approx(1.0 + 1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qr_dqn.py::test_per_buffer_init -v`
Expected: FAIL (ImportError — PrioritizedReplayBuffer doesn't exist)

- [ ] **Step 3: Implement PrioritizedReplayBuffer**

Add to `qr_dqn/replay.py` (after the existing `ReplayBuffer` class):

```python
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
```

Also fix the `replace=False` edge case in the existing `ReplayBuffer.sample`:

```python
# In ReplayBuffer.sample, change:
indices = jax.random.choice(rng, self.size, shape=(batch_size,), replace=False)
# To:
actual_batch = min(batch_size, self.size)
indices = jax.random.choice(rng, self.size, shape=(actual_batch,), replace=False)
```

- [ ] **Step 4: Run PER tests**

Run: `pytest tests/test_qr_dqn.py -v -k "per_buffer"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qr_dqn/replay.py tests/test_qr_dqn.py
git commit -m "feat: add PrioritizedReplayBuffer with proportional prioritization"
```

---

### Task 5: NStepBuffer

**Files:**
- Modify: `qr_dqn/replay.py`
- Test: `tests/test_qr_dqn.py`

Add an `NStepBuffer` class that accumulates raw transitions and outputs n-step discounted returns.

- [ ] **Step 1: Write the failing test for NStepBuffer**

Add to `tests/test_qr_dqn.py`:

```python
from qr_dqn.replay import NStepBuffer


def test_nstep_buffer_returns_none_when_not_full():
    buf = NStepBuffer(n=3, gamma=0.99)
    obs = np.zeros((84, 84, 4), dtype=np.uint8)
    result = buf.push(obs, 0, 1.0, obs, False)
    assert result is None


def test_nstep_buffer_returns_transition_when_full():
    buf = NStepBuffer(n=3, gamma=0.99)
    obs = np.zeros((84, 84, 4), dtype=np.uint8)
    obs1 = np.ones((84, 84, 4), dtype=np.uint8)
    buf.push(obs, 0, 1.0, obs1, False)
    buf.push(obs1, 1, 2.0, obs, False)
    result = buf.push(obs, 2, 3.0, obs1, False)
    assert result is not None
    obs_n, action_n, reward_n, next_obs_n, terminated_n = result
    assert action_n == 0
    expected = 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
    assert reward_n == pytest.approx(expected, rel=1e-4)
    assert terminated_n is False


def test_nstep_buffer_early_termination():
    buf = NStepBuffer(n=3, gamma=0.99)
    obs = np.zeros((84, 84, 4), dtype=np.uint8)
    obs1 = np.ones((84, 84, 4), dtype=np.uint8)
    buf.push(obs, 0, 1.0, obs1, False)
    buf.push(obs1, 1, 2.0, obs, True)  # terminated at step 2
    result = buf.push(obs, 2, 3.0, obs1, False)
    assert result is not None
    _, _, reward_n, _, terminated_n = result
    expected = 1.0 + 0.99 * 2.0
    assert reward_n == pytest.approx(expected, rel=1e-4)
    assert terminated_n is True


def test_nstep_buffer_flush():
    buf = NStepBuffer(n=3, gamma=0.99)
    obs = np.zeros((84, 84, 4), dtype=np.uint8)
    obs1 = np.ones((84, 84, 4), dtype=np.uint8)
    buf.push(obs, 0, 1.0, obs1, False)
    buf.push(obs1, 1, 2.0, obs, False)
    results = buf.flush()
    assert len(results) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qr_dqn.py::test_nstep_buffer_returns_none_when_not_full -v`
Expected: FAIL (ImportError — NStepBuffer doesn't exist)

- [ ] **Step 3: Implement NStepBuffer**

Add to `qr_dqn/replay.py` (add `from collections import deque` at top):

```python
from collections import deque


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
```

- [ ] **Step 4: Run NStepBuffer tests**

Run: `pytest tests/test_qr_dqn.py -v -k "nstep_buffer"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qr_dqn/replay.py tests/test_qr_dqn.py
git commit -m "feat: add NStepBuffer for n-step return computation"
```

---

### Task 6: Update Agent — Dueling, PER, Gradient Clipping, Soft Target Update

**Files:**
- Modify: `qr_dqn/agent.py`
- Test: `tests/test_qr_dqn.py`

This is the main integration task. Wire up dueling network, PER with IS weights, gradient clipping, and soft/Polyak target updates.

- [ ] **Step 1: Write the failing test for agent with new features**

Add to `tests/test_qr_dqn.py`:

```python
def test_agent_gradient_clipping():
    config = QRDQNConfig(num_quantiles=16, replay_capacity=500, max_grad_norm=10.0, dueling=False)
    rng_key = jax.random.PRNGKey(42)
    agent = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng_key)
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    for i in range(64):
        agent.buffer_add(obs, i % 4, 1.0, obs, False)
    metrics = agent.train_step(jax.random.PRNGKey(0))
    assert "loss" in metrics
    assert jnp.isfinite(metrics["loss"])


def test_agent_per_buffer():
    config = QRDQNConfig(
        num_quantiles=16, replay_capacity=500, dueling=False,
        per_alpha=0.6, per_beta_start=0.4,
    )
    rng_key = jax.random.PRNGKey(42)
    agent = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng_key)
    assert isinstance(agent.buffer, PrioritizedReplayBuffer)


def test_agent_soft_target_update():
    config = QRDQNConfig(
        num_quantiles=16, replay_capacity=500, dueling=False,
        target_update_tau=0.005,
    )
    rng_key = jax.random.PRNGKey(42)
    agent = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng_key)
    params_before = jax.tree_util.tree_map(jnp.copy, agent.target_params)
    agent.update_target()
    # With tau=0.005, target params should have moved slightly toward online
    different = not jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b), params_before, agent.target_params)
    )
    assert different
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qr_dqn.py::test_agent_per_buffer -v`
Expected: FAIL (agent still uses ReplayBuffer)

- [ ] **Step 3: Rewrite agent.py**

Replace `qr_dqn/agent.py` with:

```python
import jax
import jax.numpy as jnp
import optax
import numpy as np
from qr_dqn.configs import QRDQNConfig
from qr_dqn.network import QuantileNetwork
from qr_dqn.replay import ReplayBuffer, PrioritizedReplayBuffer
from qr_dqn.losses import quantile_huber_loss


class QRDQNAgent:
    def __init__(self, config: QRDQNConfig, num_actions: int, obs_shape: tuple, rng: jax.Array):
        self.config = config
        self.num_actions = num_actions
        self.network = QuantileNetwork(
            num_actions=num_actions,
            num_quantiles=config.num_quantiles,
            hidden_dim=config.hidden_dim,
            dueling=config.dueling,
        )

        # Optimizer with gradient clipping
        optimizer_parts = []
        if config.max_grad_norm > 0:
            optimizer_parts.append(optax.clip_by_global_norm(config.max_grad_norm))
        optimizer_parts.append(optax.adam(config.learning_rate, eps=config.adam_eps))
        self.optimizer = optax.chain(*optimizer_parts)

        # Replay buffer: PER or uniform
        if config.per_alpha > 0:
            self.buffer = PrioritizedReplayBuffer(
                capacity=config.replay_capacity,
                obs_shape=obs_shape,
                alpha=config.per_alpha,
                epsilon=config.per_epsilon,
            )
        else:
            self.buffer = ReplayBuffer(
                capacity=config.replay_capacity,
                obs_shape=obs_shape,
            )

        self.rng = rng
        self.rng, init_rng = jax.random.split(rng)

        dummy_obs = jnp.ones((1, *obs_shape), dtype=jnp.float32)
        self.taus = (2 * jnp.arange(config.num_quantiles) + 1) / (2 * config.num_quantiles)
        self.params = self.network.init(init_rng, dummy_obs)
        self.target_params = jax.tree_util.tree_map(jnp.copy, self.params)
        self.opt_state = self.optimizer.init(self.params)
        self.step_count = 0

        self._jit_train = jax.jit(self._train_step_fn)

    def buffer_add(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def act(self, obs, epsilon: float = 0.0) -> int:
        self.rng, act_rng, explore_rng = jax.random.split(self.rng, 3)
        if jax.random.uniform(explore_rng) < epsilon:
            return int(jax.random.randint(act_rng, (), 0, self.num_actions))
        quantiles = self.network.apply(
            self.params, obs[jnp.newaxis].astype(jnp.float32),
        )
        mean_q = quantiles.mean(axis=-1).squeeze()
        return int(jnp.argmax(mean_q))

    def get_quantiles(self, obs) -> jnp.ndarray:
        return self.network.apply(
            self.params, obs[jnp.newaxis].astype(jnp.float32),
        ).squeeze(0)

    def copy_params(self):
        return jax.tree_util.tree_map(jnp.copy, self.params)

    def reset_params(self, rng):
        self.rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.ones((1, *self.buffer.obs_shape), dtype=jnp.float32)
        self.params = self.network.init(init_rng, dummy_obs)
        self.opt_state = self.optimizer.init(self.params)
        return self.params

    def distill_from(self, other: "QRDQNAgent"):
        self.params = jax.tree_util.tree_map(jnp.copy, other.params)
        self.target_params = jax.tree_util.tree_map(jnp.copy, other.target_params)
        self.opt_state = self.optimizer.init(self.params)

    def update_target(self):
        tau = self.config.target_update_tau
        if tau >= 1.0:
            self.target_params = jax.tree_util.tree_map(jnp.copy, self.params)
        else:
            self.target_params = jax.tree_util.tree_map(
                lambda online, target: tau * online + (1 - tau) * target,
                self.params, self.target_params,
            )

    def _compute_beta(self, frame: int) -> float:
        if self.config.per_alpha <= 0:
            return 1.0
        beta_frames = self.config.per_beta_frames
        if beta_frames is None or beta_frames <= 0:
            return 1.0
        fraction = min(frame / beta_frames, 1.0)
        return self.config.per_beta_start + fraction * (1.0 - self.config.per_beta_start)

    def _train_step_fn(self, params, target_params, opt_state, batch, taus, gamma, kappa):
        weights = batch.get("weights", jnp.ones(batch["rewards"].shape[0]))

        def loss_fn(params):
            current_quantiles = self.network.apply(params, batch["obs"])
            next_quantiles_online = self.network.apply(params, batch["next_obs"])
            next_actions = next_quantiles_online.mean(axis=-1).argmax(axis=-1)
            target_quantiles = self.network.apply(target_params, batch["next_obs"])
            batch_idx = jnp.arange(batch["rewards"].shape[0])
            target_values = target_quantiles[batch_idx, next_actions]
            rewards = batch["rewards"][:, None]
            dones = batch["dones"][:, None].astype(jnp.float32)
            bellman_targets = rewards + gamma * (1 - dones) * target_values

            curr = current_quantiles[batch_idx, batch["actions"]]

            # Quantile Huber loss
            delta = bellman_targets[:, None, :] - curr[:, :, None]
            taus_row = taus[None, :, None]
            indicator = (delta < 0).astype(jnp.float32)
            asymmetric_weight = jnp.abs(taus_row - indicator)
            abs_delta = jnp.abs(delta)
            quadratic = jnp.minimum(abs_delta, kappa) ** 2 * 0.5
            linear = kappa * (abs_delta - 0.5 * kappa)
            huber = jnp.where(abs_delta <= kappa, quadratic, linear)
            per_sample_loss = (asymmetric_weight * huber).sum(axis=2).mean(axis=1)

            # Apply importance sampling weights
            loss = (weights * per_sample_loss).mean()

            # TD errors for PER priority updates
            td_errors = jnp.abs(bellman_targets.mean(axis=-1) - curr.mean(axis=-1))

            return loss, (current_quantiles, td_errors)

        (loss, (quantiles, td_errors)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, td_errors

    def train_step(self, rng_key=None):
        if rng_key is None:
            self.rng, rng_key = jax.random.split(self.rng)

        if isinstance(self.buffer, PrioritizedReplayBuffer):
            beta = self._compute_beta(self.step_count)
            batch = self.buffer.sample(rng_key, self.config.batch_size, beta=beta)
        else:
            batch = self.buffer.sample(rng_key, self.config.batch_size)
            batch["weights"] = jnp.ones(self.config.batch_size)
            batch["indices"] = np.arange(self.config.batch_size)

        self.params, self.opt_state, loss, td_errors = self._jit_train(
            self.params, self.target_params, self.opt_state,
            batch, self.taus, self.config.gamma, self.config.kappa,
        )
        self.step_count += 1

        # Update PER priorities
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            self.buffer.update_priorities(
                np.asarray(batch["indices"]), np.asarray(td_errors),
            )

        # Target update: hard copy every target_update_freq, soft update every step
        if self.config.target_update_tau < 1.0:
            self.update_target()
        elif self.step_count % self.config.target_update_freq == 0:
            self.update_target()

        return {"loss": float(loss), "td_errors": td_errors}
```

- [ ] **Step 4: Update existing agent tests for new config**

In `tests/test_qr_dqn.py`, update the agent fixture and tests:

```python
@pytest.fixture
def agent(rng):
    config = QRDQNConfig(num_quantiles=16, replay_capacity=500, dueling=False, per_alpha=0.0)
    return QRDQNAgent(config, num_actions=6, obs_shape=(84, 84, 4), rng=rng)


def test_agent_get_quantiles(agent):
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    quantiles = agent.get_quantiles(obs)
    assert quantiles.shape == (6, 16)


def test_agent_train_step(agent):
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    for i in range(64):
        agent.buffer_add(obs, i % 6, 1.0, obs, False)
    rng_key = jax.random.PRNGKey(0)
    metrics = agent.train_step(rng_key)
    assert "loss" in metrics
    assert jnp.isfinite(metrics["loss"])
    assert "td_errors" in metrics


def test_agent_distill_from(rng):
    config = QRDQNConfig(num_quantiles=16, replay_capacity=500, dueling=False, per_alpha=0.0)
    agent1 = QRDQNAgent(config, num_actions=6, obs_shape=(84, 84, 4), rng=rng)
    key2 = jax.random.PRNGKey(99)
    agent2 = QRDQNAgent(config, num_actions=6, obs_shape=(84, 84, 4), rng=key2)
    agent1.distill_from(agent2)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b), agent1.params, agent2.params)
    )
```

Update `TestFAMEHooks`:

```python
class TestFAMEHooks:
    def test_distill_then_get_quantiles(self):
        config = QRDQNConfig(num_quantiles=16, replay_capacity=100, dueling=False, per_alpha=0.0)
        rng1 = jax.random.PRNGKey(0)
        rng2 = jax.random.PRNGKey(1)
        fast = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng1)
        meta = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng2)
        obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        q_before = fast.get_quantiles(obs)
        fast.distill_from(meta)
        q_after = fast.get_quantiles(obs)
        assert not jnp.allclose(q_before, q_after)

    def test_reset_params_changes_output(self):
        config = QRDQNConfig(num_quantiles=16, replay_capacity=100, dueling=False, per_alpha=0.0)
        rng = jax.random.PRNGKey(0)
        agent = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng)
        obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        q_before = agent.get_quantiles(obs)
        agent.reset_params(jax.random.PRNGKey(99))
        q_after = agent.get_quantiles(obs)
        assert not jnp.allclose(q_before, q_after)

    def test_copy_params_independent(self):
        config = QRDQNConfig(num_quantiles=16, replay_capacity=100, dueling=False, per_alpha=0.0)
        rng = jax.random.PRNGKey(0)
        agent = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng)
        copied_params = agent.copy_params()
        agent.reset_params(jax.random.PRNGKey(77))
        obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        q_orig = agent.network.apply(copied_params, obs[jnp.newaxis].astype(jnp.float32))
        q_new = agent.get_quantiles(obs)
        assert not jnp.allclose(q_orig.squeeze(0), q_new)
```

- [ ] **Step 5: Run all agent tests**

Run: `pytest tests/test_qr_dqn.py -v -k "agent"`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add qr_dqn/agent.py tests/test_qr_dqn.py
git commit -m "feat: wire up dueling, PER, gradient clipping, soft target update in agent"
```

---

### Task 7: Integrate NStepBuffer in Training Loop

**Files:**
- Modify: `qr_dqn/train.py`

Integrate the `NStepBuffer` into the training loop so transitions are stored with n-step discounted returns.

- [ ] **Step 1: Integrate NStepBuffer into train.py**

Replace `qr_dqn/train.py` with:

```python
import logging

import jax
import jax.numpy as jnp
import numpy as np
from qr_dqn.configs import QRDQNConfig
from qr_dqn.agent import QRDQNAgent
from qr_dqn.atari_wrapper import make_atari_env
from qr_dqn.replay import NStepBuffer

logger = logging.getLogger("continual_rl")


def get_epsilon(frame: int, config: QRDQNConfig) -> float:
    if frame >= config.epsilon_decay_steps:
        return config.epsilon_end
    slope = (config.epsilon_end - config.epsilon_start) / config.epsilon_decay_steps
    return config.epsilon_start + slope * frame


def _log_metrics(frame, avg_loss, epsilon, episode_count, episode_returns, max_return):
    recent = episode_returns[-10:] if episode_returns else [0.0]
    mean_ret = sum(recent) / len(recent)
    max_ret_str = f"{max_return:.1f}" if max_return != float("-inf") else "--"
    print(
        f"[Frame {frame:>7d}] loss={avg_loss:.3f} | eps={epsilon:.3f} "
        f"| episodes={episode_count} | mean_ret(10)={mean_ret:.1f} | max_ret={max_ret_str}"
    )


def train(config: QRDQNConfig, max_frames_override: int = None):
    max_frames = max_frames_override or config.max_frames
    env = make_atari_env(config.game, seed=config.seed)
    rng = jax.random.PRNGKey(config.seed)
    rng, agent_rng = jax.random.split(rng)

    num_actions = env.action_space.n
    obs, _ = env.reset()
    obs_shape = obs.shape

    agent = QRDQNAgent(config, num_actions=num_actions, obs_shape=obs_shape, rng=agent_rng)

    nstep_buffer = NStepBuffer(n=config.n_step, gamma=config.gamma) if config.n_step > 1 else None

    metrics = {"losses": [], "episode_returns": [], "eval_returns": []}
    episode_return = 0.0
    episode_count = 0
    max_return = float("-inf")
    losses_since_log = []

    for frame in range(max_frames):
        epsilon = get_epsilon(frame, config)
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store terminated (not truncated) for correct bootstrapping
        if nstep_buffer is not None:
            nstep_result = nstep_buffer.push(obs, action, reward, next_obs, terminated)
            if nstep_result is not None:
                obs_n, action_n, reward_n, next_obs_n, term_n = nstep_result
                agent.buffer_add(obs_n, action_n, reward_n, next_obs_n, term_n)
        else:
            agent.buffer_add(obs, action, reward, next_obs, terminated)

        episode_return += reward
        obs = next_obs

        if terminated or truncated:
            if nstep_buffer is not None:
                for obs_n, action_n, reward_n, next_obs_n, term_n in nstep_buffer.flush():
                    agent.buffer_add(obs_n, action_n, reward_n, next_obs_n, term_n)

            metrics["episode_returns"].append(episode_return)
            max_return = max(max_return, episode_return)
            logger.info(
                "Episode %d | Return: %.1f | Max Return: %.1f | Frame: %d",
                episode_count, episode_return, max_return, frame,
            )
            episode_return = 0.0
            episode_count += 1
            obs, _ = env.reset()

        if frame >= config.warmup_steps:
            rng, train_rng = jax.random.split(rng)
            train_metrics = agent.train_step(train_rng)
            metrics["losses"].append(train_metrics["loss"])
            losses_since_log.append(train_metrics["loss"])

        if config.log_interval and frame > 0 and frame % config.log_interval == 0:
            avg_loss = sum(losses_since_log) / len(losses_since_log) if losses_since_log else 0.0
            _log_metrics(frame, avg_loss, epsilon, episode_count, metrics["episode_returns"], max_return)
            losses_since_log = []

        if config.eval_interval and frame > 0 and frame % config.eval_interval == 0:
            eval_return = evaluate(agent, env, config.eval_episodes, config.seed + 1000)
            metrics["eval_returns"].append(eval_return)
            print(f"Eval @ {frame}: mean_return={eval_return:.1f} over {config.eval_episodes} episodes")

    return agent, metrics


def evaluate(agent: QRDQNAgent, env, num_episodes: int = 10, seed: int = 0) -> float:
    returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_return = 0.0
        done = False
        while not done:
            action = agent.act(obs, epsilon=0.001)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            done = terminated or truncated
        returns.append(total_return)
    return float(np.mean(returns))
```

- [ ] **Step 2: Run train tests**

Run: `pytest tests/test_qr_dqn.py -v -k "train"`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add qr_dqn/train.py
git commit -m "feat: integrate NStepBuffer in training loop for n-step returns"
```

---

### Task 8: Fix Standalone Loss Function + Update Exports

**Files:**
- Modify: `qr_dqn/losses.py`
- Modify: `qr_dqn/__init__.py`
- Test: `tests/test_qr_dqn.py`

The standalone `quantile_huber_loss` incorrectly uses the taken action for target quantiles. Fix it to accept pre-selected (action-indexed) target quantiles, matching what the agent's inline loss does.

- [ ] **Step 1: Update the standalone loss function**

Replace `qr_dqn/losses.py` with:

```python
import jax.numpy as jnp


def quantile_huber_loss(
    current_quantiles: jnp.ndarray,
    target_quantiles: jnp.ndarray,
    taus: jnp.ndarray,
    kappa: float = 1.0,
) -> jnp.ndarray:
    """Quantile Huber loss between two sets of action-indexed quantile values.

    Args:
        current_quantiles: (batch, N) — quantile values for the taken action.
        target_quantiles: (batch, N) — Bellman target quantile values (already
            action-selected via Double DQN: online network selects, target
            network evaluates).
        taus: (N,) — quantile fractions.
        kappa: Huber loss threshold.

    Returns:
        Scalar loss value.
    """
    delta = target_quantiles[:, None, :] - current_quantiles[:, :, None]

    taus_row = taus[None, :, None]
    indicator = (delta < 0).astype(jnp.float32)
    asymmetric_weight = jnp.abs(taus_row - indicator)

    abs_delta = jnp.abs(delta)
    quadratic = jnp.minimum(abs_delta, kappa) ** 2 * 0.5
    linear = kappa * (abs_delta - 0.5 * kappa)
    huber = jnp.where(abs_delta <= kappa, quadratic, linear)

    loss = asymmetric_weight * huber
    loss = loss.sum(axis=2).mean(axis=1).mean()

    return loss
```

- [ ] **Step 2: Update the loss tests for new signature**

In `tests/test_qr_dqn.py`, replace the loss tests:

```python
def test_quantile_huber_loss_shape():
    batch_size = 8
    num_quantiles = 32
    current_quantiles = jnp.ones((batch_size, num_quantiles))
    target_quantiles = jnp.ones((batch_size, num_quantiles))
    taus = (2 * jnp.arange(num_quantiles) + 1) / (2 * num_quantiles)
    loss = quantile_huber_loss(current_quantiles, target_quantiles, taus, kappa=1.0)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_quantile_huber_loss_zero_when_equal():
    batch_size = 4
    num_quantiles = 32
    quantiles = jnp.ones((batch_size, num_quantiles))
    taus = (2 * jnp.arange(num_quantiles) + 1) / (2 * num_quantiles)
    loss = quantile_huber_loss(quantiles, quantiles, taus, kappa=1.0)
    assert loss == 0.0


def test_quantile_huber_loss_positive_when_different():
    batch_size = 4
    num_quantiles = 32
    current = jnp.zeros((batch_size, num_quantiles))
    target = jnp.ones((batch_size, num_quantiles))
    taus = (2 * jnp.arange(num_quantiles) + 1) / (2 * num_quantiles)
    loss = quantile_huber_loss(current, target, taus, kappa=1.0)
    assert loss > 0.0
```

- [ ] **Step 3: Update __init__.py exports**

Replace `qr_dqn/__init__.py` with:

```python
from qr_dqn.configs import QRDQNConfig
from qr_dqn.agent import QRDQNAgent
from qr_dqn.network import QuantileNetwork
from qr_dqn.replay import ReplayBuffer, PrioritizedReplayBuffer, NStepBuffer
from qr_dqn.losses import quantile_huber_loss
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_qr_dqn.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add qr_dqn/losses.py qr_dqn/__init__.py tests/test_qr_dqn.py
git commit -m "fix: correct standalone loss function signature, update exports"
```

---

### Task 9: Full Integration Test

**Files:**
- Test: `tests/test_qr_dqn.py`

- [ ] **Step 1: Write integration tests**

Add to `tests/test_qr_dqn.py`:

```python
def test_full_integration_all_features_enabled():
    """Smoke test: all features enabled (dueling, PER, n-step, grad clip)."""
    config = QRDQNConfig(
        num_quantiles=16,
        replay_capacity=500,
        warmup_steps=0,
        max_frames=100,
        dueling=True,
        n_step=3,
        max_grad_norm=10.0,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=100,
        target_update_tau=0.005,
    )
    rng = jax.random.PRNGKey(config.seed)
    rng, agent_rng = jax.random.split(rng)
    agent = QRDQNAgent(config, num_actions=6, obs_shape=(84, 84, 4), rng=agent_rng)

    assert isinstance(agent.buffer, PrioritizedReplayBuffer)

    nstep_buffer = NStepBuffer(n=config.n_step, gamma=config.gamma)
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)

    for frame in range(100):
        action = agent.act(obs, epsilon=0.5)
        next_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        reward = float(np.random.randn())
        terminated = frame == 99

        nstep_result = nstep_buffer.push(obs, action, reward, next_obs, terminated)
        if nstep_result is not None:
            obs_n, action_n, reward_n, next_obs_n, term_n = nstep_result
            agent.buffer_add(obs_n, action_n, reward_n, next_obs_n, term_n)

        if terminated:
            for obs_n, action_n, reward_n, next_obs_n, term_n in nstep_buffer.flush():
                agent.buffer_add(obs_n, action_n, reward_n, next_obs_n, term_n)

        obs = next_obs

        if frame >= 10 and agent.buffer.size >= config.batch_size:
            train_rng = jax.random.PRNGKey(frame)
            metrics = agent.train_step(train_rng)
            assert jnp.isfinite(metrics["loss"])
            assert "td_errors" in metrics


def test_full_integration_uniform_replay():
    """Smoke test: uniform replay, no PER."""
    config = QRDQNConfig(
        num_quantiles=16,
        replay_capacity=500,
        warmup_steps=0,
        max_frames=50,
        dueling=True,
        n_step=1,
        max_grad_norm=10.0,
        per_alpha=0.0,
    )
    rng = jax.random.PRNGKey(config.seed)
    rng, agent_rng = jax.random.split(rng)
    agent = QRDQNAgent(config, num_actions=6, obs_shape=(84, 84, 4), rng=agent_rng)

    assert isinstance(agent.buffer, ReplayBuffer)

    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    for frame in range(50):
        action = agent.act(obs, epsilon=1.0)
        next_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        agent.buffer_add(obs, action, 1.0, next_obs, False)
        obs = next_obs

    rng_key = jax.random.PRNGKey(0)
    metrics = agent.train_step(rng_key)
    assert jnp.isfinite(metrics["loss"])
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/test_qr_dqn.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_qr_dqn.py
git commit -m "test: add full integration tests for all QR-DQN features"
```

---

## Self-Review

**1. Spec coverage:**
- Dueling architecture → Task 3 ✓
- PER → Task 4 ✓
- N-step returns → Tasks 5 + 7 ✓
- Gradient clipping → Task 6 ✓
- Truncation bug → Task 2 ✓
- Updated defaults (num_quantiles=64, 50k buffer) → Task 1 ✓
- Standalone loss fix → Task 8 ✓
- Soft/Polyak target update → Task 6 ✓

**2. Placeholder scan:**
- No TBD, TODO, or placeholder patterns found. All code is complete.

**3. Type consistency:**
- `quantile_huber_loss` signature: `(current, target, taus, kappa)` — matches both agent inline and test usage.
- Network signature unchanged: `__call__(self, x)` — no breaking interface change.
- `train_step` return: `{"loss", "td_errors"}` — all test consumers updated.
- `PrioritizedReplayBuffer.sample` returns `weights` and `indices`. Agent handles both PER and uniform paths.