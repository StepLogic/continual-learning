# QR-DQN Training Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured terminal logging for QR-DQN training metrics at configurable frame intervals.

**Architecture:** Add a `log_interval` config field and a `_log_metrics()` helper in `train.py` that prints averaged loss, epsilon, episode count, mean return, and max return at each interval boundary. Eval logging prints a separate line. No new files or external dependencies.

**Tech Stack:** Python, JAX, existing `qr_dqn` package

---

### Task 1: Add `log_interval` to QRDQNConfig

**Files:**
- Modify: `qr_dqn/configs.py`

- [ ] **Step 1: Add `log_interval` field to `QRDQNConfig`**

Open `qr_dqn/configs.py` and add `log_interval: int = 10_000` as the last field in the dataclass:

```python
@dataclass
class QRDQNConfig:
    num_quantiles: int = 32
    kappa: float = 1.0
    learning_rate: float = 1e-4
    adam_eps: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 150_000
    batch_size: int = 32
    replay_capacity: int = 50_000
    warmup_steps: int = 10_000
    target_update_freq: int = 1_000
    gamma: float = 0.99
    max_frames: int = 5_000_000
    hidden_dim: int = 256
    eval_interval: int = 25_000
    eval_episodes: int = 10
    num_envs: int = 1
    seed: int = 42
    game: str = "Pong"
    log_interval: int = 10_000
```

- [ ] **Step 2: Run existing tests to verify no breakage**

Run: `python -m pytest tests/test_qr_dqn.py::test_config_defaults tests/test_qr_dqn.py::test_config_override -v`
Expected: Both PASS. The new field has a default so existing construction is unaffected.

- [ ] **Step 3: Commit**

```bash
git add qr_dqn/configs.py
git commit -m "feat: add log_interval to QRDQNConfig"
```

---

### Task 2: Add `_log_metrics` helper to `train.py`

**Files:**
- Modify: `qr_dqn/train.py`

- [ ] **Step 1: Write the `_log_metrics` helper function**

Add this function in `qr_dqn/train.py` after the `get_epsilon` function (after line 17):

```python
def _log_metrics(frame, avg_loss, epsilon, episode_count, episode_returns, max_return):
    recent = episode_returns[-10:] if episode_returns else [0.0]
    mean_ret = sum(recent) / len(recent)
    print(
        f"[Frame {frame:>7d}] loss={avg_loss:.3f} | eps={epsilon:.3f} "
        f"| episodes={episode_count} | mean_ret(10)={mean_ret:.1f} | max_ret={max_return:.1f}"
    )
```

- [ ] **Step 2: Run existing tests to verify no breakage**

Run: `python -m pytest tests/test_qr_dqn.py -v`
Expected: All tests PASS. The helper is a standalone function not yet called.

- [ ] **Step 3: Commit**

```bash
git add qr_dqn/train.py
git commit -m "feat: add _log_metrics helper for terminal logging"
```

---

### Task 3: Wire logging into the training loop

**Files:**
- Modify: `qr_dqn/train.py`

- [ ] **Step 1: Add loss accumulator and wire `_log_metrics` calls**

Replace the `train` function body to add loss accumulation and interval-based logging. The full replacement for the `train` function (lines 20-66):

```python
def train(config: QRDQNConfig, max_frames_override: int = None):
    max_frames = max_frames_override or config.max_frames
    env = make_atari_env(config.game, seed=config.seed)
    rng = jax.random.PRNGKey(config.seed)
    rng, agent_rng = jax.random.split(rng)

    num_actions = env.action_space.n
    obs, _ = env.reset()
    obs_shape = obs.shape

    agent = QRDQNAgent(config, num_actions=num_actions, obs_shape=obs_shape, rng=agent_rng)

    metrics = {"losses": [], "episode_returns": [], "eval_returns": []}
    episode_return = 0.0
    episode_count = 0
    max_return = float("-inf")
    losses_since_log = []

    for frame in range(max_frames):
        epsilon = get_epsilon(frame, config)
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.buffer_add(obs, action, reward, next_obs, done)
        episode_return += reward
        obs = next_obs

        if done:
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
```

Key changes from original:
- Added `losses_since_log = []` accumulator before the loop
- After `train_step`, append loss to `losses_since_log`
- Added `log_interval` check block that averages losses, calls `_log_metrics`, and resets the accumulator
- Added `print` line after evaluation call

- [ ] **Step 2: Run all tests to verify no breakage**

Run: `python -m pytest tests/test_qr_dqn.py -v`
Expected: All tests PASS. The training loop logic is unchanged for the warmup/eval paths; only logging is added.

- [ ] **Step 3: Commit**

```bash
git add qr_dqn/train.py
git commit -m "feat: wire frame-interval logging into training loop"
```

---

### Task 4: Add `--log_interval` CLI argument to `run_qr_dqn.py`

**Files:**
- Modify: `experiments/run_qr_dqn.py`

- [ ] **Step 1: Add `--log_interval` argument to argparse**

Add after the `--hidden_dim` argument (after line 24 in `run_qr_dqn.py`):

```python
    parser.add_argument("--log_interval", type=int, default=10_000)
```

And add `log_interval=args.log_interval,` to the `QRDQNConfig(...)` construction:

```python
    config = QRDQNConfig(
        game=args.game,
        num_quantiles=args.num_quantiles,
        kappa=args.kappa,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_frames=args.max_frames,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        gamma=args.gamma,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        hidden_dim=args.hidden_dim,
        log_interval=args.log_interval,
    )
```

- [ ] **Step 2: Run all tests to verify no breakage**

Run: `python -m pytest tests/test_qr_dqn.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add experiments/run_qr_dqn.py
git commit -m "feat: add --log_interval CLI argument"
```

---

### Task 5: Add unit test for `_log_metrics`

**Files:**
- Modify: `tests/test_qr_dqn.py`

- [ ] **Step 1: Write test for `_log_metrics` output format**

Add to `tests/test_qr_dqn.py` — import `_log_metrics` from `qr_dqn.train` and add:

```python
from qr_dqn.train import _log_metrics


def test_log_metrics_output(capsys):
    _log_metrics(
        frame=10000,
        avg_loss=0.523,
        epsilon=0.94,
        episode_count=42,
        episode_returns=[1.0, 2.0, 3.0],
        max_return=3.0,
    )
    captured = capsys.readouterr()
    assert "[Frame   10000]" in captured.out
    assert "loss=0.523" in captured.out
    assert "eps=0.940" in captured.out
    assert "episodes=42" in captured.out
    assert "mean_ret(10)=2.0" in captured.out
    assert "max_ret=3.0" in captured.out
```

- [ ] **Step 2: Run the new test to verify it passes**

Run: `python -m pytest tests/test_qr_dqn.py::test_log_metrics_output -v`
Expected: PASS

- [ ] **Step 3: Run all tests to verify nothing is broken**

Run: `python -m pytest tests/test_qr_dqn.py -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_qr_dqn.py
git commit -m "test: add unit test for _log_metrics output format"
```