# QR-DQN Continual Learning Evaluation Script — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evaluation script that trains QR-DQN sequentially on Atari games (Breakout → SpaceInvaders → Freeway) and evaluates on all seen tasks, producing continual learning metrics.

**Architecture:** Add `ContinualAtariEnv` for sequential Atari task management, add save/load to `QRDQNAgent`, adapt `train.py` with task-level helpers, and create `run_qr_dqn_continual.py` as the CLI entry point. Reuse existing `Evaluator` and `ContinualMetrics` for logging and metrics.

**Tech Stack:** JAX, Flax, Optax, Gymnasium[atari], NumPy

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `qr_dqn/configs.py` | Modify | Add `steps_per_task`, `games` to `QRDQNConfig` |
| `qr_dqn/agent.py` | Modify | Add `save()`, `load()` checkpoint methods to `QRDQNAgent` |
| `qr_dqn/continual_env.py` | Create | `ContinualAtariEnv` — sequential Atari task manager |
| `qr_dqn/train.py` | Modify | Add `train_on_task()` and `evaluate_on_game()` helpers |
| `experiments/run_qr_dqn_continual.py` | Create | CLI entry point, full continual learning loop |
| `tests/test_qr_dqn.py` | Modify | Add tests for `ContinualAtariEnv`, save/load, continual loop |

---

## Task 1: Extend QRDQNConfig

**Files:**
- Modify: `qr_dqn/configs.py`
- Test: `tests/test_qr_dqn.py`

- [ ] **Step 1: Write failing test for new config fields**

Add to `tests/test_qr_dqn.py`:

```python
def test_config_steps_per_task():
    config = QRDQNConfig(steps_per_task=100_000)
    assert config.steps_per_task == 100_000

def test_config_games():
    config = QRDQNConfig(games=["Breakout", "Pong"])
    assert config.games == ["Breakout", "Pong"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_qr_dqn.py::test_config_steps_per_task tests/test_qr_dqn.py::test_config_games -v`
Expected: FAIL (AttributeError — `steps_per_task` and `games` don't exist)

- [ ] **Step 3: Add fields to QRDQNConfig**

Modify `qr_dqn/configs.py`, add inside `QRDQNConfig`:

```python
    # Continual learning
    steps_per_task: int = 500_000
    games: tuple = ("Breakout", "SpaceInvaders", "Freeway")
```

Use `tuple` instead of `list` for immutability in the dataclass default.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_qr_dqn.py::test_config_steps_per_task tests/test_qr_dqn.py::test_config_games -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qr_dqn/configs.py tests/test_qr_dqn.py
git commit -m "feat: add steps_per_task and games to QRDQNConfig"
```

---

## Task 2: Add Save/Load to QRDQNAgent

**Files:**
- Modify: `qr_dqn/agent.py`
- Test: `tests/test_qr_dqn.py`

- [ ] **Step 1: Write failing tests for save/load**

Add to `tests/test_qr_dqn.py`:

```python
import tempfile
import os


def test_agent_save_load(rng):
    config = QRDQNConfig(num_quantiles=16, replay_capacity=100)
    agent = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng)
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    q_before = agent.get_quantiles(obs)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "agent.pkl")
        agent.save(path)
        assert os.path.exists(path)

        # Create fresh agent and load
        agent2 = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng)
        agent2.load(path)
        q_after = agent2.get_quantiles(obs)

    assert jnp.allclose(q_before, q_after)


def test_agent_save_load_step_count(rng):
    config = QRDQNConfig(num_quantiles=16, replay_capacity=100)
    agent = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng)
    agent.step_count = 42

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "agent.pkl")
        agent.save(path)

        agent2 = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng)
        agent2.load(path)
        assert agent2.step_count == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_qr_dqn.py::test_agent_save_load -v`
Expected: FAIL (AttributeError — `save` and `load` don't exist)

- [ ] **Step 3: Implement save/load**

Add to `qr_dqn/agent.py`, inside `QRDQNAgent` class (after `distill_from`):

```python
    def save(self, path: str):
        """Save agent checkpoint to disk."""
        import pickle
        checkpoint = {
            "params": self.params,
            "target_params": self.target_params,
            "opt_state": self.opt_state,
            "step_count": self.step_count,
            "rng": self.rng,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load(self, path: str):
        """Load agent checkpoint from disk."""
        import pickle
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        self.params = checkpoint["params"]
        self.target_params = checkpoint["target_params"]
        self.opt_state = checkpoint["opt_state"]
        self.step_count = checkpoint["step_count"]
        self.rng = checkpoint["rng"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_qr_dqn.py::test_agent_save_load tests/test_qr_dqn.py::test_agent_save_load_step_count -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qr_dqn/agent.py tests/test_qr_dqn.py
git commit -m "feat: add save/load checkpoint to QRDQNAgent"
```

---

## Task 3: Create ContinualAtariEnv

**Files:**
- Create: `qr_dqn/continual_env.py`
- Test: `tests/test_qr_dqn.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_qr_dqn.py`:

```python
from qr_dqn.continual_env import ContinualAtariEnv


def test_continual_env_init():
    env = ContinualAtariEnv(games=["Breakout"], steps_per_task=100, seed=0)
    assert env.get_num_tasks() == 1
    assert env.get_current_game() == "Breakout"


def test_continual_env_step_counts():
    env = ContinualAtariEnv(games=["Breakout", "Pong"], steps_per_task=5, seed=0)
    obs = env.reset()
    assert obs.shape == (84, 84, 4)

    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(env.env.action_space.sample())
        assert info["task_idx"] == 0
        assert info["task_switched"] is False

    # 5th step triggers switch
    obs, reward, terminated, truncated, info = env.step(env.env.action_space.sample())
    assert info["task_switched"] is True
    assert info["task_idx"] == 1
    assert env.get_current_game() == "Pong"


def test_continual_env_close():
    env = ContinualAtariEnv(games=["Breakout"], steps_per_task=10, seed=0)
    env.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_qr_dqn.py::test_continual_env_init -v`
Expected: FAIL (ImportError — `qr_dqn.continual_env` doesn't exist)

- [ ] **Step 3: Implement ContinualAtariEnv**

Create `qr_dqn/continual_env.py`:

```python
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

    def reset(self) -> np.ndarray:
        """Reset current environment."""
        obs, _ = self.env.reset(seed=self.seed + self.current_task_idx)
        return obs

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_qr_dqn.py::test_continual_env_init tests/test_qr_dqn.py::test_continual_env_step_counts tests/test_qr_dqn.py::test_continual_env_close -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qr_dqn/continual_env.py tests/test_qr_dqn.py
git commit -m "feat: add ContinualAtariEnv for sequential Atari tasks"
```

---

## Task 4: Adapt Training Helpers for Continual Learning

**Files:**
- Modify: `qr_dqn/train.py`
- Test: `tests/test_qr_dqn.py`

- [ ] **Step 1: Write failing test for evaluate_on_game**

Add to `tests/test_qr_dqn.py`:

```python
from qr_dqn.train import evaluate_on_game


def test_evaluate_on_game():
    config = QRDQNConfig(
        num_quantiles=8,
        replay_capacity=200,
        warmup_steps=32,
        batch_size=8,
        max_frames=200,
        target_update_freq=50,
        eval_interval=1000,
        game="Pong",
    )
    env = make_atari_env(config.game, seed=config.seed)
    rng = jax.random.PRNGKey(config.seed)
    rng, agent_rng = jax.random.split(rng)
    num_actions = env.action_space.n
    obs, _ = env.reset()
    obs_shape = obs.shape
    agent = QRDQNAgent(config, num_actions=num_actions, obs_shape=obs_shape, rng=agent_rng)

    # Train briefly
    for _ in range(100):
        action = agent.act(obs, epsilon=1.0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.buffer_add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    eval_return = evaluate_on_game(agent, "Pong", num_episodes=1, seed=config.seed)
    assert isinstance(eval_return, float)
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qr_dqn.py::test_evaluate_on_game -v`
Expected: FAIL (ImportError — `evaluate_on_game` doesn't exist)

- [ ] **Step 3: Implement train_on_task and evaluate_on_game**

Modify `qr_dqn/train.py`. Add after the existing `evaluate` function:

```python
def train_on_task(
    agent: QRDQNAgent,
    env,
    config: QRDQNConfig,
    steps_per_task: int,
    evaluator=None,
    metrics_tracker=None,
    task_idx: int = 0,
):
    """
    Train agent on a single task for a fixed number of steps.

    Args:
        agent: Pre-initialized QRDQNAgent (params persist across tasks).
        env: Environment to train on.
        config: QRDQNConfig.
        steps_per_task: Number of frames to train for this task.
        evaluator: Optional Evaluator for logging.
        metrics_tracker: Optional ContinualMetrics for tracking.
        task_idx: Index of current task.

    Returns:
        Updated agent, dict of metrics.
    """
    from src.evaluation.evaluator import Evaluator
    from src.evaluation.metrics import ContinualMetrics

    obs, _ = env.reset()
    episode_return = 0.0
    episode_count = 0
    max_return = float("-inf")
    losses_since_log = []
    metrics = {"losses": [], "episode_returns": [], "eval_returns": []}

    for frame in range(steps_per_task):
        epsilon = get_epsilon(agent.step_count, config)
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store terminated (not truncated) for correct bootstrapping
        agent.buffer_add(obs, action, reward, next_obs, terminated)

        episode_return += reward
        obs = next_obs

        if terminated or truncated:
            metrics["episode_returns"].append(episode_return)
            max_return = max(max_return, episode_return)

            if evaluator is not None:
                task_name = info.get("task_name", env.get_current_game())
                task_switched = info.get("task_switched", False)
                evaluator.log_episode(
                    episode_return=episode_return,
                    episode_length=frame,  # approximate
                    task_idx=task_idx,
                    task_name=task_name,
                    task_switched=task_switched,
                )

            if metrics_tracker is not None:
                metrics_tracker.log_episode(episode_return, task_idx, evaluator.global_step if evaluator else frame)

            episode_return = 0.0
            episode_count += 1
            obs, _ = env.reset()

        if agent.step_count >= config.warmup_steps:
            rng_key = jax.random.PRNGKey(config.seed + agent.step_count)
            train_metrics = agent.train_step(rng_key)
            metrics["losses"].append(train_metrics["loss"])
            losses_since_log.append(train_metrics["loss"])

        if config.log_interval and frame > 0 and frame % config.log_interval == 0:
            avg_loss = sum(losses_since_log) / len(losses_since_log) if losses_since_log else 0.0
            _log_metrics(frame, avg_loss, epsilon, episode_count, metrics["episode_returns"], max_return)
            losses_since_log = []

        if config.eval_interval and frame > 0 and frame % config.eval_interval == 0:
            eval_return = evaluate(agent, env, config.eval_episodes, config.seed + 1000)
            metrics["eval_returns"].append(eval_return)
            if evaluator is not None:
                evaluator.log_evaluation(eval_return, 0)
            print(f"Eval @ task {task_idx} frame {frame}: mean_return={eval_return:.1f}")

        if evaluator is not None:
            evaluator.step()

    return agent, metrics


def evaluate_on_game(agent: QRDQNAgent, game: str, num_episodes: int = 10, seed: int = 0) -> float:
    """
    Evaluate agent on a specific Atari game.

    Creates a fresh environment for the game, runs evaluation,
    and closes the environment.
    """
    env = make_atari_env(game, seed=seed)
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
    env.close()
    return float(np.mean(returns))
```

Also update the imports at the top of `qr_dqn/train.py` to include `Evaluator` and `ContinualMetrics`:

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_qr_dqn.py::test_evaluate_on_game -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qr_dqn/train.py tests/test_qr_dqn.py
git commit -m "feat: add train_on_task and evaluate_on_game helpers"
```

---

## Task 5: Create Continual Learning Runner Script

**Files:**
- Create: `experiments/run_qr_dqn_continual.py`
- Test: `tests/test_qr_dqn.py`

- [ ] **Step 1: Write integration test for continual runner**

Add to `tests/test_qr_dqn.py`:

```python
import tempfile
import os


def test_continual_runner_smoke():
    """Smoke test: 2 games, 500 steps each."""
    from experiments.run_qr_dqn_continual import run_continual_experiment

    config = QRDQNConfig(
        num_quantiles=8,
        replay_capacity=500,
        warmup_steps=50,
        batch_size=8,
        max_frames=500,
        target_update_freq=50,
        eval_interval=1000,  # disable periodic eval
        log_interval=0,
        steps_per_task=500,
        games=("Pong", "Breakout"),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_continual_experiment(
            config=config,
            seed=42,
            checkpoint_dir=os.path.join(tmpdir, "checkpoints"),
            results_dir=os.path.join(tmpdir, "results"),
            smoke_test=True,
        )
        assert "task_results" in results
        assert len(results["task_results"]) == 2
        assert "continual_metrics" in results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qr_dqn.py::test_continual_runner_smoke -v`
Expected: FAIL (ImportError — `run_qr_dqn_continual` doesn't exist)

- [ ] **Step 3: Implement run_qr_dqn_continual.py**

Create `experiments/run_qr_dqn_continual.py`:

```python
"""
Continual Learning Evaluation Script for QR-DQN.

Trains QR-DQN sequentially on Atari games and evaluates on all seen tasks.
Usage:
    python experiments/run_qr_dqn_continual.py \
        --games Breakout SpaceInvaders Freeway \
        --steps-per-task 500000 \
        --seed 42
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax

from qr_dqn.configs import QRDQNConfig
from qr_dqn.agent import QRDQNAgent
from qr_dqn.continual_env import ContinualAtariEnv
from qr_dqn.train import train_on_task, evaluate_on_game
from qr_dqn.atari_wrapper import make_atari_env
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import ContinualMetrics
from src.utils.logging import setup_logging, get_logger


def evaluate_all_tasks(agent: QRDQNAgent, games: List[str], eval_episodes: int, seed: int) -> Dict[int, float]:
    """Evaluate agent on all tasks seen so far."""
    per_task_returns = {}
    for task_idx, game in enumerate(games):
        eval_return = evaluate_on_game(agent, game, num_episodes=eval_episodes, seed=seed)
        per_task_returns[task_idx] = eval_return
        print(f"  Task {task_idx} ({game}): return={eval_return:.1f}")
    return per_task_returns


def run_continual_experiment(
    config: QRDQNConfig,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    smoke_test: bool = False,
) -> Dict:
    """
    Run continual learning experiment with QR-DQN.

    For each task:
    1. Check if checkpoint exists -> load and evaluate
    2. Else train on task -> save checkpoint
    3. Evaluate on all seen tasks
    4. Log continual metrics

    Returns:
        Dict with task_results and continual_metrics.
    """
    logger = get_logger()
    logger.info(f"Starting continual QR-DQN experiment (seed={seed})")

    # Set seeds
    np.random.seed(seed)

    # Determine parameters
    games = list(config.games)
    steps_per_task = 5000 if smoke_test else config.steps_per_task
    eval_episodes = 1 if smoke_test else config.eval_episodes
    eval_interval = 0 if smoke_test else config.eval_interval

    logger.info(f"Games: {games}")
    logger.info(f"Steps per task: {steps_per_task}")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create continual environment
    env = ContinualAtariEnv(games=games, steps_per_task=steps_per_task, seed=seed)

    # Get observation and action dimensions from first environment
    first_env = make_atari_env(games[0], seed=seed)
    num_actions = first_env.action_space.n
    obs, _ = first_env.reset()
    obs_shape = obs.shape
    first_env.close()

    # Create agent
    rng = jax.random.PRNGKey(seed)
    rng, agent_rng = jax.random.split(rng)
    agent = QRDQNAgent(config, num_actions=num_actions, obs_shape=obs_shape, rng=agent_rng)

    # Create evaluator
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"qr_dqn_continual_{timestamp}"
    evaluator = Evaluator(log_dir=results_dir, experiment_name=experiment_name, seed=seed)

    # Create metrics tracker
    metrics_tracker = ContinualMetrics(num_tasks=len(games))

    # Process each task
    all_task_results = []

    for task_idx in range(len(games)):
        game = games[task_idx]
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {task_idx}/{len(games)-1}: {game}")
        logger.info(f"{'='*60}")

        # Checkpoint path
        checkpoint_path = os.path.join(checkpoint_dir, f"qr_dqn_task{task_idx}_seed{seed}.pkl")

        if os.path.exists(checkpoint_path):
            # Load checkpoint and evaluate
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            agent.load(checkpoint_path)
            logger.info(f"Checkpoint loaded. Evaluating on task {task_idx}...")

            eval_return = evaluate_on_game(agent, game, num_episodes=eval_episodes, seed=seed)
            logger.info(f"Task {task_idx} eval return: {eval_return:.2f}")

            task_result = {
                "task_idx": task_idx,
                "game": game,
                "status": "evaluated",
                "checkpoint_path": checkpoint_path,
                "eval_return": eval_return,
            }
        else:
            # Train on this task
            logger.info(f"Training on task {task_idx} for {steps_per_task} steps...")

            # Set environment to this task
            while env.current_task_idx != task_idx:
                env._switch_task()

            # Create task-specific evaluator config
            task_config = QRDQNConfig(
                **{k: v for k, v in vars(config).items() if k not in ("steps_per_task", "games")}
            )
            task_config.eval_interval = eval_interval

            agent, task_metrics = train_on_task(
                agent=agent,
                env=env,
                config=task_config,
                steps_per_task=steps_per_task,
                evaluator=evaluator,
                metrics_tracker=metrics_tracker,
                task_idx=task_idx,
            )

            # Save checkpoint
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # End task in metrics tracker
            metrics_tracker.end_task(task_idx)

            task_result = {
                "task_idx": task_idx,
                "game": game,
                "status": "trained",
                "checkpoint_path": checkpoint_path,
                "num_episodes": len(task_metrics["episode_returns"]),
                "final_eval_returns": task_metrics["eval_returns"],
            }

        # Evaluate on all seen tasks
        logger.info(f"Evaluating on all tasks 0..{task_idx}...")
        seen_games = games[: task_idx + 1]
        per_task_returns = evaluate_all_tasks(agent, seen_games, eval_episodes, seed)

        # Log continual metrics
        continual_summary = metrics_tracker.get_summary()
        evaluator.log_continual_metrics(continual_summary)

        logger.info(f"Continual metrics: {continual_summary}")

        task_result["per_task_eval_returns"] = per_task_returns
        task_result["continual_metrics"] = continual_summary
        all_task_results.append(task_result)

        # Switch to next task if not last
        if task_idx < len(games) - 1:
            if not env.is_task_switch():
                env._switch_task()

    env.close()
    evaluator.close()

    # Compile final results
    final_results = {
        "config": {k: str(v) if isinstance(v, tuple) else v for k, v in vars(config).items()},
        "seed": seed,
        "task_results": all_task_results,
        "continual_metrics": metrics_tracker.get_summary(),
    }

    # Save results
    results_path = os.path.join(results_dir, f"qr_dqn_continual_seed{seed}.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description="QR-DQN Continual Learning on Atari")
    parser.add_argument("--games", type=str, nargs="+", default=["Breakout", "SpaceInvaders", "Freeway"],
                        help="Ordered list of Atari games")
    parser.add_argument("--steps-per-task", type=int, default=500_000,
                        help="Training steps per task")
    parser.add_argument("--eval-interval", type=int, default=25_000,
                        help="Steps between periodic evaluations")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Episodes per evaluation")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save/load checkpoints")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run minimal test")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoints")

    args = parser.parse_args()

    setup_logging(level="INFO")
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("QR-DQN Continual Learning Evaluation")
    logger.info("=" * 60)

    config = QRDQNConfig(
        games=tuple(args.games),
        steps_per_task=args.steps_per_task,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )

    results = run_continual_experiment(
        config=config,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        smoke_test=args.smoke_test,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Experiment complete!")
    logger.info(f"Final continual metrics: {results['continual_metrics']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run integration test**

Run: `python -m pytest tests/test_qr_dqn.py::test_continual_runner_smoke -v -s`
Expected: PASS (may take 1-2 minutes due to env creation)

- [ ] **Step 5: Commit**

```bash
git add experiments/run_qr_dqn_continual.py tests/test_qr_dqn.py
git commit -m "feat: add QR-DQN continual learning runner script"
```

---

## Task 6: Final Integration and Full Test Suite

**Files:**
- Modify: `tests/test_qr_dqn.py`
- Modify: `qr_dqn/__init__.py`

- [ ] **Step 1: Update qr_dqn/__init__.py exports**

Add to `qr_dqn/__init__.py`:

```python
from qr_dqn.continual_env import ContinualAtariEnv
```

- [ ] **Step 2: Add final integration test**

Add to `tests/test_qr_dqn.py`:

```python
class TestContinualLearning:
    def test_checkpoint_resume(self):
        """Test that training can resume from a checkpoint."""
        import tempfile
        config = QRDQNConfig(
            num_quantiles=8,
            replay_capacity=200,
            warmup_steps=20,
            batch_size=8,
            max_frames=200,
            target_update_freq=50,
            eval_interval=1000,
            log_interval=0,
            steps_per_task=200,
            games=("Pong",),
        )
        env = make_atari_env("Pong", seed=42)
        rng = jax.random.PRNGKey(42)
        rng, agent_rng = jax.random.split(rng)
        obs, _ = env.reset()
        agent = QRDQNAgent(config, num_actions=env.action_space.n, obs_shape=obs.shape, rng=agent_rng)

        # Train briefly
        for _ in range(100):
            action = agent.act(obs, epsilon=1.0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.buffer_add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        step_count_before = agent.step_count

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "agent.pkl")
            agent.save(path)

            agent2 = QRDQNAgent(config, num_actions=env.action_space.n, obs_shape=obs.shape, rng=agent_rng)
            agent2.load(path)
            assert agent2.step_count == step_count_before

        env.close()
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/test_qr_dqn.py -v`
Expected: ALL PASS

- [ ] **Step 4: Run smoke test of the full script**

Run: `python experiments/run_qr_dqn_continual.py --smoke-test --seed 42`
Expected: Runs 2 games × 5000 steps, saves checkpoints, prints continual metrics, no crashes.

- [ ] **Step 5: Commit**

```bash
git add qr_dqn/__init__.py tests/test_qr_dqn.py
git commit -m "test: add continual learning integration tests"
```

---

## Self-Review

**1. Spec coverage:**
- `ContinualAtariEnv` with task switching: Task 3 ✓
- Save/load checkpoints for QR-DQN: Task 2 ✓
- `train_on_task` helper (pre-built agent/env): Task 4 ✓
- `evaluate_on_game` helper: Task 4 ✓
- Task-boundary evaluation on all seen tasks: Task 5 ✓
- ContinualMetrics integration: Task 5 ✓
- Evaluator integration: Task 5 ✓
- CLI with `--games`, `--steps-per-task`, `--smoke-test`: Task 5 ✓
- Checkpoint resume: Task 5, Task 6 ✓

**2. Placeholder scan:** No TBDs, TODOs, or vague requirements found.

**3. Type consistency:**
- `QRDQNConfig.games` is `tuple` (immutable dataclass default) ✓
- `ContinualAtariEnv.games` is `List[str]` (constructor takes list) ✓
- `agent.save(path)` and `agent.load(path)` signatures consistent ✓
- `evaluate_on_game(agent, game, num_episodes, seed)` signature consistent ✓

**4. DRY/YAGNI check:**
- Reuses existing `Evaluator` and `ContinualMetrics` instead of duplicating logging ✓
- Reuses existing `make_atari_env` instead of reimplementing Atari wrapper ✓
- No unnecessary abstractions — single script, focused env class ✓

**5. Testing coverage:**
- Unit tests for config extension: Task 1 ✓
- Unit tests for save/load: Task 2 ✓
- Unit tests for ContinualAtariEnv: Task 3 ✓
- Integration test for evaluate_on_game: Task 4 ✓
- Smoke test for full continual runner: Task 5 ✓
- Checkpoint resume test: Task 6 ✓

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-21-qr-dqn-continual-eval.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach would you like?