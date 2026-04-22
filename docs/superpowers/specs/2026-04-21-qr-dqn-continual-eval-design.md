# QR-DQN Continual Learning Evaluation Script — Design Spec

**Date:** 2026-04-21
**Status:** Approved
**Purpose:** Integrate the JAX QR-DQN agent into the existing continual RL benchmark framework by training sequentially on Atari games and evaluating on all seen tasks.

## Overview

The existing QR-DQN (`qr_dqn/`) is a single-task Atari agent. The existing continual benchmark (`experiments/evaluate_or_train.py`) supports PyTorch SAC-based agents on MetaWorld/Procgen. This design bridges the gap: a new evaluation script that reuses the QR-DQN agent and continual metrics infrastructure to run standard continual learning experiments on a sequence of Atari games.

## Architecture

### New Files

| File | Responsibility |
|------|---------------|
| `qr_dqn/continual_env.py` | `ContinualAtariEnv` — sequential Atari task manager |
| `experiments/run_qr_dqn_continual.py` | CLI entry point, train/eval loop, checkpointing |

### Existing Files Reused

| File | Role in this design |
|------|---------------------|
| `qr_dqn/configs.py` | `QRDQNConfig` (extended with `steps_per_task`) |
| `qr_dqn/agent.py` | `QRDQNAgent` (acts, trains, checkpoints) |
| `qr_dqn/train.py` | `train()` and `evaluate()` (adapted to accept pre-built env/agent) |
| `src/evaluation/evaluator.py` | `Evaluator` — TensorBoard, CSV logging |
| `src/evaluation/metrics.py` | `ContinualMetrics` — forgetting, transfer |

---

## Component: `ContinualAtariEnv`

### Interface

```python
class ContinualAtariEnv:
    def __init__(self, games: List[str], steps_per_task: int, seed: int):
        self.games = games
        self.steps_per_task = steps_per_task
        self.seed = seed
        self.current_task_idx = 0
        self.steps_in_task = 0
        self.env = make_atari_env(games[0], seed=seed)

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset(seed=self.seed + self.current_task_idx)
        return obs

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps_in_task += 1
        task_switched = False
        if self.steps_in_task >= self.steps_per_task:
            task_switched = True
            self._switch_task()
        info["task_idx"] = self.current_task_idx
        info["task_name"] = self.games[self.current_task_idx]
        info["task_switched"] = task_switched
        return obs, reward, terminated, truncated, info

    def _switch_task(self):
        """Close old env, create new env for next game."""
        self.env.close()
        self.current_task_idx = (self.current_task_idx + 1) % len(self.games)
        self.steps_in_task = 0
        self.env = make_atari_env(self.games[self.current_task_idx], seed=self.seed)
        self.env.reset(seed=self.seed + self.current_task_idx)

    def get_current_game(self) -> str: ...
    def get_num_tasks(self) -> int: ...
    def close(self): ...
```

### Key Behaviors
- **State isolation:** On task switch, the old env (including its 4-frame stack) is fully closed. A fresh env is created. This prevents frame-stack leakage between games.
- **Task metadata:** `info` dict carries `task_idx`, `task_name`, and `task_switched` for the training loop to log and track.

---

## Component: `run_qr_dqn_continual.py`

### Per-Task Flow

For each task `t` in `games`:

```
1. Check checkpoint for task t
   - If exists: load agent → evaluate on task t → record result
   - If not: train agent on task t for steps_per_task frames → save checkpoint

2. Evaluate agent on ALL tasks 0..t (task-boundary evaluation)
   - For each seen task i:
       - Set env to task i
       - Run evaluate(agent, env, eval_episodes)
       - Log per-task return
   - Update ContinualMetrics with per-task returns

3. Log continual metrics
   - average_return, forward_transfer, backward_transfer, forgetting_measure

4. Switch env to next task (env handles this)
```

### Training Function Adaptation

The existing `qr_dqn/train.py:train()` initializes a fresh agent and env. We extract a `train_on_task(agent, env, config, steps_per_task, evaluator, metrics_tracker)` helper that:

- Takes a **pre-initialized** agent and env.
- Runs the epsilon-greedy loop for `steps_per_task` frames.
- Logs episodes via `evaluator.log_episode()`.
- Runs periodic evaluation on the current task via `evaluate()`.
- Returns the trained agent and updated metrics.

**No parameter reset between tasks.** The agent's network params, target params, and replay buffer persist across tasks. This is the standard continual learning setup — we measure how well the same network adapts to new tasks without forgetting old ones.

### Checkpoint Format

QR-DQN is JAX/Flax. Checkpoints save:
- `params` (online network)
- `target_params`
- `opt_state`
- `step_count`
- `rng`

Using `flax.serialization.to_bytes()` or `orbax-checkpoint` if available. Fallback to `pickle` with `jax.tree_util.tree_map(np.array, params)` for portability.

```python
checkpoint = {
    "params": params_bytes,
    "target_params": target_params_bytes,
    "opt_state": opt_state_bytes,
    "step_count": agent.step_count,
    "rng": agent.rng,
}
```

Path: `checkpoints/qr_dqn/qr_dqn_task{t}_seed{seed}.pkl`

### Evaluation at Task Boundaries

After training on task `t`, we evaluate on every task `i ∈ [0, t]`:

```python
for i in range(t + 1):
    env = make_atari_env(games[i], seed=eval_seed)
    ret = evaluate(agent, env, eval_episodes, seed)
    continual_metrics.log_task_evaluation(t, i, ret)
```

This produces the per-task performance matrix required for `ContinualMetrics.compute_backward_transfer()` and `compute_forgetting_measure()`.

---

## Metrics & Logging

### ContinualMetrics Integration

`ContinualMetrics` is instantiated once for the full run. At each task boundary:
- `metrics_tracker.end_task(t)` — computes average return for the just-finished task.
- Per-task eval returns are stored for the forgetting matrix.
- `metrics_tracker.get_summary()` yields:
  - `average_return`
  - `forward_transfer`
  - `backward_transfer`
  - `forgetting_measure`

### Evaluator Integration

`Evaluator` logs:
- Episode returns (per episode)
- Training losses (from `agent.train_step()`)
- Periodic eval returns (during training on current task)
- Task-boundary eval returns (on all seen tasks)
- Continual metrics summary (at each task boundary)

Output directory: `results/qr_dqn_continual_seed{seed}/`

### CSV Output

The evaluator writes a `metrics.csv` with columns:
```
step, episode, return, length, task_idx, task_name, task_switched
```

Additional columns for task-boundary eval:
```
eval_task_idx, eval_return
```

---

## CLI Interface

```bash
python experiments/run_qr_dqn_continual.py \
    --games Breakout SpaceInvaders Freeway \
    --steps-per-task 500000 \
    --eval-interval 25000 \
    --eval-episodes 10 \
    --checkpoint-dir checkpoints \
    --results-dir results \
    --seed 42 \
    --smoke-test
```

### Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--games` | `Breakout SpaceInvaders Freeway` | Ordered list of Atari games |
| `--steps-per-task` | 500000 | Training frames per game |
| `--eval-interval` | 25000 | Frames between periodic evals |
| `--eval-episodes` | 10 | Episodes per evaluation |
| `--checkpoint-dir` | `checkpoints` | Where to save/load checkpoints |
| `--results-dir` | `results` | Where to save logs |
| `--seed` | 42 | Random seed |
| `--smoke-test` | False | Run minimal test (5000 steps/game) |
| `--resume` | False | Resume from existing checkpoints |

---

## Task Sequence

Default: **Breakout → SpaceInvaders → Freeway**

These games are visually and mechanically distinct (brick-breaking, shooting, racing) while sharing the Atari action space, making them a good continual learning testbed. The action space is shared across all Atari games (usually 18 discrete actions), so the QR-DQN network architecture does not need to change between tasks.

---

## FAME Hooks Reuse

The QR-DQN agent already exposes FAME-ready hooks:
- `copy_params()` — for meta-learner target network
- `distill_from(other)` — for fast-learner initialization
- `reset_params(rng)` — for random reinitialization experiments
- `get_quantiles(obs)` — for policy evaluation / behavior cloning

These are not used in the single-agent continual script, but the script is designed so that a future FAME dual-learner version can swap in a `DualQRDQNAgent` (fast + meta + gating) with minimal changes.

---

## Testing Strategy

1. **Unit test:** `ContinualAtariEnv` task switching closes old env and creates new env correctly.
2. **Integration test:** Run `run_qr_dqn_continual.py` with `--smoke-test` (3 games × 1000 steps). Verify:
   - Checkpoints are created for each task.
   - CSV log contains entries for all tasks.
   - `ContinualMetrics.get_summary()` returns non-zero forgetting after task 2.
3. **Checkpoint test:** Save checkpoint after task 0, load it, resume task 1. Verify loss continuity.
4. **Eval test:** After training task 1, evaluate on task 0. Verify return is lower than when task 0 ended (detect forgetting).

---

## Success Criteria

- Script runs end-to-end on Breakout → SpaceInvaders → Freeway without crashes.
- Checkpoints are saved after each task and loadable for resume.
- `ContinualMetrics` produces plausible forgetting and transfer values.
- Periodic evaluation during training logs per-task returns.
- Task-boundary evaluation on all seen tasks logs a per-task performance matrix.
- Smoke test completes in under 5 minutes.
