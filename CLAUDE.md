# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Continual RL Benchmark — a research codebase comparing continual reinforcement learning methods. Three distinct components with different frameworks:

| Component | Framework | Environment | Purpose |
|-----------|-----------|-------------|---------|
| `qr_dqn/` | JAX/Flax/Optax | Atari (ALE) | QR-DQN agent for continual Atari learning (primary research contribution) |
| `src/` | PyTorch | MetaWorld | General benchmark framework with 6 continual RL methods built on SAC |
| `FAME/` | PyTorch | MinAtar, Atari, MetaWorld | Official FAME implementation (ICLR 2026, Sun et al.) — vendored reference |

The `qr_dqn/` agent is compared against FAME baselines via cross-game evaluation on Atari.

## Commands

```bash
# Install
pip install -e .              # Core
pip install -e ".[envs]"     # With environment packages
pip install -e ".[dev]"      # With linters (black, ruff)

# Tests
pytest                       # All tests
pytest tests/ -v             # Verbose
pytest tests/test_qr_dqn.py -v -k "agent"   # Specific test file/keyword

# QR-DQN experiments
python experiments/run_qr_dqn_continual.py   # Continual training across Atari games
python experiments/eval_qr_dqn_game.py       # Evaluate a saved checkpoint
bash experiments/run_qr_dqn_cross_game.sh    # Full train+eval pipeline

# FAME baselines (from FAME/Atari/)
cd FAME/Atari
python run_baselines_cross_game.py --model-type packnet --seed 1 --total-timesteps 1000000
bash run_baselines_cross_game.sh all 1 1000000   # Run all baselines

# Linting
black --line-length 100 .
ruff check .
```

## Architecture

### QR-DQN (`qr_dqn/`)

JAX-based QR-DQN with continual learning extensions:
- `configs.py` — `QRDQNConfig` dataclass with all hyperparameters
- `network.py` — `QuantileNetwork` (Flax Module): CNN + optional dueling (value/advantage streams)
- `agent.py` — `QRDQNAgent`: Double DQN, PER, n-step returns, gradient clipping, hard/soft target updates, checkpoint save/load, FAME integration hooks (`distill_from`, `reset_params`, `copy_params`)
- `losses.py` — `quantile_huber_loss` with asymmetric weighting
- `replay.py` — `ReplayBuffer` (uniform), `PrioritizedReplayBuffer` (alpha/beta/IS), `NStepBuffer`
- `atari_wrapper.py` — `make_atari_env()`: grayscale, 84x84, frame skip=4, reward clip, 4-frame stack
- `continual_env.py` — `ContinualAtariEnv`: sequential game switching with auto-task-boundary
- `train.py` — `train()`, `train_on_task()`, `evaluate()`, `evaluate_on_game()`

### Benchmark Framework (`src/`)

PyTorch SAC-based continual RL methods for MetaWorld:
- `src/agents/` — 6 agents: SAC (base), FastTRAC, Parseval, DualLearner, WorldModel, DualActor
- `src/environments/continual_env.py` — MetaWorld continual environment wrapper
- `src/evaluation/metrics.py` — `ContinualMetrics`: average return, forward/backward transfer, forgetting
- `src/evaluation/evaluator.py` — TensorBoard + CSV logging, checkpointing
- Configured via Hydra YAML in `continual_rl_benchmark/configs/`

### FAME (`FAME/`)

Self-contained reference implementation, not integrated into `src/`:
- `FAME/Atari/` — PPO-based with cross-game evaluation, 6 baseline models in `models/`
- `FAME/MinAtar/` — DQN-based on simplified Atari
- `FAME/Metaworld/` — SAC-based on MetaWorld

## Key Conventions

- **Framework split**: `qr_dqn/` uses JAX/Flax/Optax; `src/` and `FAME/` use PyTorch. Both are dependencies.
- **Cross-game evaluation protocol**: Train on Breakout (1 mode) → Freeway (8 modes) → SpaceInvaders (10 modes), then evaluate on all 19 game/mode combos. Action spaces padded to MAX_ACTIONS=6 with logits masking.
- **Config over code**: QR-DQN uses `QRDQNConfig` dataclass; `src/` uses Hydra YAML configs; FAME uses `tyro`-parsed dataclasses.
- **Checkpoint format**: QR-DQN uses `.pkl` (pickle); FAME baselines use `.pt` (PyTorch state dicts).