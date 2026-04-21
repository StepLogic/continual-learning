# QR-DQN in JAX — Design Spec

**Date:** 2026-04-17
**Status:** Draft
**Purpose:** Implement Quantile Regression DQN in JAX (Flax + Optax) for Atari environments, with FAME-ready abstractions for future continual RL integration.

## Overview

QR-DQN (Dabney et al., 2018) replaces DQN's single Q-value estimate with a distribution over N quantiles. This enables richer value representations that capture the stochasticity of returns. The implementation targets Atari 2600 games and is designed to integrate with FAME's dual-learner continual RL architecture in the future.

## Architecture Decision

**Monolithic Agent** approach: a single `QRDQNAgent` class encapsulating the network, training step, and replay buffer. FAME integration hooks are added as optional methods on the agent.

Rationale: simpler to implement and test, matches the paper pseudocode, and the FAME-ready hooks make dual-learner composition straightforward without upfront complexity.

## Project Structure

```
continual_learning/
├── FAME/                          # existing
├── qr_dqn/                        # new
│   ├── __init__.py                # exports QRDQNAgent, QRDQNConfig
│   ├── agent.py                   # QRDQNAgent class (monolithic)
│   ├── network.py                 # QuantileNetwork (Flax module)
│   ├── replay.py                  # JAX-native replay buffer
│   ├── train.py                   # Training loop + evaluation
│   ├── configs.py                 # QRDQNConfig dataclass
│   └── atari_wrapper.py           # Atari env preprocessing
├── experiments/
│   └── run_qr_dqn.py             # Entry point
└── tests/
    └── test_qr_dqn.py            # Unit tests
```

## Components

### QuantileNetwork (`network.py`)

Flax module implementing the QR-DQN architecture:

- **Input:** stacked grayscale frames `(batch, 84, 84, 4)`
- **CNN backbone:** Conv(32, 8×8, stride 4) → ReLU → Conv(64, 4×4, stride 2) → ReLU → Conv(64, 3×3, stride 1) → ReLU → Flatten
- **Head:** Dense(512) → ReLU → Dense(num_actions × num_quantiles) → reshape to `(batch, num_actions, num_quantiles)`
- **Output:** quantile values per action per state

### QRDQNAgent (`agent.py`)

Monolithic agent class containing:

- **Network:** online QuantileNetwork + target QuantileNetwork
- **Optimizer:** Optax Adam (lr=1e-4, eps=1e-4)
- **Replay buffer:** JAX-native uniform replay buffer (capacity 100k)
- **Training state:** `TrainState` carrying params, target params, optimizer state, step counter, rng key

Key methods:

| Method | Purpose |
|--------|---------|
| `act(obs, epsilon)` | Select action via ε-greedy over mean quantile values |
| `train_step(batch)` | Single gradient update (JIT-compiled) |
| `update_target()` | Copy online params → target params |
| `get_quantiles(obs)` | Return quantile distribution for all actions — **FAME-ready** |
| `copy_params()` | Deep copy of network parameters — **FAME-ready** |
| `reset_params(rng)` | Reinitialize all parameters — **FAME-ready** |
| `distill_from(other_agent)` | Copy parameters from another QRDQNAgent — **FAME-ready** |

### Replay Buffer (`replay.py`)

JAX-native circular replay buffer:

- Stores `(state, action, reward, next_state, done)` as flat `jax.numpy` arrays
- Uniform sampling via `jax.random.choice`
- Capacity: configurable, default 100k
- Thread-safe for single-agent training

### Atari Wrapper (`atari_wrapper.py`)

Standard Atari preprocessing pipeline (Gymnasium-based):

1. Grayscale conversion
2. Resize to 84×84
3. Max-pooling over last 2 frames (flicker removal)
4. Frame skipping (4 frames per action)
5. Reward clipping to [-1, +1]
6. Life tracking (end episode on life loss)
7. Frame stacking (4 frames)

### Configuration (`configs.py`)

```python
@dataclass
class QRDQNConfig:
    num_quantiles: int = 64
    kappa: float = 1.0
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 250_000
    batch_size: int = 32
    replay_capacity: int = 100_000
    warmup_steps: int = 20_000
    target_update_freq: int = 1_000
    gamma: float = 0.99
    max_frames: int = 10_000_000
    eval_interval: int = 50_000
    eval_episodes: int = 10
```

## Algorithm Details

### Quantile Regression Loss

The quantile Huber loss operates over all N² pairs of quantiles:

1. Compute cumulative density thresholds: `τ_i = (2i + 1) / (2N)` for i ∈ {0, ..., N-1}
2. For each transition, compute Bellman target quantiles: `Tθ_j = r + γ * target_quantiles(next_state, argmax_a(mean(online_quantiles(next_state))))`
   - Double Q-learning: action selected by online network, evaluated by target network
3. Compute pairwise differences: `δ_ij = Tθ_j - θ_i(s, a)`
4. Apply asymmetric Huber loss with threshold κ=1:
   - `L(δ, τ) = |τ - I(δ < 0)| * H_κ(δ)` where `H_κ(δ) = 0.5δ²` if `|δ| ≤ κ`, else `κ(|δ| - 0.5κ)`
5. Average over all quantile pairs and batch

### Training Procedure

1. Collect warmup transitions (random actions) for `warmup_steps`
2. For each training step:
   a. Select action via ε-greedy (ε linearly decayed)
   b. Execute action, observe transition, store in replay buffer
   c. Sample batch from replay buffer
   d. Compute quantile Huber loss with double Q-learning
   e. Update online network parameters via Optax Adam
   f. Periodically update target network (hard update every `target_update_freq` steps)
3. Evaluate every `eval_interval` steps (10 episodes, ε=0.001)
4. Save checkpoints (best model + latest every 100k steps)

### Epsilon Schedule

Linear annealing from `epsilon_start=1.0` to `epsilon_end=0.01` over `epsilon_decay_steps=250,000` frames.

### Exploration

- During training: ε-greedy with linear decay
- Action selection: `argmax_a(mean(quantiles(s, a)))` — take the action with highest expected return
- During evaluation: ε=0.001 (near-greedy)

## FAME-Ready Hooks

The agent exposes methods that map directly to FAME's dual-learner operations:

| FAME Operation | QR-DQN Hook | Description |
|----------------|-------------|-------------|
| Fast learner initialization | `distill_from(meta_agent)` | Copy meta learner params to fast learner |
| Random initialization | `reset_params(rng)` | Reinitialize all parameters |
| Meta learner update | `copy_params()` | Deep copy of parameters for target/meta network |
| Detection (policy evaluation) | `get_quantiles(obs)` | Get full quantile distribution for action comparison |
| Behavior cloning regularization | `get_quantiles(obs)` | KL or WD between fast and meta quantile distributions |

These hooks enable future FAME integration without modifying the core QR-DQN algorithm.

## Dependencies

```
jax>=0.4.20
jaxlib>=0.4.20
flax>=0.7.0
optax>=0.1.7
gymnasium[atari]>=0.29.0
ale-py>=0.8.1
numpy>=1.24.0
tqdm>=4.65.0
tensorboard>=2.14.0
```

No PyTorch dependency. Pure JAX/Flax/Optax implementation.

## Testing Strategy

1. **Unit tests** for QuantileNetwork forward pass shape correctness
2. **Unit tests** for replay buffer add/sample operations
3. **Unit tests** for quantile Huber loss computation (known values)
4. **Unit tests** for epsilon schedule
5. **Integration test:** train for 100 steps on a simple environment, verify loss decreases
6. **Smoke test:** run one full episode of Pong, verify no crashes

## Success Criteria

- QR-DQN agent trains on Atari Pong to ≥15 average return within 2M frames
- All unit tests pass
- JIT compilation works on GPU (no recompilation per step after warmup)
- FAME-ready hooks are functional (copy, reset, distill, get_quantiles)