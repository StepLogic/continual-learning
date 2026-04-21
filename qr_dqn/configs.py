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
    batch_size: int = 8
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
    # Continual learning
    steps_per_task: int = 500_000
    games: tuple = ("Breakout", "SpaceInvaders", "Freeway")