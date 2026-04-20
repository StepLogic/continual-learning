from dataclasses import dataclass, field


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