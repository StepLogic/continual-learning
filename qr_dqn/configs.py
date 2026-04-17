from dataclasses import dataclass, field


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
    replay_capacity: int = 100_000
    warmup_steps: int = 20_000
    target_update_freq: int = 1_000
    gamma: float = 0.99
    max_frames: int = 10_000_000
    eval_interval: int = 50_000
    eval_episodes: int = 10
    num_envs: int = 1
    seed: int = 42
    game: str = "Pong"