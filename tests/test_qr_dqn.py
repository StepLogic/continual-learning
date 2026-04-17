import pytest
from qr_dqn.configs import QRDQNConfig


def test_config_defaults():
    config = QRDQNConfig()
    assert config.num_quantiles == 64
    assert config.kappa == 1.0
    assert config.learning_rate == 1e-4
    assert config.gamma == 0.99
    assert config.replay_capacity == 100_000


def test_config_override():
    config = QRDQNConfig(num_quantiles=32, kappa=0.5)
    assert config.num_quantiles == 32
    assert config.kappa == 0.5