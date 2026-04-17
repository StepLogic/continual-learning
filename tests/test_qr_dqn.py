import pytest
import jax
import jax.numpy as jnp
import numpy as np
from qr_dqn.configs import QRDQNConfig
from qr_dqn.network import QuantileNetwork


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


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


def test_network_forward_shape(rng):
    num_actions = 6
    num_quantiles = 64
    net = QuantileNetwork(num_actions=num_actions, num_quantiles=num_quantiles)
    dummy_obs = jnp.ones((4, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert output.shape == (4, num_actions, num_quantiles)


def test_network_single_obs(rng):
    num_actions = 6
    num_quantiles = 32
    net = QuantileNetwork(num_actions=num_actions, num_quantiles=num_quantiles)
    dummy_obs = jnp.ones((1, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert output.shape == (1, num_actions, num_quantiles)


def test_network_output_finite(rng):
    num_actions = 4
    num_quantiles = 64
    net = QuantileNetwork(num_actions=num_actions, num_quantiles=num_quantiles)
    dummy_obs = jnp.ones((2, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert jnp.all(jnp.isfinite(output))