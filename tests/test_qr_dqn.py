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


from qr_dqn.replay import ReplayBuffer


def test_replay_buffer_init():
    buf = ReplayBuffer(capacity=100, obs_shape=(84, 84, 4))
    assert buf.size == 0


def test_replay_buffer_add():
    buf = ReplayBuffer(capacity=100, obs_shape=(84, 84, 4))
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    next_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    buf.add(obs, 1, 1.0, next_obs, False)
    assert buf.size == 1


def test_replay_buffer_wrap():
    buf = ReplayBuffer(capacity=5, obs_shape=(84, 84, 4))
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    next_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    for i in range(10):
        buf.add(obs, i % 4, float(i), next_obs, i == 9)
    assert buf.size == 5


def test_replay_buffer_sample(rng):
    buf = ReplayBuffer(capacity=100, obs_shape=(84, 84, 4))
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    next_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    for i in range(50):
        buf.add(obs, i % 4, float(i), next_obs, False)
    batch = buf.sample(rng, batch_size=8)
    assert batch["obs"].shape == (8, 84, 84, 4)
    assert batch["actions"].shape == (8,)
    assert batch["rewards"].shape == (8,)
    assert batch["next_obs"].shape == (8, 84, 84, 4)
    assert batch["dones"].shape == (8,)