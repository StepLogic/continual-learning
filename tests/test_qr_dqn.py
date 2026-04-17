import pytest
import jax
import jax.numpy as jnp
import numpy as np
from qr_dqn.configs import QRDQNConfig
from qr_dqn.network import QuantileNetwork
from qr_dqn.atari_wrapper import make_atari_env


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
from qr_dqn.losses import quantile_huber_loss


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


def test_quantile_huber_loss_shape():
    batch_size = 8
    num_quantiles = 64
    num_actions = 4
    current_quantiles = jnp.ones((batch_size, num_actions, num_quantiles))
    target_quantiles = jnp.ones((batch_size, num_actions, num_quantiles))
    actions = jnp.zeros(batch_size, dtype=jnp.int32)
    taus = (2 * jnp.arange(num_quantiles) + 1) / (2 * num_quantiles)
    loss = quantile_huber_loss(current_quantiles, target_quantiles, actions, taus, kappa=1.0)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_quantile_huber_loss_zero_when_equal():
    batch_size = 4
    num_quantiles = 32
    num_actions = 2
    quantiles = jnp.ones((batch_size, num_actions, num_quantiles))
    actions = jnp.zeros(batch_size, dtype=jnp.int32)
    taus = (2 * jnp.arange(num_quantiles) + 1) / (2 * num_quantiles)
    loss = quantile_huber_loss(quantiles, quantiles, actions, taus, kappa=1.0)
    assert loss == 0.0


def test_quantile_huber_loss_positive_when_different():
    batch_size = 4
    num_quantiles = 32
    num_actions = 2
    current = jnp.zeros((batch_size, num_actions, num_quantiles))
    target = jnp.ones((batch_size, num_actions, num_quantiles))
    actions = jnp.zeros(batch_size, dtype=jnp.int32)
    taus = (2 * jnp.arange(num_quantiles) + 1) / (2 * num_quantiles)
    loss = quantile_huber_loss(current, target, actions, taus, kappa=1.0)
    assert loss > 0.0


from qr_dqn.agent import QRDQNAgent


@pytest.fixture
def agent(rng):
    config = QRDQNConfig(num_quantiles=32, replay_capacity=500)
    return QRDQNAgent(config, num_actions=6, obs_shape=(84, 84, 4), rng=rng)


def test_agent_init(agent):
    assert agent.config.num_quantiles == 32


def test_agent_act_greedy(agent):
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    action = agent.act(obs, epsilon=0.0)
    assert 0 <= action < 6


def test_agent_act_explore(agent):
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    actions = [agent.act(obs, epsilon=1.0) for _ in range(20)]
    assert len(set(actions)) > 1


def test_agent_get_quantiles(agent):
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    quantiles = agent.get_quantiles(obs)
    assert quantiles.shape == (6, 32)


def test_agent_copy_params(agent):
    copied = agent.copy_params()
    assert copied is not agent.params


def test_agent_distill_from(rng):
    config = QRDQNConfig(num_quantiles=32, replay_capacity=500)
    agent1 = QRDQNAgent(config, num_actions=6, obs_shape=(84, 84, 4), rng=rng)
    key2 = jax.random.PRNGKey(99)
    agent2 = QRDQNAgent(config, num_actions=6, obs_shape=(84, 84, 4), rng=key2)
    agent1.distill_from(agent2)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b), agent1.params, agent2.params)
    )


def test_agent_train_step(agent):
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    for i in range(64):
        agent.buffer_add(obs, i % 6, 1.0, obs, False)
    rng_key = jax.random.PRNGKey(0)
    metrics = agent.train_step(rng_key)
    assert "loss" in metrics
    assert jnp.isfinite(metrics["loss"])


def test_make_atari_env_obs_shape():
    env = make_atari_env("PongNoFrameskip-v4")
    obs, _ = env.reset()
    assert obs.shape == (84, 84, 4)
    assert obs.dtype == np.uint8
    env.close()


def test_make_atari_env_step():
    env = make_atari_env("PongNoFrameskip-v4")
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (84, 84, 4)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    env.close()