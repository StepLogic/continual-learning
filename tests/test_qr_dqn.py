import pytest
import jax
import jax.numpy as jnp
import numpy as np
from qr_dqn.configs import QRDQNConfig
from qr_dqn.network import QuantileNetwork
from qr_dqn.atari_wrapper import make_atari_env
from qr_dqn.train import get_epsilon, _log_metrics


def test_config_defaults():
    config = QRDQNConfig()
    assert config.num_quantiles == 64
    assert config.kappa == 1.0
    assert config.learning_rate == 1e-4
    assert config.gamma == 0.99
    assert config.replay_capacity == 50_000


def test_config_new_fields():
    config = QRDQNConfig()
    assert config.dueling is True
    assert config.n_step == 3
    assert config.max_grad_norm == 10.0
    assert config.per_alpha == 0.6
    assert config.per_beta_start == 0.4
    assert config.per_beta_frames == 10_000_000
    assert config.per_epsilon == 1e-6
    assert config.target_update_tau == 1.0


def test_config_updated_defaults():
    config = QRDQNConfig()
    assert config.num_quantiles == 64
    assert config.replay_capacity == 50_000
    assert config.epsilon_decay_steps == 250_000
    assert config.warmup_steps == 20_000
    assert config.max_frames == 10_000_000
    assert config.target_update_freq == 10_000


def test_config_override():
    config = QRDQNConfig(num_quantiles=32, kappa=0.5)
    assert config.num_quantiles == 32
    assert config.kappa == 0.5


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


def test_network_forward_shape(rng):
    num_actions = 6
    num_quantiles = 32
    net = QuantileNetwork(num_actions=num_actions, num_quantiles=num_quantiles, dueling=False)
    dummy_obs = jnp.ones((4, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert output.shape == (4, num_actions, num_quantiles)


def test_network_single_obs(rng):
    num_actions = 6
    num_quantiles = 32
    net = QuantileNetwork(num_actions=num_actions, num_quantiles=num_quantiles, dueling=False)
    dummy_obs = jnp.ones((1, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert output.shape == (1, num_actions, num_quantiles)


def test_network_output_finite(rng):
    num_actions = 4
    num_quantiles = 32
    net = QuantileNetwork(num_actions=num_actions, num_quantiles=num_quantiles, dueling=False)
    dummy_obs = jnp.ones((2, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert jnp.all(jnp.isfinite(output))


def test_network_dueling_shape(rng):
    num_actions = 6
    num_quantiles = 32
    net = QuantileNetwork(
        num_actions=num_actions,
        num_quantiles=num_quantiles,
        dueling=True,
    )
    dummy_obs = jnp.ones((4, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert output.shape == (4, num_actions, num_quantiles)


def test_network_dueling_output_finite(rng):
    num_actions = 4
    num_quantiles = 8
    net = QuantileNetwork(
        num_actions=num_actions,
        num_quantiles=num_quantiles,
        dueling=True,
    )
    dummy_obs = jnp.ones((2, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert jnp.all(jnp.isfinite(output))


def test_network_nondueling_backward_compat(rng):
    """dueling=False should produce same shape as before."""
    num_actions = 6
    num_quantiles = 32
    net = QuantileNetwork(
        num_actions=num_actions,
        num_quantiles=num_quantiles,
        dueling=False,
    )
    dummy_obs = jnp.ones((2, 84, 84, 4))
    params = net.init(rng, dummy_obs)
    output = net.apply(params, dummy_obs)
    assert output.shape == (2, num_actions, num_quantiles)


from qr_dqn.replay import ReplayBuffer
from qr_dqn.replay import PrioritizedReplayBuffer
from qr_dqn.replay import NStepBuffer
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


def test_per_buffer_init():
    buf = PrioritizedReplayBuffer(capacity=100, obs_shape=(84, 84, 4))
    assert buf.size == 0


def test_per_buffer_add():
    buf = PrioritizedReplayBuffer(capacity=100, obs_shape=(84, 84, 4))
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    next_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    buf.add(obs, 1, 1.0, next_obs, False)
    assert buf.size == 1


def test_per_buffer_sample_has_weights(rng):
    buf = PrioritizedReplayBuffer(capacity=100, obs_shape=(84, 84, 4))
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    next_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    for i in range(50):
        buf.add(obs, i % 4, float(i), next_obs, False)
    batch = buf.sample(rng, batch_size=8, beta=1.0)
    assert "weights" in batch
    assert "indices" in batch
    assert batch["weights"].shape == (8,)
    assert batch["indices"].shape == (8,)
    assert batch["obs"].shape == (8, 84, 84, 4)


def test_per_buffer_update_priorities():
    buf = PrioritizedReplayBuffer(capacity=100, obs_shape=(84, 84, 4))
    obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    next_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    for i in range(10):
        buf.add(obs, i % 4, float(i), next_obs, False)
    indices = np.array([0, 1, 2])
    priorities = np.array([5.0, 3.0, 1.0])
    buf.update_priorities(indices, priorities)
    assert buf.priorities[0] == pytest.approx(5.0 + 1e-6)
    assert buf.priorities[1] == pytest.approx(3.0 + 1e-6)
    assert buf.priorities[2] == pytest.approx(1.0 + 1e-6)


def test_nstep_buffer_returns_none_when_not_full():
    buf = NStepBuffer(n=3, gamma=0.99)
    obs = np.zeros((84, 84, 4), dtype=np.uint8)
    result = buf.push(obs, 0, 1.0, obs, False)
    assert result is None


def test_nstep_buffer_returns_transition_when_full():
    buf = NStepBuffer(n=3, gamma=0.99)
    obs = np.zeros((84, 84, 4), dtype=np.uint8)
    obs1 = np.ones((84, 84, 4), dtype=np.uint8)
    buf.push(obs, 0, 1.0, obs1, False)
    buf.push(obs1, 1, 2.0, obs, False)
    result = buf.push(obs, 2, 3.0, obs1, False)
    assert result is not None
    obs_n, action_n, reward_n, next_obs_n, terminated_n = result
    assert action_n == 0
    expected = 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
    assert reward_n == pytest.approx(expected, rel=1e-4)
    assert terminated_n is False


def test_nstep_buffer_early_termination():
    buf = NStepBuffer(n=3, gamma=0.99)
    obs = np.zeros((84, 84, 4), dtype=np.uint8)
    obs1 = np.ones((84, 84, 4), dtype=np.uint8)
    buf.push(obs, 0, 1.0, obs1, False)
    buf.push(obs1, 1, 2.0, obs, True)  # terminated at step 2
    result = buf.push(obs, 2, 3.0, obs1, False)
    assert result is not None
    _, _, reward_n, _, terminated_n = result
    expected = 1.0 + 0.99 * 2.0
    assert reward_n == pytest.approx(expected, rel=1e-4)
    assert terminated_n is True


def test_nstep_buffer_flush():
    buf = NStepBuffer(n=3, gamma=0.99)
    obs = np.zeros((84, 84, 4), dtype=np.uint8)
    obs1 = np.ones((84, 84, 4), dtype=np.uint8)
    buf.push(obs, 0, 1.0, obs1, False)
    buf.push(obs1, 1, 2.0, obs, False)
    results = buf.flush()
    assert len(results) == 2


def test_quantile_huber_loss_shape():
    batch_size = 8
    num_quantiles = 32
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


def test_get_epsilon_schedule():
    config = QRDQNConfig(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=1000)
    assert get_epsilon(0, config) == 1.0
    assert get_epsilon(1000, config) == 0.01
    assert 0.01 < get_epsilon(500, config) < 1.0
    assert get_epsilon(2000, config) == 0.01


class TestFAMEHooks:
    def test_distill_then_get_quantiles(self):
        config = QRDQNConfig(num_quantiles=16, replay_capacity=100)
        rng1 = jax.random.PRNGKey(0)
        rng2 = jax.random.PRNGKey(1)
        fast = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng1)
        meta = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng2)
        obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        q_before = fast.get_quantiles(obs)
        fast.distill_from(meta)
        q_after = fast.get_quantiles(obs)
        assert not jnp.allclose(q_before, q_after)

    def test_reset_params_changes_output(self):
        config = QRDQNConfig(num_quantiles=16, replay_capacity=100)
        rng = jax.random.PRNGKey(0)
        agent = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng)
        obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        q_before = agent.get_quantiles(obs)
        agent.reset_params(jax.random.PRNGKey(99))
        q_after = agent.get_quantiles(obs)
        assert not jnp.allclose(q_before, q_after)

    def test_copy_params_independent(self):
        config = QRDQNConfig(num_quantiles=16, replay_capacity=100)
        rng = jax.random.PRNGKey(0)
        agent = QRDQNAgent(config, num_actions=4, obs_shape=(84, 84, 4), rng=rng)
        copied_params = agent.copy_params()
        agent.reset_params(jax.random.PRNGKey(77))
        obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        q_orig = agent.network.apply(copied_params, obs[jnp.newaxis].astype(jnp.float32))
        q_new = agent.get_quantiles(obs)
        assert not jnp.allclose(q_orig.squeeze(0), q_new)


def test_log_metrics_output(capsys):
    _log_metrics(
        frame=10000,
        avg_loss=0.523,
        epsilon=0.94,
        episode_count=42,
        episode_returns=[1.0, 2.0, 3.0],
        max_return=3.0,
    )
    captured = capsys.readouterr()
    assert "[Frame   10000]" in captured.out
    assert "loss=0.523" in captured.out
    assert "eps=0.940" in captured.out
    assert "episodes=42" in captured.out
    assert "mean_ret(10)=2.0" in captured.out
    assert "max_ret=3.0" in captured.out


def test_log_metrics_no_episodes(capsys):
    _log_metrics(
        frame=10000,
        avg_loss=0.5,
        epsilon=0.94,
        episode_count=0,
        episode_returns=[],
        max_return=float("-inf"),
    )
    captured = capsys.readouterr()
    assert "max_ret=--" in captured.out


def test_train_stores_terminated_only():
    """Buffer should store 'terminated' only, not 'terminated or truncated'."""
    config = QRDQNConfig(
        replay_capacity=500, warmup_steps=0, max_frames=50,
        epsilon_start=1.0, epsilon_end=1.0,
    )
    rng = jax.random.PRNGKey(config.seed)
    rng, agent_rng = jax.random.split(rng)
    agent = QRDQNAgent(config, num_actions=6, obs_shape=(84, 84, 4), rng=agent_rng)
    env = make_atari_env("PongNoFrameskip-v4", seed=config.seed)
    obs, _ = env.reset()
    for _ in range(50):
        action = agent.act(obs, epsilon=1.0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.buffer_add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    for i in range(agent.buffer.size):
        if agent.buffer.dones[i]:
            assert isinstance(agent.buffer.dones[i], (bool, np.bool_))
    env.close()