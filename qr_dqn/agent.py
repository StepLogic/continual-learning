import jax
import jax.numpy as jnp
import optax
import numpy as np
from qr_dqn.configs import QRDQNConfig
from qr_dqn.network import QuantileNetwork
from qr_dqn.replay import ReplayBuffer, PrioritizedReplayBuffer
from qr_dqn.losses import quantile_huber_loss


class QRDQNAgent:
    def __init__(self, config: QRDQNConfig, num_actions: int, obs_shape: tuple, rng: jax.Array):
        self.config = config
        self.num_actions = num_actions
        self.network = QuantileNetwork(
            num_actions=num_actions,
            num_quantiles=config.num_quantiles,
            hidden_dim=config.hidden_dim,
            dueling=config.dueling,
        )

        # Optimizer with gradient clipping
        optimizer_parts = []
        if config.max_grad_norm > 0:
            optimizer_parts.append(optax.clip_by_global_norm(config.max_grad_norm))
        optimizer_parts.append(optax.adam(config.learning_rate, eps=config.adam_eps))
        self.optimizer = optax.chain(*optimizer_parts)

        # Replay buffer: PER or uniform
        if config.per_alpha > 0:
            self.buffer = PrioritizedReplayBuffer(
                capacity=config.replay_capacity,
                obs_shape=obs_shape,
                alpha=config.per_alpha,
                epsilon=config.per_epsilon,
            )
        else:
            self.buffer = ReplayBuffer(
                capacity=config.replay_capacity,
                obs_shape=obs_shape,
            )

        self.rng = rng
        self.rng, init_rng = jax.random.split(rng)

        dummy_obs = jnp.ones((1, *obs_shape), dtype=jnp.float32)
        self.taus = (2 * jnp.arange(config.num_quantiles) + 1) / (2 * config.num_quantiles)
        self.params = self.network.init(init_rng, dummy_obs)
        self.target_params = jax.tree_util.tree_map(jnp.copy, self.params)
        self.opt_state = self.optimizer.init(self.params)
        self.step_count = 0

        self._jit_train = jax.jit(self._train_step_fn)

    def buffer_add(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def act(self, obs, epsilon: float = 0.0) -> int:
        self.rng, act_rng, explore_rng = jax.random.split(self.rng, 3)
        if jax.random.uniform(explore_rng) < epsilon:
            return int(jax.random.randint(act_rng, (), 0, self.num_actions))
        quantiles = self.network.apply(
            self.params, obs[jnp.newaxis].astype(jnp.float32),
        )
        mean_q = quantiles.mean(axis=-1).squeeze()
        return int(jnp.argmax(mean_q))

    def get_quantiles(self, obs) -> jnp.ndarray:
        return self.network.apply(
            self.params, obs[jnp.newaxis].astype(jnp.float32),
        ).squeeze(0)

    def copy_params(self):
        return jax.tree_util.tree_map(jnp.copy, self.params)

    def reset_params(self, rng):
        self.rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.ones((1, *self.buffer.obs_shape), dtype=jnp.float32)
        self.params = self.network.init(init_rng, dummy_obs)
        self.opt_state = self.optimizer.init(self.params)
        return self.params

    def distill_from(self, other: "QRDQNAgent"):
        self.params = jax.tree_util.tree_map(jnp.copy, other.params)
        self.target_params = jax.tree_util.tree_map(jnp.copy, other.target_params)
        self.opt_state = self.optimizer.init(self.params)

    def update_target(self):
        tau = self.config.target_update_tau
        if tau >= 1.0:
            self.target_params = jax.tree_util.tree_map(jnp.copy, self.params)
        else:
            self.target_params = jax.tree_util.tree_map(
                lambda online, target: tau * online + (1 - tau) * target,
                self.params, self.target_params,
            )

    def _compute_beta(self, frame: int) -> float:
        if self.config.per_alpha <= 0:
            return 1.0
        beta_frames = self.config.per_beta_frames
        if beta_frames is None or beta_frames <= 0:
            return 1.0
        fraction = min(frame / beta_frames, 1.0)
        return self.config.per_beta_start + fraction * (1.0 - self.config.per_beta_start)

    def _train_step_fn(self, params, target_params, opt_state, batch, taus, gamma, kappa):
        weights = batch.get("weights", jnp.ones(batch["rewards"].shape[0]))

        def loss_fn(params):
            current_quantiles = self.network.apply(params, batch["obs"])
            next_quantiles_online = self.network.apply(params, batch["next_obs"])
            next_actions = next_quantiles_online.mean(axis=-1).argmax(axis=-1)
            target_quantiles = self.network.apply(target_params, batch["next_obs"])
            batch_idx = jnp.arange(batch["rewards"].shape[0])
            target_values = target_quantiles[batch_idx, next_actions]
            rewards = batch["rewards"][:, None]
            dones = batch["dones"][:, None].astype(jnp.float32)
            bellman_targets = rewards + gamma * (1 - dones) * target_values

            curr = current_quantiles[batch_idx, batch["actions"]]

            # Quantile Huber loss
            delta = bellman_targets[:, None, :] - curr[:, :, None]
            taus_row = taus[None, :, None]
            indicator = (delta < 0).astype(jnp.float32)
            asymmetric_weight = jnp.abs(taus_row - indicator)
            abs_delta = jnp.abs(delta)
            quadratic = jnp.minimum(abs_delta, kappa) ** 2 * 0.5
            linear = kappa * (abs_delta - 0.5 * kappa)
            huber = jnp.where(abs_delta <= kappa, quadratic, linear)
            per_sample_loss = (asymmetric_weight * huber).sum(axis=2).mean(axis=1)

            # Apply importance sampling weights
            loss = (weights * per_sample_loss).mean()

            # TD errors for PER priority updates
            td_errors = jnp.abs(bellman_targets.mean(axis=-1) - curr.mean(axis=-1))

            return loss, (current_quantiles, td_errors)

        (loss, (quantiles, td_errors)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, td_errors

    def train_step(self, rng_key=None):
        if rng_key is None:
            self.rng, rng_key = jax.random.split(self.rng)

        if isinstance(self.buffer, PrioritizedReplayBuffer):
            beta = self._compute_beta(self.step_count)
            batch = self.buffer.sample(rng_key, self.config.batch_size, beta=beta)
        else:
            batch = self.buffer.sample(rng_key, self.config.batch_size)
            batch["weights"] = jnp.ones(self.config.batch_size)
            batch["indices"] = np.arange(self.config.batch_size)

        self.params, self.opt_state, loss, td_errors = self._jit_train(
            self.params, self.target_params, self.opt_state,
            batch, self.taus, self.config.gamma, self.config.kappa,
        )
        self.step_count += 1

        # Update PER priorities
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            self.buffer.update_priorities(
                np.asarray(batch["indices"]), np.asarray(td_errors),
            )

        # Target update: hard copy every target_update_freq, soft update every step
        if self.config.target_update_tau < 1.0:
            self.update_target()
        elif self.step_count % self.config.target_update_freq == 0:
            self.update_target()

        return {"loss": float(loss), "td_errors": td_errors}