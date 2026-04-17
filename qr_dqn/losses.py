import jax
import jax.numpy as jnp


def quantile_huber_loss(
    current_quantiles: jnp.ndarray,
    target_quantiles: jnp.ndarray,
    actions: jnp.ndarray,
    taus: jnp.ndarray,
    kappa: float = 1.0,
) -> jnp.ndarray:
    batch_size = current_quantiles.shape[0]
    num_quantiles = current_quantiles.shape[-1]

    # Select quantiles for taken actions: (batch, N)
    action_indices = jnp.arange(batch_size)
    curr = current_quantiles[action_indices, actions]  # (batch, N)
    tgt = target_quantiles[action_indices, actions]     # (batch, N)

    # Pairwise differences: (batch, N, N)
    delta = tgt[:, None, :] - curr[:, :, None]

    # Asymmetric indicator: |tau_i - I(delta < 0)| — (batch, N, N)
    taus_row = taus[None, :, None]
    indicator = (delta < 0).astype(jnp.float32)
    asymmetric_weight = jnp.abs(taus_row - indicator)

    # Huber loss
    abs_delta = jnp.abs(delta)
    quadratic = jnp.minimum(abs_delta, kappa) ** 2 * 0.5
    linear = kappa * (abs_delta - 0.5 * kappa)
    huber = jnp.where(abs_delta <= kappa, quadratic, linear)

    # Weighted loss: (batch, N, N) → scalar
    loss = asymmetric_weight * huber
    loss = loss.sum(axis=2).mean(axis=1).mean()

    return loss