import jax.numpy as jnp


def quantile_huber_loss(
    current_quantiles: jnp.ndarray,
    target_quantiles: jnp.ndarray,
    taus: jnp.ndarray,
    kappa: float = 1.0,
) -> jnp.ndarray:
    """Quantile Huber loss between two sets of action-indexed quantile values.

    Args:
        current_quantiles: (batch, N) — quantile values for the taken action.
        target_quantiles: (batch, N) — Bellman target quantile values (already
            action-selected via Double DQN: online network selects, target
            network evaluates).
        taus: (N,) — quantile fractions.
        kappa: Huber loss threshold.

    Returns:
        Scalar loss value.
    """
    delta = target_quantiles[:, None, :] - current_quantiles[:, :, None]

    taus_row = taus[None, :, None]
    indicator = (delta < 0).astype(jnp.float32)
    asymmetric_weight = jnp.abs(taus_row - indicator)

    abs_delta = jnp.abs(delta)
    quadratic = jnp.minimum(abs_delta, kappa) ** 2 * 0.5
    linear = kappa * (abs_delta - 0.5 * kappa)
    huber = jnp.where(abs_delta <= kappa, quadratic, linear)

    loss = asymmetric_weight * huber
    loss = loss.sum(axis=2).mean(axis=1).mean()

    return loss