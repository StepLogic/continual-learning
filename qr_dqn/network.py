import jax.numpy as jnp
import flax.linen as nn


class QuantileNetwork(nn.Module):
    num_actions: int
    num_quantiles: int
    hidden_dim: int = 256
    dueling: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        if self.dueling:
            # Value stream
            val = nn.Dense(self.hidden_dim)(x)
            val = nn.relu(val)
            val = nn.Dense(self.num_quantiles)(val)  # (batch, N)

            # Advantage stream
            adv = nn.Dense(self.hidden_dim)(x)
            adv = nn.relu(adv)
            adv = nn.Dense(self.num_actions * self.num_quantiles)(adv)
            adv = adv.reshape((-1, self.num_actions, self.num_quantiles))  # (batch, A, N)

            # Q = V + A - mean(A)
            q = val[:, None, :] + adv - adv.mean(axis=1, keepdims=True)  # (batch, A, N)
            return q
        else:
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(self.num_actions * self.num_quantiles)(x)
            x = x.reshape((-1, self.num_actions, self.num_quantiles))
            return x