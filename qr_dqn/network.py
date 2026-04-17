import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


class QuantileNetwork(nn.Module):
    num_actions: int
    num_quantiles: int
    hidden_dim: int = 512

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
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions * self.num_quantiles)(x)
        x = x.reshape((-1, self.num_actions, self.num_quantiles))
        return x