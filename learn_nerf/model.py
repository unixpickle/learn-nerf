import flax.linen as nn
import jax.numpy as jnp


class NeRFModel(nn.Module):
    """
    A model architecture based directly on Mildenhall et al. (2020).
    """

    input_layers: int = 5
    mid_layers: int = 4
    hidden_dim: int = 256
    color_layer_dim: int = 128
    x_freqs: int = 10
    d_freqs: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
        x_emb = sinusoidal_emb(x, self.x_freqs)
        d_emb = sinusoidal_emb(d, self.d_freqs)

        z = x_emb
        for _ in range(self.input_layers):
            z = nn.relu(nn.Dense(self.hidden_dim)(z))
        z = jnp.concatenate([z, x_emb], axis=-1)
        for i in range(self.mid_layers):
            if i > 0:
                z = nn.relu(z)
            z = nn.Dense(self.hidden_dim)(z)
        density = jnp.log(1 + jnp.exp(nn.Dense(1)(z)))
        z = jnp.concatenate([z, d_emb], axis=-1)
        z = nn.relu(nn.Dense(self.color_layer_dim)(z))
        rgb = nn.tanh(nn.Dense(3)(z))

        return density, rgb


def sinusoidal_emb(coords: jnp.ndarray, freqs: int) -> jnp.ndarray:
    """
    Compute sinusoidal embeddings for some input coordinates.

    :param coords: an [N x D] array of coordinates.
    :return: an [N x D*freqs*2] array of embeddings.
    """
    coeffs = 2 ** jnp.arange(freqs, dtype=jnp.float32)
    inputs = coords[..., None] * coeffs
    sines = jnp.sin(inputs)
    cosines = jnp.cos(inputs)
    combined = jnp.concatenate([sines, cosines], axis=-1)
    return combined.reshape(combined.shape[:-2] + (-1,))
