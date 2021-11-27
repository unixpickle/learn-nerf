import jax
import jax.numpy as jnp


def stratified_sampling(
    key: jax.random.PRNGKey,
    count: int,
    batch_size: int,
    min: jnp.ndarray,
    max: jnp.ndarray,
) -> jnp.ndarray:
    """
    :param key: RNG key for sampling.
    :param count: number of samples per batch element.
    :param batch_size: number of batch elements.
    :param min: a scalar array containing the minimum value.
    :param max: a scalar array containing the maximum value.
    :return: a [batch_size x count] batch of t values, where
             min <= t <= max and t_i < t_{i+1} for all i for
             each batch element.
    """
    bin_size = (max - min) / count
    midpoints = jnp.arange(0, count, dtype=jnp.float32) * bin_size + min
    randoms = jax.random.uniform(key, (batch_size, count), maxval=bin_size)
    return randoms + midpoints


def render_ray(
    ts: jnp.ndarray, t_min: jnp.ndarray, densities: jnp.ndarray, rgbs: jnp.ndarray
) -> jnp.ndarray:
    """
    Perform volumetric rendering given density and color samples along a ray.

    :param ts: an [N x T] batch of ts.
    :param t_min: a scalar array containing the minimum t value.
    :param densities: an [N x T] batch of non-negative density outputs.
    :param rgbs: an [N x T x 3] batch of RGB values.
    :return: an [N x 3] batch of RGB values.
    """
    # Compute density*dt (the integrand)
    delta = ts - jnp.concatenate(
        [jnp.tile(t_min.reshape([1, 1]), [ts.shape[0], 1]), ts[:, :-1]], axis=1
    )
    density_dt = delta * densities

    # Compute the integral of termination probability over
    # time, to get the probability we make it to time t.
    acc_densities_cur = jnp.cumsum(density_dt, axis=1)
    acc_densities_prev = jnp.concatenate(
        [jnp.zeros_like(acc_densities_cur[:, :1]), acc_densities_cur[:, :-1]], axis=1
    )
    prob_survive = jnp.exp(-acc_densities_prev)

    # Compute the probability of terminating at time t.
    prob_terminate = 1 - jnp.exp(-density_dt)

    return jnp.sum((prob_survive * prob_terminate)[:, :, None] * rgbs, axis=1)
