from typing import Tuple

import jax
import jax.numpy as jnp


def stratified_sampling(
    batch_size: int,
    count: int,
    key: jax.random.PRNGKey,
    t_min: jnp.ndarray,
    t_max: jnp.ndarray,
) -> jnp.ndarray:
    """
    :param batch_size: number of batch elements.
    :param count: number of samples per batch element.
    :param key: RNG key for sampling.
    :param t_min: a scalar array containing the minimum value.
    :param t_max: a scalar array containing the maximum value.
    :return: a [batch_size x count] batch of t values, where
             min <= t <= max and t_i < t_{i+1} for all i for
             each batch element.
    """
    bin_size = (t_max - t_min) / count
    midpoints = jnp.arange(0, count, dtype=jnp.float32) * bin_size + t_min
    randoms = jax.random.uniform(key, (batch_size, count), maxval=bin_size)
    return randoms + midpoints


def render_rays(
    ts: jnp.ndarray,
    t_min: jnp.ndarray,
    densities: jnp.ndarray,
    rgbs: jnp.ndarray,
    background: jnp.ndarray,
) -> jnp.ndarray:
    """
    Perform volumetric rendering given density and color samples along a batch
    of rays.

    :param ts: an [N x T] batch of ts.
    :param t_min: a scalar array containing the minimum t value.
    :param densities: an [N x T] batch of non-negative density outputs.
    :param rgbs: an [N x T x 3] batch of RGB values.
    :param background: an RGB background color, of shape [3].
    :return: an [N x 3] batch of RGB values.
    """
    w, bg_prob = _termination_probabilities(ts, t_min, densities)
    return jnp.sum(w[:, :, None] * rgbs, axis=1) + bg_prob * background


def fine_sampling(
    count: int,
    key: jax.random.PRNGKey,
    ts: jnp.ndarray,
    t_min: jnp.ndarray,
    t_max: jnp.ndarray,
    densities: jnp.ndarray,
    combine: bool = True,
) -> jnp.ndarray:
    """
    Sample points along a ray leveraging density information from a coarsely
    sampled ray.

    :param count: the number of points to sample.
    :param key: the RNG key to use for sampling.
    :param ts: the sampled ts from the coarse ray.
    :param t_min: the minimum t value to sample.
    :param t_max: the maximum t value to sample.
    :param densities: the sampled non-negative densities for ts.
    :param combine: if True, combine the new sampled points with the old
                    sampled points in one sorted array.
    :return: an [N x T'] array of sampled ts, similar to stratified_sampling().
    """
    w, bg_prob = _termination_probabilities(ts, t_min, densities)
    w = jnp.concatenate([w, bg_prob], axis=1)

    # Setup an inverse CDF for inverse transform sampling.
    xs = jnp.cumsum(w, axis=1)
    xs = jnp.concatenate([jnp.zeros_like(xs[:, :1]), xs], axis=1)
    xs = xs / xs[:, -1:]  # normalize
    ys = jnp.concatenate(
        [jnp.zeros_like(ts[:, :1]) + t_min, ts, jnp.zeros_like(ts[:, :1]) + t_max],
        axis=1,
    )

    # Evaluate the inverse CDF at quasi-random points.
    input_samples = stratified_sampling(
        batch_size=ts.shape[0],
        count=count,
        key=key,
        t_min=jnp.array(0.0),
        t_max=jnp.array(1.0),
    )
    new_ts = jax.vmap(jnp.interp)(input_samples, xs, ys)

    if combine:
        combined = jnp.concatenate([ts, new_ts], axis=1)
        return jnp.sort(combined, axis=1)
    else:
        return new_ts


def _termination_probabilities(
    ts: jnp.ndarray, t_min: jnp.ndarray, densities: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Compute density*dt (the integrand)
    delta = ts - jnp.concatenate(
        [jnp.zeros_like(ts[:, :1]) + t_min, ts[:, :-1]], axis=1
    )
    density_dt = delta * densities

    # Compute the integral of termination probability over
    # time, to get the probability we make it to time t.
    acc_densities_cur = jnp.cumsum(density_dt, axis=1)
    acc_densities_prev = jnp.concatenate(
        [jnp.zeros_like(acc_densities_cur[:, :1]), acc_densities_cur[:, :-1]], axis=1
    )
    prob_survive = jnp.exp(-acc_densities_prev)

    # Compute the probability of terminating at time t given
    # that we made it to time t.
    prob_terminate = 1 - jnp.exp(-density_dt)

    return prob_survive * prob_terminate, jnp.exp(-acc_densities_cur[:, -1:])
