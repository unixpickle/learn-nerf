from dataclasses import dataclass
from typing import Union

import jax
import jax.numpy as jnp


@dataclass
class RaySamples:
    t_min: jnp.ndarray
    t_max: jnp.ndarray
    ts: jnp.ndarray

    @classmethod
    def stratified_sampling(
        cls,
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
        return cls(t_min=t_min, t_max=t_max, ts=randoms + midpoints)

    def render_rays(
        self,
        densities: jnp.ndarray,
        rgbs: jnp.ndarray,
        background: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Perform volumetric rendering given density and color samples along a batch
        of rays.

        :param densities: an [N x T] batch of non-negative density outputs.
        :param rgbs: an [N x T x 3] batch of RGB values.
        :param background: an RGB background color, of shape [3].
        :return: an [N x 3] batch of RGB values.
        """
        probs = self.termination_probs(densities)
        colors = jnp.concatenate(
            [rgbs, jnp.tile(background[None], [rgbs.shape[0], 1])], axis=1
        )
        return jnp.sum(probs[..., None] * colors, axis=1)

    def fine_sampling(
        self,
        count: int,
        key: jax.random.PRNGKey,
        densities: jnp.ndarray,
        combine: bool = True,
    ) -> "RaySamples":
        """
        Sample points along a ray leveraging density information from a
        coarsely sampled ray (stored in self).

        :param count: the number of points to sample.
        :param key: the RNG key to use for sampling.
        :param densities: the sampled non-negative densities for ts.
        :param combine: if True, combine the new sampled points with the old
                        sampled points in one sorted array.
        :return: an [N x T'] array of sampled ts, similar to stratified_sampling().
        """
        w = self.termination_probs(densities)[:, :-1]

        # Setup an inverse CDF for inverse transform sampling.
        xs = jnp.cumsum(w, axis=1)
        xs = jnp.concatenate([self._const_vec(0.0), xs], axis=1)
        xs = xs / xs[:, -1:]  # normalize
        ys = jnp.concatenate(
            [self._const_vec(self.t_min), self.ends()],
            axis=1,
        )

        print("xs", xs)
        print("ys", ys)

        # Evaluate the inverse CDF at quasi-random points.
        input_samples = self.stratified_sampling(
            batch_size=self.ts.shape[0],
            count=count,
            key=key,
            t_min=jnp.array(0.0),
            t_max=jnp.array(1.0),
        )
        print(input_samples.ts)
        new_ts = jax.vmap(jnp.interp)(input_samples.ts, xs, ys)

        if combine:
            combined = jnp.concatenate([self.ts, new_ts], axis=1)
            new_ts = jnp.sort(combined, axis=1)

        return RaySamples(t_min=self.t_min, t_max=self.t_max, ts=new_ts)

    def starts(self) -> jnp.ndarray:
        t_mid = (self.ts[:, 1:] + self.ts[:, :-1]) / 2
        return jnp.concatenate([self._const_vec(self.t_min), t_mid], axis=1)

    def ends(self) -> jnp.ndarray:
        t_mid = (self.ts[:, 1:] + self.ts[:, :-1]) / 2
        return jnp.concatenate([t_mid, self._const_vec(self.t_max)], axis=1)

    def deltas(self) -> jnp.ndarray:
        return self.ends() - self.starts()

    def termination_probs(self, densities: jnp.ndarray):
        density_dt = densities * self.deltas()

        # Compute the integral of termination probability over
        # time, to get the probability we make it to time t.
        acc_densities_cur = jnp.cumsum(density_dt, axis=1)
        acc_densities_prev = jnp.concatenate(
            [jnp.zeros_like(acc_densities_cur[:, :1]), acc_densities_cur], axis=1
        )
        prob_survive = jnp.exp(-acc_densities_prev)

        # Compute the probability of terminating at time t given
        # that we made it to time t.
        prob_terminate = jnp.concatenate(
            [1 - jnp.exp(-density_dt), self._const_vec(1.0)], axis=1
        )

        return prob_survive * prob_terminate

    def _const_vec(self, x: Union[jnp.ndarray, float]) -> jnp.ndarray:
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(x)
        return jnp.tile(x.reshape([1, 1]), [self.ts.shape[0], 1])
