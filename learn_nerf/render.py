from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray as KeyArray

from .model import ModelBase


@dataclass
class NeRFRenderer:
    """
    A NeRF hierarchy with corresponding settings for rendering rays.

    :param coarse: the coarse model.
    :param fine: the fine model.
    :param coarse_params: params of the coarse model.
    :param fine_params: params of the fine model.
    :param background: the [3] RGB array background color.
    :param bbox_min: minimum point of the scene bounding box.
    :param bbox_max: maximum point of the scene bounding box.
    :param coarse_ts: samples per ray for coarse model.
    :param fine_ts: additional samples per ray for fine model.
    """

    coarse: ModelBase
    fine: ModelBase
    coarse_params: Any
    fine_params: Any
    background: jnp.ndarray
    bbox_min: jnp.ndarray
    bbox_max: jnp.ndarray
    coarse_ts: int
    fine_ts: int

    min_t_range: float = 1e-3

    def render_rays(
        self,
        key: KeyArray,
        batch: jnp.ndarray,
    ) -> Dict[str, Union[str, jnp.ndarray]]:
        """
        :param key: an RNG key for sampling points along rays.
        :param batch: an [N x 2 x 3] batch of (origin, direction) rays.
        :return: a dict with "fine" and "coarse" keys mapping to [N x 3] arrays of
                 RGB colors. Additionally, there are "fine_aux" and "coarse_aux"
                 keys mapping to dicts of (averaged) auxiliary losses.
        """
        t_min, t_max, mask = self.t_range(batch)

        coarse_key, fine_key = jax.random.split(key)
        # Evaluate the coarse model using regular stratified sampling.
        coarse_ts = RaySamples.stratified_sampling(
            t_min=t_min,
            t_max=t_max,
            mask=mask,
            count=self.coarse_ts,
            key=coarse_key,
        )
        coarse_out, coarse_aux = render_rays(
            model=self.coarse,
            params=self.coarse_params,
            background=self.background,
            batch=batch,
            ts=coarse_ts,
        )

        # Evaluate the fine model using a combined set of points.
        fine_ts = coarse_ts.fine_sampling(
            count=self.fine_ts,
            key=fine_key,
            densities=jax.lax.stop_gradient(coarse_out["densities"]),
        )
        fine_out, fine_aux = render_rays(
            model=self.fine,
            params=self.fine_params,
            background=self.background,
            batch=batch,
            ts=fine_ts,
        )

        return dict(
            coarse=coarse_out["outputs"],
            fine=fine_out["outputs"],
            coarse_aux=coarse_aux,
            fine_aux=fine_aux,
        )

    def t_range(
        self, batch: jnp.ndarray, epsilon: float = 1e-8
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        For a batch of rays, compute the t_min and t_max for each ray
        according to the scene bounding box.

        :param batch: a batch of rays, each of [N x 2 x 3] rays.
        :param epsilon: small offset to add to ray directions to prevent NaNs.
        :return: a tuple (t_min, t_max, mask) of [N] arrays. The mask array is
                 a boolean array, where False means not to render the ray.
        """
        bbox = jnp.stack([self.bbox_min, self.bbox_max])
        bounds, mask = jax.vmap(
            lambda ray: ray_t_range(
                bbox, ray, min_t_range=self.min_t_range, epsilon=epsilon
            )
        )(batch)
        return bounds[:, 0], bounds[:, 1], mask


@dataclass
class RaySamples:
    t_min: jnp.ndarray
    t_max: jnp.ndarray
    mask: jnp.ndarray
    ts: jnp.ndarray

    @classmethod
    def stratified_sampling(
        cls,
        t_min: jnp.ndarray,
        t_max: jnp.ndarray,
        mask: jnp.ndarray,
        count: int,
        key: KeyArray,
    ) -> "RaySamples":
        """
        :param t_min: a batch of minimum values.
        :param t_max: a batch of maximum values.
        :param mask: a batch of masks for each ray.
        :param count: number of samples per batch element.
        :param key: RNG key for sampling.
        :return: samples along the rays.
        """
        bin_size = ((t_max - t_min) / count)[:, None]
        bin_starts = (
            jnp.arange(0, count, dtype=jnp.float32)[None] * bin_size + t_min[:, None]
        )
        randoms = jax.random.uniform(key, bin_starts.shape) * bin_size
        return cls(t_min=t_min, t_max=t_max, mask=mask, ts=bin_starts + randoms)

    def points(self, rays: jnp.ndarray) -> jnp.ndarray:
        """
        For each ray, compute the points at all ts.

        :param rays: a batch of rays of shape [N x 2 x 3] where each ray is a
                     tuple (origin, direction).
        :return: a batch of points of shape [N x T x 3].
        """
        return rays[:, :1] + (rays[:, 1:] * self.ts[:, :, None])

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
            [rgbs, jnp.tile(background[None, None], [rgbs.shape[0], 1, 1])], axis=1
        )
        return jnp.where(
            self.mask[:, None], jnp.sum(probs[..., None] * colors, axis=1), background
        )

    def average_aux_losses(
        self,
        densities: jnp.ndarray,
        aux: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Compute an average of auxiliary losses over each ray, weighted by the
        volume density.

        :param densities: an [N x T] batch of non-negative density outputs.
        :param aux: a dict mapping loss names to [N x T] batches.
        :return: a dict mapping loss names to mean losses.
        """
        probs = self.termination_probs(densities)[:, :-1]
        return {
            k: jnp.sum(jnp.where(self.mask[:, None], jnp.sum(v * probs, axis=-1), 0.0))
            for k, v in aux.items()
        }

    def fine_sampling(
        self,
        count: int,
        key: KeyArray,
        densities: jnp.ndarray,
        combine: bool = True,
        eps: float = 1e-8,
    ) -> "RaySamples":
        """
        Sample points along a ray leveraging density information from a
        coarsely sampled ray (stored in self).

        :param count: the number of points to sample.
        :param key: the RNG key to use for sampling.
        :param densities: the sampled non-negative densities for ts.
        :param combine: if True, combine the new sampled points with the old
                        sampled points in one sorted array.
        :param eps: a small probability to add to termination probs to avoid
                    division by zero.
        :return: an [N x T'] array of sampled ts, similar to stratified_sampling().
        """
        w = self.termination_probs(densities)[:, :-1] + eps

        # Setup an inverse CDF for inverse transform sampling.
        xs = jnp.cumsum(w, axis=1)
        xs = jnp.concatenate([self._const_vec(0.0), xs], axis=1)
        xs = xs / xs[:, -1:]  # normalize
        ys = jnp.concatenate(
            [self.t_min[:, None], self.ends()],
            axis=1,
        )

        # Evaluate the inverse CDF at quasi-random points.
        input_samples = self.stratified_sampling(
            t_min=jnp.zeros_like(self.t_min),
            t_max=jnp.ones_like(self.t_max),
            mask=self.mask,
            count=count,
            key=key,
        )
        new_ts = jax.vmap(jnp.interp)(input_samples.ts, xs, ys)

        if combine:
            combined = jnp.concatenate([self.ts, new_ts], axis=1)
            new_ts = jnp.sort(combined, axis=1)

        return RaySamples(t_min=self.t_min, t_max=self.t_max, mask=self.mask, ts=new_ts)

    def starts(self) -> jnp.ndarray:
        t_mid = (self.ts[:, 1:] + self.ts[:, :-1]) / 2
        return jnp.concatenate([self.t_min[:, None], t_mid], axis=1)

    def ends(self) -> jnp.ndarray:
        t_mid = (self.ts[:, 1:] + self.ts[:, :-1]) / 2
        return jnp.concatenate([t_mid, self.t_max[:, None]], axis=1)

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

    def _const_vec(self, x: float) -> jnp.ndarray:
        return jnp.tile(jnp.array(x).reshape([1, 1]), [self.ts.shape[0], 1])


def render_rays(
    model: ModelBase,
    params: Any,
    background: jnp.ndarray,
    batch: jnp.ndarray,
    ts: "RaySamples",
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    Render a batch of rays using a model.

    :param model: the NeRF model to run.
    :param params: the parameter object for the model.
    :param background: the [3] array of background color.
    :param batch: an [N x 2 x 3] array of (origin, direction) rays.
    :param ts: samples along the rays.
    :return: a tuple (out, aux).
             - out: a dict of results with the following keys:
                    - outputs: an [N x 3] array of RGB colors.
                    - rgbs: an [N x T x 3] array of per-point RGB outputs.
                    - densities: an [N x T] array of model density outputs.
             - aux: a dict mapping loss names to (scalar) means of the losses
                    across all unmasked rays.
    """
    all_points = ts.points(batch)
    direction_batch = jnp.tile(batch[:, 1:2], [1, all_points.shape[1], 1])
    densities, rgbs, aux = model.apply(
        dict(params=params),
        all_points.reshape([-1, 3]),
        direction_batch.reshape([-1, 3]),
    )
    densities = densities.reshape(all_points.shape[:-1])
    rgbs = rgbs.reshape(all_points.shape)
    aux = {k: v.reshape(densities.shape) for k, v in aux.items()}

    outputs = ts.render_rays(densities, rgbs, background)
    aux_mean = ts.average_aux_losses(densities, aux)

    return dict(outputs=outputs, rgbs=rgbs, densities=densities), aux_mean


def ray_t_range(
    bbox: jnp.ndarray,
    ray: jnp.ndarray,
    min_t_range: float = 1e-3,
    epsilon: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    For a single ray, compute the t_min and t_max for it and return a mask
    indicating whether on not the ray intersects the bounding box at all.

    :param bbox: a [2 x 3] array of (bbox_min, bbox_max).
    :param ray: a [2 x 3] array containing ray (origin, direction).
    :param min_t_range: the minimum space between t_min and t_max.
    :param epsilon: small offset to add to ray directions to prevent NaNs.
    :return: a tuple (ts, mask) where ts is of shape [2] storing (t_min, t_max)
             and mask is a boolean scalar.
    """
    origin = ray[0]
    direction = ray[1]

    # Find timesteps of collision on each axis:
    # o+t*d=b
    # t*d=b-o
    # t=(b-o)/d
    offsets = bbox - origin
    ts = offsets / (direction + epsilon)

    # Sort so that the minimum t always comes first.
    ts = jnp.concatenate(
        [
            jnp.min(ts, axis=0, keepdims=True),
            jnp.max(ts, axis=0, keepdims=True),
        ],
        axis=0,
    )

    # Find overlapping bounds and apply constraints.
    min_t = jnp.maximum(0, jnp.max(ts[0]))
    max_t = jnp.min(ts[1])
    max_t_clipped = jnp.maximum(max_t, min_t + min_t_range)
    real_range = jnp.stack([min_t, max_t_clipped])
    null_range = jnp.array([0, min_t_range])
    mask = min_t < max_t
    return jnp.where(mask, real_range, null_range), mask
