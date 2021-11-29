import os
import pickle
from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core.scope import VariableDict
from flax.training import train_state

from .model import NeRFModel
from .render import RaySamples


class TrainLoop:
    """
    A stateful training loop.
    """

    def __init__(
        self,
        coarse: NeRFModel,
        fine: NeRFModel,
        init_rng: jax.random.PRNGKey,
        lr: float,
        coarse_ts: int,
        fine_ts: int,
    ):
        self.coarse = coarse
        self.fine = fine
        self.coarse_ts = coarse_ts
        self.fine_ts = fine_ts

        coarse_rng, fine_rng = jax.random.split(init_rng)
        example_batch = jnp.array([[0.0, 0.0, 0.0]])
        coarse_vars = coarse.init(dict(params=coarse_rng), example_batch, example_batch)
        fine_vars = fine.init(dict(params=fine_rng), example_batch, example_batch)
        self.state = train_state.TrainState.create(
            apply_fn=self.losses,
            params=dict(coarse=coarse_vars["params"], fine=fine_vars["params"]),
            tx=optax.adam(lr),
        )

    def save(self, path: str):
        """
        Save the model parameters to a file.
        """
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(self.state.params, f)
        os.rename(tmp_path, path)

    def load(self, path: str):
        """
        Load the model parameters from a file.
        """
        with open(path, "rb") as f:
            self.state = self.state.replace(params=pickle.load(f))

    def step_fn(
        self, t_min: jnp.ndarray, t_max: jnp.ndarray, background: jnp.ndarray
    ) -> Callable[[jax.random.PRNGKey, jnp.ndarray], Dict[str, jnp.ndarray]]:
        """
        Create a function that steps in place and returns a logging dict.
        """

        @jax.jit
        def step_fn(
            state: train_state.TrainState, key: jax.random.PRNGKey, batch: jnp.ndarray
        ) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
            loss_fn = partial(self.losses, key, t_min, t_max, background, batch)
            grad, values = jax.grad(loss_fn, has_aux=True)(state.params)
            return state.apply_gradients(grads=grad), values

        def in_place_step(
            key: jax.random.PRNGKey, batch: jnp.ndarray
        ) -> Dict[str, jnp.ndarray]:
            self.state, ret_val = step_fn(self.state, key, batch)
            return ret_val

        return in_place_step

    def losses(
        self,
        key: jax.random.PRNGKey,
        t_min: jnp.ndarray,
        t_max: jnp.ndarray,
        background: jnp.ndarray,
        batch: jnp.ndarray,
        params: VariableDict,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute losses and a logging dict for a given batch and settings.
        """
        rays = batch[:, :2]
        targets = batch[:, 2]
        coarse_key, fine_key = jax.random.split(key)

        # Evaluate the coarse model using regular stratified sampling.
        coarse_ts = RaySamples.stratified_sampling(
            batch_size=batch.shape[0],
            count=self.coarse_ts,
            key=coarse_key,
            t_min=t_min,
            t_max=t_max,
        )
        all_points = coarse_ts.points(rays)
        direction_batch = jnp.tile(rays[:, 1:2], [1, all_points.shape[1], 1])
        coarse_densities, coarse_rgbs = self.coarse.apply(
            dict(params=params["coarse"]),
            all_points.reshape([-1, 3]),
            direction_batch.reshape([-1, 3]),
        )
        coarse_densities = coarse_densities.reshape(all_points.shape[:-1])
        coarse_rgbs = coarse_rgbs.reshape(all_points.shape)
        coarse_outputs = coarse_ts.render_rays(
            coarse_densities, coarse_rgbs, background
        )
        coarse_loss = jnp.mean((coarse_outputs - targets) ** 2)

        # Evaluate the fine model using a combined set of points.
        fine_ts = coarse_ts.fine_sampling(
            count=self.fine_ts,
            key=fine_key,
            densities=jax.lax.stop_gradient(coarse_densities),
        )
        all_points = fine_ts.points(rays)
        direction_batch = jnp.tile(rays[:, 1:2], [1, all_points.shape[1], 1])
        fine_densities, fine_rgbs = self.fine.apply(
            dict(params=params["fine"]),
            all_points.reshape([-1, 3]),
            direction_batch.reshape([-1, 3]),
        )
        fine_densities = fine_densities.reshape(all_points.shape[:-1])
        fine_rgbs = fine_rgbs.reshape(all_points.shape)
        fine_outputs = fine_ts.render_rays(fine_densities, fine_rgbs, background)
        fine_loss = jnp.mean((fine_outputs - targets) ** 2)

        return coarse_loss + fine_loss, dict(coarse=coarse_loss, fine=fine_loss)
