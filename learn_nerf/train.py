import os
import pickle
from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core.scope import VariableDict
from flax.training import train_state
from jax._src.prng import PRNGKeyArray as KeyArray

from .model import NeRFModel
from .render import NeRFRenderer


class TrainLoop:
    """
    A stateful training loop.
    """

    def __init__(
        self,
        coarse: NeRFModel,
        fine: NeRFModel,
        init_rng: KeyArray,
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
            tx=optax.adam(lr, eps=1e-7),
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
            state: train_state.TrainState, key: KeyArray, batch: jnp.ndarray
        ) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
            loss_fn = partial(self.losses, key, t_min, t_max, background, batch)
            grad, values = jax.grad(loss_fn, has_aux=True)(state.params)
            return state.apply_gradients(grads=grad), values

        def in_place_step(key: KeyArray, batch: jnp.ndarray) -> Dict[str, jnp.ndarray]:
            self.state, ret_val = step_fn(self.state, key, batch)
            return ret_val

        return in_place_step

    def losses(
        self,
        key: KeyArray,
        t_min: jnp.ndarray,
        t_max: jnp.ndarray,
        background: jnp.ndarray,
        batch: jnp.ndarray,
        params: VariableDict,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute losses and a logging dict for a given batch and settings.
        """
        renderer = NeRFRenderer(
            coarse=self.coarse,
            fine=self.fine,
            coarse_params=params["coarse"],
            fine_params=params["fine"],
            background=background,
            t_min=t_min,
            t_max=t_max,
            coarse_ts=self.coarse_ts,
            fine_ts=self.fine_ts,
        )

        predictions = renderer.render_rays(key, batch[:, :2])
        targets = batch[:, 2]
        coarse_loss = jnp.mean((predictions["coarse"] - targets) ** 2)
        fine_loss = jnp.mean((predictions["fine"] - targets) ** 2)

        return coarse_loss + fine_loss, dict(coarse=coarse_loss, fine=fine_loss)
