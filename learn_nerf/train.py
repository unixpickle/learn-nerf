import os
import pickle
from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core.scope import VariableDict
from flax.training import train_state
from jax._src.prng import PRNGKeyArray as KeyArray

from .dataset import Vec3
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
            apply_fn=None,
            params=dict(
                coarse=coarse_vars["params"],
                fine=fine_vars["params"],
                # Initialize background as all black.
                background=jnp.array([-1.0, -1.0, -1.0]),
            ),
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
        self, bbox_min: jnp.ndarray, bbox_max: jnp.ndarray
    ) -> Callable[[jax.random.PRNGKey, jnp.ndarray], Dict[str, jnp.ndarray]]:
        """
        Create a function that steps in place and returns a logging dict.
        """

        @jax.jit
        def step_fn(
            state: train_state.TrainState, key: KeyArray, batch: jnp.ndarray
        ) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
            loss_fn = partial(self.losses, key, bbox_min, bbox_max, batch)
            grad, values = jax.grad(loss_fn, has_aux=True)(state.params)

            def tree_norm(tree: Any) -> jnp.ndarray:
                return jnp.sqrt(
                    jax.tree_util.tree_reduce(
                        lambda total, x: total + jnp.sum(x ** 2), tree, jnp.array(0.0)
                    )
                )

            values.update(
                dict(grad_norm=tree_norm(grad), param_norm=tree_norm(state.params))
            )
            return state.apply_gradients(grads=grad), values

        def in_place_step(key: KeyArray, batch: jnp.ndarray) -> Dict[str, jnp.ndarray]:
            self.state, ret_val = step_fn(self.state, key, batch)
            return ret_val

        return in_place_step

    def losses(
        self,
        key: KeyArray,
        bbox_min: jnp.ndarray,
        bbox_max: jnp.ndarray,
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
            background=params["background"],
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            coarse_ts=self.coarse_ts,
            fine_ts=self.fine_ts,
        )

        predictions = renderer.render_rays(key, batch[:, :2])
        targets = batch[:, 2]
        coarse_loss = jnp.mean((predictions["coarse"] - targets) ** 2)
        fine_loss = jnp.mean((predictions["fine"] - targets) ** 2)

        return coarse_loss + fine_loss, dict(coarse=coarse_loss, fine=fine_loss)
