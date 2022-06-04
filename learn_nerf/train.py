import os
import pickle
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core.scope import VariableDict
from flax.training import train_state
from jax._src.prng import PRNGKeyArray as KeyArray

from .model import ModelBase
from .render import NeRFRenderer


class TrainLoop:
    """
    A stateful training loop.
    """

    def __init__(
        self,
        coarse: ModelBase,
        fine: ModelBase,
        init_rng: KeyArray,
        lr: float,
        coarse_ts: int,
        fine_ts: int,
        adam_b1: float = 0.9,
        adam_b2: float = 0.999,
        adam_eps: float = 1e-7,
        loss_weights: Dict[str, float] = None,
        density_penalty: Optional[float] = None,
        density_penalty_batch_size: int = 128,
    ):
        self.coarse = coarse
        self.fine = fine
        self.coarse_ts = coarse_ts
        self.fine_ts = fine_ts
        self.loss_weights = (
            loss_weights if loss_weights is not None else default_loss_weights()
        )
        self.density_penalty = density_penalty
        self.density_penalty_batch_size = density_penalty_batch_size

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
            tx=optax.adam(lr, b1=adam_b1, b2=adam_b2, eps=adam_eps),
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
                dict(
                    grad_norm=tree_norm(grad),
                    param_norm=tree_norm(state.params),
                )
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

        key, density_key = jax.random.split(key)

        render_out = renderer.render_rays(key, batch[:, :2])
        targets = batch[:, 2]
        coarse_loss = jnp.mean((render_out["coarse"]["outputs"] - targets) ** 2)
        fine_loss = jnp.mean((render_out["fine"]["outputs"] - targets) ** 2)

        loss_dict = dict(coarse=coarse_loss, fine=fine_loss)
        total_loss = coarse_loss + fine_loss
        for name, loss in render_out["coarse_aux"].items():
            loss_dict[f"coarse_{name}"] = loss
            total_loss = total_loss + self.loss_weights[name] * loss
        for name, loss in render_out["fine_aux"].items():
            loss_dict[f"fine_{name}"] = loss
            total_loss = total_loss + self.loss_weights[name] * loss

        if self.density_penalty is not None:
            for prefix, model in [("fine", self.fine), ("coarse", self.coarse)]:
                penalty = self.average_density(
                    key=density_key,
                    model=model,
                    params=params[prefix],
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                )
                loss_dict[f"{prefix}_density"] = penalty
                total_loss = total_loss + self.density_penalty * penalty

        return total_loss, loss_dict

    def average_density(
        self,
        key: KeyArray,
        model: ModelBase,
        params: Any,
        bbox_min: jnp.ndarray,
        bbox_max: jnp.ndarray,
    ) -> jnp.ndarray:
        coords = (
            jax.random.uniform(key, shape=(self.density_penalty_batch_size, 3))
            * (bbox_max - bbox_min)
            + bbox_min
        )
        dirs = jax.random.normal(key, shape=(self.density_penalty_batch_size, 3))
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)

        densities, _, _ = model.apply(dict(params=params), coords, dirs)
        return jnp.mean(densities)


def default_loss_weights() -> Dict[str, float]:
    return dict(
        normal_mse=3e-4,
        neg_normal=0.1,
    )
