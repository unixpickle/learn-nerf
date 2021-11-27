from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .dataset import NeRFDataset, NeRFView


@dataclass
class DummyView(NeRFView):
    dummy_image: jnp.ndarray

    def image(self) -> jnp.ndarray:
        return self.dummy_image


def test_nerf_dataset_iterate_batches():
    dataset = NeRFDataset(
        views=[
            DummyView(
                camera_direction=(0.0, 1.0, 0.0),
                camera_origin=(2.0, 2.0, 2.0),
                x_axis=(-1.0, 0.0, 0.0),
                y_axis=(0.0, 0.0, 1.0),
                x_fov=60.0,
                y_fov=60.0,
                dummy_image=jax.random.uniform(jax.random.PRNGKey(1337), (10, 10, 3)),
            ),
            DummyView(
                camera_direction=(1.0, 0.0, 0.0),
                camera_origin=(-2.0, 2.0, 2.0),
                x_axis=(-0.0, 0.0, -1.0),
                y_axis=(0.0, 1.0, 0.0),
                x_fov=60.0,
                y_fov=60.0,
                dummy_image=jax.random.uniform(jax.random.PRNGKey(1338), (10, 10, 3)),
            ),
        ],
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
    )
    batches = list(
        dataset.iterate_batches(jax.random.PRNGKey(1234), batch_size=51, repeat=False)
    )
    assert len(batches) == 4, "unexpected number of batches"
    assert batches[-1].shape[0] == 200 - 51 * 3, "unexpected last batch size"

    combined = jnp.concatenate(batches, axis=0)

    # Verify origin count.
    for view in dataset.views:
        origin = jnp.array(view.camera_origin, dtype=jnp.float32)
        origins = combined[:, 0]

        view_mask = jnp.sum(jnp.abs(origins - origin), axis=-1) < 1e-5
        count = jnp.sum(view_mask)
        num_pixels = view.dummy_image.shape[0] * view.dummy_image.shape[1]
        assert (
            int(count) == num_pixels
        ), f"unexpected number of samples with origin {origin}"

        view_rays = combined[view_mask]
        directions = view_rays[:, 1]
        mean_direction = jnp.mean(directions, axis=0)
        mean_direction = mean_direction / jnp.linalg.norm(mean_direction)
        camera_dot = jnp.sum(
            mean_direction * jnp.array(view.camera_direction, dtype=jnp.float32)
        )
        assert (
            abs(float(camera_dot) - 1) < 1e-5
        ), f"mean direction was {mean_direction} but should be {view.camera_direction}"

        colors = view_rays[:, 2]
        mean_color = jnp.mean(colors, axis=0)
        actual_mean = jnp.mean(view.dummy_image / 127.5 - 1, axis=(0, 1))
        diff = jnp.mean(jnp.abs(mean_color - actual_mean))
        assert float(diff) < 1e-5, "invalid colors for rays"
