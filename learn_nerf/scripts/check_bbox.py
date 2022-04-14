"""
Compute (min, max, mean) of the pixels for rays shooting outside of a scene's
bounding box to make sure the bounding box actually covers everything.
"""

import argparse

import jax
import jax.numpy as jnp
from learn_nerf.dataset import load_dataset
from learn_nerf.render import ray_t_range
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)

    bbox = jnp.array((dataset.metadata.bbox_min, dataset.metadata.bbox_max))
    ray_masks = jax.jit(
        lambda rays: jax.vmap(lambda ray: ray_t_range(bbox, ray))(rays)[1]
    )

    min_color = None
    max_color = None
    color_sum = None
    total_colors = 0.0
    for view in tqdm(dataset.views):
        colored_rays = view.rays()
        rays, colors = colored_rays[:, :2], colored_rays[:, 2]
        masked_colors = colors[~ray_masks(rays)]
        if not jnp.any(masked_colors):
            continue
        local_min = jnp.min(masked_colors, axis=0)
        local_max = jnp.max(masked_colors, axis=0)
        local_sum = jnp.sum(masked_colors, axis=0)
        if min_color is None:
            min_color, max_color, color_sum = local_min, local_max, local_sum
        else:
            min_color = jnp.minimum(min_color, local_min)
            max_color = jnp.maximum(max_color, local_max)
            color_sum = color_sum + local_sum
        total_colors += masked_colors.shape[0]
    mean_color = color_sum / total_colors
    print("min color", min_color.tolist())
    print("max color", max_color.tolist())
    print("mean color", mean_color.tolist())


if __name__ == "__main__":
    main()
