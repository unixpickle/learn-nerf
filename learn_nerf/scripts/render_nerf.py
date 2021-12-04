"""
Render a view using a NeRF model.
"""

import argparse
import json
import pickle
import random

import jax
import jax.numpy as jnp
import numpy as np
from learn_nerf.dataset import CameraView
from learn_nerf.model import NeRFModel
from learn_nerf.render import NeRFRenderer
from PIL import Image
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1024, help="rays per batch")
    parser.add_argument(
        "--coarse_samples", type=int, default=64, help="samples per coarse ray"
    )
    parser.add_argument(
        "--fine_samples",
        type=int,
        default=128,
        help="samples per fine ray (not including coarse samples)",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--t_min", type=float, default=0.0)
    parser.add_argument("--t_max", type=float, default=15.0)
    parser.add_argument("--model_path", type=str, default="nerf.pkl")
    parser.add_argument("metadata_json", type=str)
    parser.add_argument("view_json", type=str)
    parser.add_argument("output_png", type=str)
    args = parser.parse_args()

    print("loading view and metadata...")
    view = CameraView.from_json(args.view_json)
    with open(args.metadata_path, "rb") as f:
        metadata = json.load(f)

    print("gathering rays...")
    rays = view.bare_rays(args.width, args.height)

    print("loading model...")
    coarse = NeRFModel()
    fine = NeRFModel()
    with open(args.model_path, "rb") as f:
        params = pickle.load(f)

    renderer = NeRFRenderer(
        coarse=coarse,
        fine=fine,
        coarse_params=params["coarse"],
        fine_params=params["fine"],
        background=jnp.array([-1.0, -1.0, -1.0]),
        bbox_min=jnp.array(metadata["min"], dtype=jnp.float32),
        bbox_max=jnp.array(metadata["max"], dtype=jnp.float32),
        coarse_ts=args.coarse_samples,
        fine_ts=args.fine_samples,
    )
    render_fn = jax.jit(lambda *args: renderer.render_rays(*args))

    key = jax.random.PRNGKey(
        args.seed if args.seed is not None else random.randint(0, 2 ** 32 - 1)
    )

    print("sampling pixels...")
    colors = jnp.zeros([0, 3])
    for i in tqdm(range(0, rays.shape[0], args.batch_size)):
        sub_batch = rays[i : i + args.batch_size]
        key, this_key = jax.random.split(key)
        sub_colors = render_fn(this_key, sub_batch)
        colors = jnp.concatenate([colors, sub_colors["fine"]], axis=0)
    image = (
        (np.array(colors).reshape([args.height, args.width, 3]) + 1) * 127.5
    ).astype(jnp.uint8)
    Image.fromarray(image).save(args.output_png)


if __name__ == "__main__":
    main()
