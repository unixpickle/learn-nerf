"""
Render a view using a NeRF model.
"""

import argparse
import pickle
import random

import jax
import jax.numpy as jnp
import numpy as np
from learn_nerf.dataset import CameraView, ModelMetadata
from learn_nerf.render import NeRFRenderer
from learn_nerf.scripts.train_nerf import add_model_args, create_model
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
    parser.add_argument("--model_path", type=str, default="nerf.pkl")
    add_model_args(parser)
    parser.add_argument("metadata_json", type=str)
    parser.add_argument("view_json", type=str, nargs="+")
    parser.add_argument("output_png", type=str)
    args = parser.parse_args()

    print("loading metadata...")
    metadata = ModelMetadata.from_json(args.metadata_json)

    print("loading model...")
    coarse, fine, _ = create_model(args, metadata)
    with open(args.model_path, "rb") as f:
        params = pickle.load(f)

    renderer = NeRFRenderer(
        coarse=coarse,
        fine=fine,
        coarse_params=params["coarse"],
        fine_params=params["fine"],
        background=params["background"],
        bbox_min=jnp.array(metadata.bbox_min, dtype=jnp.float32),
        bbox_max=jnp.array(metadata.bbox_max, dtype=jnp.float32),
        coarse_ts=args.coarse_samples,
        fine_ts=args.fine_samples,
    )
    render_fn = jax.jit(lambda *args: renderer.render_rays(*args))

    key = jax.random.PRNGKey(
        args.seed if args.seed is not None else random.randint(0, 2 ** 32 - 1)
    )

    images = []
    for view_json in args.view_json:
        print(f"rendering view {view_json}...")
        view = CameraView.from_json(view_json)
        rays = view.bare_rays(args.width, args.height)
        colors = jnp.zeros([0, 3])
        for i in tqdm(range(0, rays.shape[0], args.batch_size)):
            sub_batch = rays[i : i + args.batch_size]
            key, this_key = jax.random.split(key)
            sub_colors = render_fn(this_key, sub_batch)
            colors = jnp.concatenate([colors, sub_colors["fine"]], axis=0)
        image = (
            (np.array(colors).reshape([args.height, args.width, 3]) + 1) * 127.5
        ).astype(np.uint8)
    image = np.concatenate(images, axis=1)
    Image.fromarray(image).save(args.output_png)


if __name__ == "__main__":
    main()
