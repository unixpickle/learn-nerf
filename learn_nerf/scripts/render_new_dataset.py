"""
Create a new NeRF dataset using a trained NeRF model by rendering random
viewing angles.
"""

import argparse
import math
import os
import pickle
import random
import shutil

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
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--model_path", type=str, default="nerf.pkl")
    add_model_args(parser)
    parser.add_argument("metadata_json", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        raise FileExistsError(f"output directory exists: {args.output_dir}")

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

    os.makedirs(args.output_dir)
    shutil.copy(args.metadata_json, os.path.join(args.output_dir, "metadata.json"))

    scale = float(jnp.linalg.norm(renderer.bbox_min - renderer.bbox_max))
    center = np.array((renderer.bbox_min + renderer.bbox_max) / 2)

    for frame in range(args.num_images):
        print(f"sampling frame {frame}...")
        z = np.random.normal(size=(3,))
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        x = x / np.linalg.norm(x)
        y = -np.cross(z, x)
        view = CameraView(
            camera_direction=tuple(z),
            camera_origin=tuple(-z * scale * args.distance + center),
            x_axis=tuple(x),
            y_axis=tuple(y),
            x_fov=60.0 * math.pi / 180,
            y_fov=60.0 * math.pi / 180,
        )
        with open(os.path.join(args.output_dir, f"{frame:05}.json"), "w") as f:
            f.write(view.to_json())
        rays = view.bare_rays(args.size, args.size)
        colors = jnp.zeros([0, 3])
        for i in tqdm(range(0, rays.shape[0], args.batch_size)):
            sub_batch = rays[i : i + args.batch_size]
            key, this_key = jax.random.split(key)
            sub_colors = render_fn(this_key, sub_batch)
            colors = jnp.concatenate([colors, sub_colors["fine"]], axis=0)
        image = (
            (np.array(colors).reshape([args.size, args.size, 3]) + 1) * 127.5
        ).astype(np.uint8)
        Image.fromarray(image).save(os.path.join(args.output_dir, f"{frame:05}.png"))


if __name__ == "__main__":
    main()
