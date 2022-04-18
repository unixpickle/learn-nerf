"""
Render a panning view of a NeRF model. Work in progress.
"""

import argparse
import math
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
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--random_axis", action="store_true")
    parser.add_argument("--model_path", type=str, default="nerf.pkl")
    add_model_args(parser)
    parser.add_argument("metadata_json", type=str)
    parser.add_argument("output_png", type=str)
    args = parser.parse_args()

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

    scale = float(jnp.linalg.norm(renderer.bbox_min - renderer.bbox_max))
    center = np.array((renderer.bbox_min + renderer.bbox_max) / 2)

    rot_axis = np.array([0.0, 0.0, -1.0])
    rot_basis_1 = np.array([1.0, 0.0, 0.0])
    if args.random_axis:
        rot_axis = np.random.normal(size=(3,))
        rot_axis /= np.linalg.norm(rot_axis)
        rot_basis_1 = np.array([-rot_axis[2], 0.0, rot_axis[0]])
        rot_basis_1 /= np.linalg.norm(rot_basis_1)
    rot_basis_2 = np.cross(rot_axis, rot_basis_1)

    frame_arrays = []
    for frame in range(args.frames):
        print(f"sampling frame {frame}...")
        theta = (frame / args.frames) * math.pi * 2
        direction = np.cos(theta) * rot_basis_1 + np.sin(theta) * rot_basis_2
        view = CameraView(
            camera_direction=tuple(direction),
            camera_origin=tuple(-direction * scale * args.distance + center),
            x_axis=tuple(
                np.cos(theta + np.pi / 2) * rot_basis_1
                + np.sin(theta + np.pi / 2) * rot_basis_2
            ),
            y_axis=tuple(rot_axis),
            x_fov=60.0 * math.pi / 180,
            y_fov=60.0 * math.pi / 180,
        )
        rays = view.bare_rays(args.width, args.height)
        colors = jnp.zeros([0, 3])
        for i in tqdm(range(0, rays.shape[0], args.batch_size)):
            sub_batch = rays[i : i + args.batch_size]
            key, this_key = jax.random.split(key)
            sub_colors = render_fn(this_key, sub_batch)
            colors = jnp.concatenate([colors, sub_colors["fine"]], axis=0)
        image = (
            (np.array(colors).reshape([args.height, args.width, 3]) + 1) * 127.5
        ).astype(jnp.uint8)
        frame_arrays.append(image)
    joined = np.concatenate(frame_arrays, axis=1)
    Image.fromarray(joined).save(args.output_png)


if __name__ == "__main__":
    main()
