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
    parser = argparser()
    parser.add_argument("view_json", type=str, nargs="+")
    parser.add_argument("output_png", type=str)
    args = parser.parse_args()

    renderer = RenderSession(args)
    for view_json in args.view_json:
        print(f"rendering view {view_json}...")
        renderer.render_view(CameraView.from_json(view_json))
    renderer.save(args.output_png)


def argparser():
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
    return parser


class RenderSession:
    def __init__(self, args: argparse.Namespace):
        print("loading metadata...")
        self.metadata = ModelMetadata.from_json(args.metadata_json)

        print("loading model...")
        coarse, fine, _ = create_model(args, self.metadata)
        with open(args.model_path, "rb") as f:
            params = pickle.load(f)

        self.renderer = NeRFRenderer(
            coarse=coarse,
            fine=fine,
            coarse_params=params["coarse"],
            fine_params=params["fine"],
            background=params["background"],
            bbox_min=jnp.array(self.metadata.bbox_min, dtype=jnp.float32),
            bbox_max=jnp.array(self.metadata.bbox_max, dtype=jnp.float32),
            coarse_ts=args.coarse_samples,
            fine_ts=args.fine_samples,
        )
        self.render_fn = jax.jit(
            lambda *args: self.renderer.render_rays(*args)["fine"]["outputs"]
        )

        self.key = jax.random.PRNGKey(
            args.seed if args.seed is not None else random.randint(0, 2 ** 32 - 1)
        )

        self.args = args
        self.images = []

    def render_view(self, view: CameraView):
        rays = view.bare_rays(self.args.width, self.args.height)
        colors = jnp.zeros([0, 3])
        for i in tqdm(range(0, rays.shape[0], self.args.batch_size)):
            sub_batch = rays[i : i + self.args.batch_size]
            self.key, this_key = jax.random.split(self.key)
            sub_colors = self.render_fn(this_key, sub_batch)
            colors = jnp.concatenate([colors, sub_colors], axis=0)
        image = (
            (np.array(colors).reshape([self.args.height, self.args.width, 3]) + 1)
            * 127.5
        ).astype(np.uint8)
        self.images.append(image)

    def save(self, output_path: str):
        image = np.concatenate(self.images, axis=1)
        Image.fromarray(image).save(output_path)


if __name__ == "__main__":
    main()
