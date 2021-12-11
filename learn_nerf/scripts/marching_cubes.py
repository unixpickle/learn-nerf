"""
Render a view using a NeRF model.
"""

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import skimage
from learn_nerf.dataset import ModelMetadata
from learn_nerf.model import NeRFModel
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024, help="rays per batch")
    parser.add_argument(
        "--resolution", type=int, default=32, help="steps along each direction"
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--model_path", type=str, default="nerf.pkl")
    parser.add_argument("metadata_json", type=str)
    parser.add_argument("output_png", type=str)
    args = parser.parse_args()

    print("loading metadata...")
    metadata = ModelMetadata.from_json(args.metadata_json)

    print("loading model...")
    fine = NeRFModel()
    with open(args.model_path, "rb") as f:
        params = pickle.load(f)["fine"]
    density_fn = jax.jit(
        lambda coords: fine.apply(dict(params=params), coords, jnp.zeros_like(coords))[
            0
        ]
    )

    input_xs = np.linspace(
        metadata.bbox_min[0], metadata.bbox_max[0], num=args.resolution
    )
    input_ys = np.linspace(
        metadata.bbox_min[1], metadata.bbox_max[1], num=args.resolution
    )
    input_zs = np.linspace(
        metadata.bbox_min[2], metadata.bbox_max[2], num=args.resolution
    )
    input_coords = jnp.array(
        [[x, y, z] for z in input_zs for y in input_ys for x in input_xs]
    )

    outputs = []
    for i in tqdm(range(0, input_coords.shape[0], args.batch_size)):
        batch = input_coords[i : i + args.batch_size]
        density = density_fn(batch)
        outputs.append(density - args.threshold)

    volume = np.array(jnp.concatenate(outputs, axis=0).reshape([args.resolution] * 3))

    verts, faces, normals, values = skimage.measure.marching_cubes
    # TODO: figure out how to render/save the model.


if __name__ == "__main__":
    main()
