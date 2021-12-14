"""
Apply marching cubes on a trained NeRF model to reproduce a mesh.
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
    parser.add_argument("output_obj", type=str)
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

    input_steps = [
        pad_edges(np.linspace(bbox_min, bbox_max, num=args.resolution))
        for bbox_min, bbox_max in zip(metadata.bbox_min, metadata.bbox_max)
    ]
    input_coords = jnp.array(
        [
            [x, y, z]
            for z in input_steps[2]
            for y in input_steps[1]
            for x in input_steps[0]
        ]
    )

    outputs = []
    for i in tqdm(range(0, input_coords.shape[0], args.batch_size)):
        batch = input_coords[i : i + args.batch_size]
        density = density_fn(batch)
        outputs.append(density - args.threshold)

    volume = np.array(
        jnp.concatenate(outputs, axis=0).reshape([args.resolution + 2] * 3)
    )

    # Adapted from https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html.
    verts, faces, _normals, _values = skimage.measure.marching_cubes(volume, level=0)

    verts = flip_x_and_z(verts)
    size = np.array(metadata.bbox_max) - np.array(metadata.bbox_min)
    verts *= size / args.resolution
    verts -= (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2

    write_obj(args.output_obj, verts, faces)


def flip_x_and_z(tris: np.ndarray) -> np.ndarray:
    return np.stack([tris[..., 2], tris[..., 1], tris[..., 0]], axis=-1)


def pad_edges(arr: np.ndarray) -> np.ndarray:
    step = arr[1] - arr[0]
    return np.concatenate([arr[:1] - step, arr, arr[-1:] + step])


def write_obj(path: str, vertices: np.ndarray, faces: np.ndarray):
    vertex_strs = [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in vertices.tolist()]
    face_strs = [f"f {x[0]+1} {x[1]+1} {x[2]+1}" for x in faces.tolist()]
    with open(path, "w") as f:
        f.write("\n".join(vertex_strs) + "\n")
        f.write("\n".join(face_strs) + "\n")


if __name__ == "__main__":
    main()
