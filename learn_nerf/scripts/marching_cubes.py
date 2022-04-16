"""
Apply marching cubes on a trained NeRF model to reproduce a mesh.
"""

import argparse
import math
import pickle
import struct
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import skimage
from learn_nerf.dataset import ModelMetadata
from learn_nerf.scripts.train_nerf import add_model_args, create_model
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024, help="rays per batch")
    parser.add_argument(
        "--resolution", type=int, default=32, help="steps along each direction"
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--model_path", type=str, default="nerf.pkl")
    add_model_args(parser)
    parser.add_argument("metadata_json", type=str)
    parser.add_argument("output_obj", type=str)
    args = parser.parse_args()

    print("loading metadata...")
    metadata = ModelMetadata.from_json(args.metadata_json)

    print("loading model...")
    _, fine, _ = create_model(args, metadata)
    with open(args.model_path, "rb") as f:
        params = pickle.load(f)["fine"]

    density_fn = jax.jit(
        lambda coords: (
            1
            - jnp.exp(
                -fine.apply(dict(params=params), coords, jnp.zeros_like(coords))[0]
            )
        )
    )

    input_coords = grid_coordinates(
        bbox_min=metadata.bbox_min,
        bbox_max=metadata.bbox_max,
        grid_size=args.resolution,
    ).reshape([-1, 3])

    print("computing densities...")
    outputs = []
    for i in tqdm(range(0, input_coords.shape[0], args.batch_size)):
        batch = input_coords[i : i + args.batch_size]
        density = density_fn(batch)
        outputs.append(density)

    volume = np.array(jnp.concatenate(outputs, axis=0).reshape([args.resolution] * 3))
    volume = np.pad(volume, 1, mode="constant", constant_values=0)

    # Adapted from https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html.
    verts, faces, normals, _values = skimage.measure.marching_cubes(
        volume, level=args.threshold
    )

    verts = flip_x_and_z(verts)
    size = np.array(metadata.bbox_max) - np.array(metadata.bbox_min)
    verts *= size / args.resolution
    verts -= (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2

    if args.output_obj.endswith(".obj"):
        write_obj(args.output_obj, verts, faces)
    elif args.output_obj.endswith(".stl"):
        write_stl(args.output_stl, verts, faces, normals)


def flip_x_and_z(tris: np.ndarray) -> np.ndarray:
    return np.stack([tris[..., 2], tris[..., 1], tris[..., 0]], axis=-1)


def grid_coordinates(
    bbox_min: Sequence[float], bbox_max: Sequence[float], grid_size: int
) -> np.ndarray:
    result = np.empty([grid_size] * 3 + [3])
    for i, (bbox_min, bbox_max) in enumerate(zip(bbox_min, bbox_max)):
        sub_size = [grid_size if i == j else 1 for j in range(3)]
        result[..., i] = np.linspace(bbox_min, bbox_max, num=grid_size).reshape(
            sub_size
        )
    return result


def write_obj(path: str, vertices: np.ndarray, faces: np.ndarray):
    vertex_strs = [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in vertices.tolist()]
    face_strs = [f"f {x[0]+1} {x[1]+1} {x[2]+1}" for x in faces.tolist()]
    with open(path, "w") as f:
        f.write("\n".join(vertex_strs) + "\n")
        f.write("\n".join(face_strs) + "\n")


def write_stl(path: str, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray):
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(faces)))
        combined = np.concatenate([normals[:, None], vertices[faces]], axis=1)
        packed_coords = struct.pack("<{'f'*12}", combined)

        # Add b'\x00\x00' after each triangle.
        rows = np.frombuffer(packed_coords, dtype=np.uint8).reshape([-1, 3 * 4 * 4])
        padded = np.concatenate([rows, np.zeros_like(rows[:, :2])], axis=-1).tobytes()

        f.write(padded)


if __name__ == "__main__":
    main()
