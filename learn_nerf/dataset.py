import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import jax.numpy as jnp
from PIL import Image

Vec3 = Tuple[float, float, float]


@dataclass
class NeRFView:
    camera_direction: Vec3
    camera_origin: Vec3
    x_axis: Vec3
    y_axis: Vec3
    x_fov: float
    y_fov: float
    image_path: str

    def image(self) -> jnp.ndarray:
        """
        Load the image as a [Height x Width x 3] array of uint8 RGB values.
        """
        return jnp.array(Image.open(self.image_path).convert("RGB"))

    def rays(self) -> jnp.ndarray:
        """
        Get all of the rays in the view with their corresponding colors as a
        single compact array.

        Returns an array of shape [N x 3 x 3] where each [3 x 3] element is a
        row-major tuple (origin, direction, color). Colors are stored as RGB
        values in the range [-1, 1].
        """
        img = self.image()
        z = jnp.array(self.camera_direction, dtype=jnp.float32) / math.tan(
            (math.pi / 180) * self.fov / 2
        )
        ys = jnp.linspace(-1, 1, num=img.shape[0])[:, None, None] * jnp.array(
            self.y_axis, dtype=jnp.float32
        )
        xs = jnp.linspace(-1, 1, num=img.shape[1])[None, :, None] * jnp.array(
            self.x_axis, dtype=jnp.float32
        )
        directions = jnp.reshape(xs + ys + z, [-1, 3])
        directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
        origins = jnp.reshape(
            jnp.tile(
                jnp.array(self.camera_origin, dtype=jnp.float32)[None, None],
                img.shape[:2] + [1],
            ),
            [-1, 3],
        )
        colors = jnp.reshape(img, [-1, 3]).astype(jnp.float32) / 127.5 - 1
        return jnp.concatenate([origins, directions, colors], axis=1)


@dataclass
class NeRFDataset:
    views: List[NeRFView]

    # Scene/object bounding box.
    bbox_min: Vec3
    bbox_max: Vec3


def load_dataset(directory: str) -> NeRFDataset:
    """
    Load a dataset from a directory on disk.

    The dataset is stored as a combination of png files and json metadata files
    for each PNG file. For a file X.png, X.json is a file containing the
    following keys: "origin", "x", "y", "z", "x_fov", "y_fov" describing the
    camera. There is also a global "metadata.json" file containing the bounding
    box of the scene, stored as a dictionary with keys "min" and "max".
    """
    with open(os.path.join(directory, "metadata.json"), "rb") as f:
        metadata = json.load(f)
    dataset = NeRFDataset(
        views=[], bbox_min=tuple(metadata["min"]), bbox_max=tuple(metadata["max"])
    )
    for img_name in os.listdir(directory):
        if img_name.startswith(".") or not img_name.endswith(".png"):
            continue
        img_path = os.path.join(directory, img_name)
        json_path = img_path[: -len(".png")] + ".json"
        with open(json_path, "rb") as f:
            camera_info = json.load(f)
        dataset.views.append(
            NeRFView(
                camera_direction=tuple(camera_info["z"]),
                camera_origin=tuple(camera_info["origin"]),
                x_axis=tuple(camera_info["x"]),
                y_axis=tuple(camera_info["y"]),
                x_fov=float(camera_info["x_fov"]),
                y_fov=float(camera_info["y_fov"]),
                image_path=img_path,
            )
        )
    return dataset
