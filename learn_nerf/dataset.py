import json
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
    image: jnp.ndarray


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
        image = jnp.array(Image.open(img_path))
        dataset.views.append(
            NeRFView(
                camera_direction=tuple(camera_info["z"]),
                camera_origin=tuple(camera_info["origin"]),
                x_axis=tuple(camera_info["x"]),
                y_axis=tuple(camera_info["y"]),
                x_fov=float(camera_info["x_fov"]),
                y_fov=float(camera_info["y_fov"]),
                image=image,
            )
        )
    return dataset
