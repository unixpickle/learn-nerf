from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import jax.numpy as jnp

Vec3 = Tuple[float, float, float]


@dataclass
class NeRFView:
    size: Tuple[int, int]  # (width, height)
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
