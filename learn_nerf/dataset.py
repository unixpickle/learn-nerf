import json
import math
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

Vec3 = Tuple[float, float, float]


@dataclass
class NeRFView(ABC):
    camera_direction: Vec3
    camera_origin: Vec3
    x_axis: Vec3
    y_axis: Vec3
    x_fov: float
    y_fov: float

    @abstractmethod
    def image(self) -> jnp.ndarray:
        """
        Load the image as a [Height x Width x 3] array of uint8 RGB values.
        """

    def rays(self) -> jnp.ndarray:
        """
        Get all of the rays in the view with their corresponding colors as a
        single compact array.

        Returns an array of shape [N x 3 x 3] where each [3 x 3] element is a
        row-major tuple (origin, direction, color). Colors are stored as RGB
        values in the range [-1, 1].
        """
        img = self.image()
        z = jnp.array(self.camera_direction, dtype=jnp.float32)
        ys = (
            math.tan((math.pi / 180) * self.y_fov / 2)
            * jnp.linspace(-1, 1, num=img.shape[0])[:, None, None]
            * jnp.array(self.y_axis, dtype=jnp.float32)
        )
        xs = (
            math.tan((math.pi / 180) * self.x_fov / 2)
            * jnp.linspace(-1, 1, num=img.shape[1])[None, :, None]
            * jnp.array(self.x_axis, dtype=jnp.float32)
        )
        directions = jnp.reshape(xs + ys + z, [-1, 3])
        directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
        origins = jnp.reshape(
            jnp.tile(
                jnp.array(self.camera_origin, dtype=jnp.float32)[None, None],
                img.shape[:2] + (1,),
            ),
            [-1, 3],
        )
        colors = jnp.reshape(img, [-1, 3]).astype(jnp.float32) / 127.5 - 1
        return jnp.stack([origins, directions, colors], axis=1)


@dataclass
class FileNeRFView(NeRFView):
    image_path: str

    def image(self) -> jnp.ndarray:
        return jnp.array(Image.open(self.image_path).convert("RGB"))


@dataclass
class NeRFDataset:
    views: List[NeRFView]

    # Scene/object bounding box.
    bbox_min: Vec3
    bbox_max: Vec3

    def iterate_batches(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        repeat: bool = True,
        num_shards: int = 32,
    ) -> Iterator[jnp.ndarray]:
        """
        Load batches of colored rays from the dataset.

        :param key: the RNG seed for shuffling the data.
        :param batch_size: the number of rays to load per batch.
        :param repeat: if True, repeat the data after all rays have been
                       exhausted. If this is False, then the final batch may be
                       smaller than batch_size.
        :param num_shards: the number of temporary files to split the ray data
                           into while shuffling. Using more shards increases
                           the number of open file descriptors but reduces the
                           RAM usage of the dataset.
        :return: an iterator over [N x 3 x 3] batches of rays, where each ray
                 is a tuple (origin, direction, color).
        """
        # Perform the Jane Street two-stage shuffle as described in:
        # https://blog.janestreet.com/how-to-shuffle-a-big-dataset/.
        with _ShardManager(num_shards) as shards:
            # Stage 1: randomly assign rays to shards.
            for view in self.views:
                rays = view.rays()
                key, this_key = jax.random.split(key)
                assignments = jax.random.randint(
                    this_key, [rays.shape[0]], 0, num_shards
                )
                for shard in range(num_shards):
                    sub_batch = rays[assignments == shard]
                    if sub_batch.shape[0]:
                        shards.append_shard(shard, sub_batch)

            # Stage 2: read shards, and shuffle within each shard.
            cur_batch = None
            while True:
                for shard in range(num_shards):
                    key, this_key = jax.random.split(key)
                    shard_rays = jax.random.permutation(
                        this_key, shards.read_shard(shard)
                    )
                    if cur_batch is not None:
                        cur_batch = jnp.concatenate([cur_batch, shard_rays], axis=0)
                    else:
                        cur_batch = shard_rays
                    while cur_batch.shape[0] >= batch_size:
                        yield cur_batch[:batch_size]
                        cur_batch = cur_batch[batch_size:]
                if not repeat:
                    break
            if cur_batch.shape[0]:
                yield cur_batch

    def t_bounds(self) -> Tuple[float, float]:
        t_min = math.inf
        t_max = -math.inf

        corners = [
            (self.bbox_min[0], self.bbox_min[1], self.bbox_min[2]),
            (self.bbox_max[0], self.bbox_min[1], self.bbox_min[2]),
            (self.bbox_min[0], self.bbox_max[1], self.bbox_min[2]),
            (self.bbox_min[0], self.bbox_min[1], self.bbox_max[2]),
            (self.bbox_min[0], self.bbox_max[1], self.bbox_max[2]),
            (self.bbox_max[0], self.bbox_min[1], self.bbox_max[2]),
            (self.bbox_max[0], self.bbox_max[1], self.bbox_min[2]),
            (self.bbox_max[0], self.bbox_max[1], self.bbox_max[2]),
        ]

        for view in self.views:
            origin = view.camera_origin
            for corner in corners:
                dist = math.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(corner, origin)))
                t_min = min(t_min, dist)
                t_max = max(t_max, dist)

        return t_min, t_max


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
            FileNeRFView(
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


class _ShardManager:
    def __init__(self, num_files: int):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.fds = [
            open(os.path.join(self.tmp_dir.name, f"{i}"), "wb+")
            for i in range(num_files)
        ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for fd in self.fds:
            fd.close()
        self.tmp_dir.cleanup()

    def append_shard(self, shard: int, arr: jnp.ndarray):
        data = arr.astype(jnp.float32).tobytes()
        self.fds[shard].write(data)

    def read_shard(self, shard: int) -> jnp.ndarray:
        f = self.fds[shard]
        f.seek(0)
        data = f.read()
        return jnp.array(np.frombuffer(data, dtype=jnp.float32).reshape([-1, 3, 3]))
