import json
import math
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.prng import PRNGKeyArray as KeyArray
from PIL import Image

Vec3 = Tuple[float, float, float]


@dataclass
class CameraView:
    camera_direction: Vec3
    camera_origin: Vec3
    x_axis: Vec3
    y_axis: Vec3
    x_fov: float
    y_fov: float

    @classmethod
    def from_json(cls, path: str, **kwargs) -> "CameraView":
        with open(path, "rb") as f:
            camera_info = json.load(f)
        return cls(
            camera_direction=tuple(camera_info["z"]),
            camera_origin=tuple(camera_info["origin"]),
            x_axis=tuple(camera_info["x"]),
            y_axis=tuple(camera_info["y"]),
            x_fov=float(camera_info["x_fov"]),
            y_fov=float(camera_info["y_fov"]),
            **kwargs,
        )

    def to_json(self) -> str:
        return json.dumps(
            dict(
                z=self.camera_direction,
                origin=self.camera_origin,
                x=self.x_axis,
                y=self.y_axis,
                x_fov=self.x_fov,
                y_fov=self.y_fov,
            )
        )

    def bare_rays(self, width: int, height: int) -> jnp.ndarray:
        """
        Get all of the rays in the view in raster scan order.

        Returns an [N x 2 x 3] array of (origin, direction) pairs.
        """
        z = jnp.array(self.camera_direction, dtype=jnp.float32)
        ys = (
            math.tan(self.y_fov / 2)
            * jnp.linspace(-1, 1, num=height)[:, None, None]
            * jnp.array(self.y_axis, dtype=jnp.float32)
        )
        xs = (
            math.tan(self.x_fov / 2)
            * jnp.linspace(-1, 1, num=width)[None, :, None]
            * jnp.array(self.x_axis, dtype=jnp.float32)
        )
        directions = jnp.reshape(xs + ys + z, [-1, 3])
        directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
        origins = jnp.reshape(
            jnp.tile(
                jnp.array(self.camera_origin, dtype=jnp.float32)[None, None],
                (height, width, 1),
            ),
            [-1, 3],
        )
        return jnp.stack([origins, directions], axis=1)


@dataclass
class NeRFView(CameraView):
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
        bare = self.bare_rays(img.shape[1], img.shape[0])
        colors = jnp.reshape(img, [-1, 3]).astype(jnp.float32) / 127.5 - 1
        return jnp.concatenate([bare, colors[:, None]], axis=1)


@dataclass
class FileNeRFView(NeRFView):
    image_path: str

    def image(self) -> jnp.ndarray:
        # Premultiply alpha to prevent egregious errors at the border.
        rgba = jnp.array(Image.open(self.image_path).convert("RGBA"))
        return jnp.round((rgba[:, :, :3] * (rgba[:, :, 3:] / 255))).astype(jnp.uint8)


@dataclass
class ModelMetadata:
    # Scene/object bounding box.
    bbox_min: Vec3
    bbox_max: Vec3

    @classmethod
    def from_json(cls, path: str) -> "ModelMetadata":
        with open(path, "rb") as f:
            metadata = json.load(f)
        return ModelMetadata(
            bbox_min=tuple(metadata["min"]), bbox_max=tuple(metadata["max"])
        )


@dataclass
class NeRFDataset:
    metadata: ModelMetadata
    views: List[NeRFView]

    def iterate_batches(
        self,
        dir_path: str,
        key: KeyArray,
        batch_size: int,
        repeat: bool = True,
        num_shards: int = 32,
    ) -> Iterator[jnp.ndarray]:
        """
        Load batches of colored rays from the dataset in a shuffled fashion.

        :param dir_path: directory where the shuffled data is stored.
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
        with ShuffledDataset(dir_path, self, key, num_shards=num_shards) as sd:
            yield from sd.iterate_batches(batch_size, repeat=repeat)


class ShuffledDataset:
    """
    A pre-shuffled version of the rays in a NeRFDataset.

    Uses the Jane Street two-stage shuffle as described in:
    https://blog.janestreet.com/how-to-shuffle-a-big-dataset/.

    :param dir_path: the directory to store results.
    :param dataset: the dataset to shuffle.
    :param key: the RNG key for shuffling.
    :param num_shards: the number of files to split rays into. More shards
                       uses less memory but more file descriptors.
    """

    def __init__(
        self,
        dir_path: str,
        dataset: NeRFDataset,
        key: KeyArray,
        num_shards: int = 32,
    ):
        self.num_shards = num_shards
        self.shard_key, self.shuffle_key = jax.random.split(key)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        done_path = os.path.join(dir_path, "done")
        if os.path.exists(done_path):
            self.fds = [
                open(os.path.join(dir_path, f"{i}"), "rb") for i in range(num_shards)
            ]
        else:
            self.fds = [
                open(os.path.join(dir_path, f"{i}"), "wb+") for i in range(num_shards)
            ]
            self._create_shards(dataset)
            with open(done_path, "wb+") as f:
                f.write(b"done\n")

    def iterate_batches(
        self, batch_size: int, repeat: bool = False
    ) -> Iterator[jnp.ndarray]:
        """
        Load batches of colored rays from the dataset.

        :param batch_size: the number of rays to load per batch.
        :param repeat: if True, repeat the data after all rays have been
                       exhausted. If this is False, then the final batch may be
                       smaller than batch_size.
        :return: an iterator over [N x 3 x 3] batches of rays, where each ray
                 is a tuple (origin, direction, color).
        """
        key = self.shuffle_key
        cur_batch = None
        while True:
            key, this_key = jax.random.split(key)
            shard_indices = np.array(
                jax.random.permutation(this_key, jnp.arange(self.num_shards))
            ).tolist()
            for shard in shard_indices:
                key, this_key = jax.random.split(key)
                shard_rays = jax.random.permutation(this_key, self._read_shard(shard))
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for fd in self.fds:
            fd.close()

    def _create_shards(self, dataset: NeRFDataset):
        key = self.shard_key
        for view in dataset.views:
            rays = view.rays()
            key, this_key = jax.random.split(key)
            assignments = jax.random.randint(
                this_key, [rays.shape[0]], 0, self.num_shards
            )
            for shard in range(self.num_shards):
                sub_batch = rays[assignments == shard]
                if sub_batch.shape[0]:
                    self._append_shard(shard, sub_batch)

    def _append_shard(self, shard: int, arr: jnp.ndarray):
        data = arr.astype(jnp.float32).tobytes()
        self.fds[shard].write(data)

    def _read_shard(self, shard: int) -> jnp.ndarray:
        f = self.fds[shard]
        f.seek(0)
        data = f.read()
        return jnp.array(np.frombuffer(data, dtype=jnp.float32).reshape([-1, 3, 3]))


def load_dataset(directory: str) -> NeRFDataset:
    """
    Load a dataset from a directory on disk.

    The dataset is stored as a combination of png files and json metadata files
    for each PNG file. For a file X.png, X.json is a file containing the
    following keys: "origin", "x", "y", "z", "x_fov", "y_fov" describing the
    camera. There is also a global "metadata.json" file containing the bounding
    box of the scene, stored as a dictionary with keys "min" and "max".
    """
    dataset = NeRFDataset(
        metadata=ModelMetadata.from_json(os.path.join(directory, "metadata.json")),
        views=[],
    )
    for img_name in os.listdir(directory):
        if img_name.startswith(".") or not img_name.endswith(".png"):
            continue
        img_path = os.path.join(directory, img_name)
        json_path = img_path[: -len(".png")] + ".json"
        dataset.views.append(FileNeRFView.from_json(json_path, image_path=img_path))
    return dataset
