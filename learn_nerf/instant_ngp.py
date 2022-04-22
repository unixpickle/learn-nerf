"""
A simple JAX re-implementation of Instant NGP:
https://arxiv.org/abs/2201.05989.
"""

from typing import Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .model import ModelBase, sinusoidal_emb
from .ref_nerf import RefNERFBase


class InstantNGPModel(ModelBase):
    """
    A NeRF model that utilizes a multilevel hash table.
    """

    table_sizes: List[int]
    grid_sizes: List[int]
    bbox_min: jnp.ndarray
    bbox_max: jnp.ndarray
    table_feature_dim: int = 2
    table_smooth: bool = False
    d_freqs: int = 4
    hidden_dim: int = 64
    density_dim: int = 16
    density_layers: int = 1
    color_layers: int = 2

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, d: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        d_emb = sinusoidal_emb(d, self.d_freqs)
        out = MultiresHashTableEncoding(
            table_sizes=self.table_sizes,
            grid_sizes=self.grid_sizes,
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
            feature_dim=self.table_feature_dim,
            smooth=self.table_smooth,
        )(x)
        for _ in range(self.density_layers):
            out = nn.relu(nn.Dense(self.hidden_dim)(out))
        out = nn.Dense(self.density_dim)(out)
        density = jnp.exp(out[:, :1])
        out = jnp.concatenate([d_emb, out], axis=1)
        for _ in range(self.color_layers):
            out = nn.relu(nn.Dense(self.hidden_dim)(out))
        color = nn.tanh(nn.Dense(3)(out))
        return density, color, {}


class InstantNGPRefNERFModel(RefNERFBase):
    """
    A NeRF model that utilizes a multilevel hash table.
    """

    table_sizes: List[int]
    grid_sizes: List[int]
    bbox_min: jnp.ndarray
    bbox_max: jnp.ndarray
    table_feature_dim: int = 2
    d_freqs: int = 4
    hidden_dim: int = 64
    density_dim: int = 16
    density_layers: int = 1
    color_layers: int = 2

    def spatial_block(self, x: jnp.ndarray) -> jnp.ndarray:
        x = MultiresHashTableEncoding(
            table_sizes=self.table_sizes,
            grid_sizes=self.grid_sizes,
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
            feature_dim=self.table_feature_dim,
            smooth=True,
        )(x)
        for _ in range(self.density_layers):
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.density_dim)(x)

    def directional_block(self, x: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.color_layers):
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(3)(x)


class MultiresHashTableEncoding(nn.Module):
    """
    Encode real-valued spatial coordinates using a multiresolution hash table.
    """

    table_sizes: List[int]
    grid_sizes: List[int]
    bbox_min: jnp.ndarray
    bbox_max: jnp.ndarray
    feature_dim: int = 2
    smooth: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        results = []
        for table_size, grid_size in zip(self.table_sizes, self.grid_sizes):
            results.append(
                HashTableEncoding(
                    table_size=table_size,
                    grid_size=grid_size,
                    bbox_min=self.bbox_min,
                    bbox_max=self.bbox_max,
                    feature_dim=self.feature_dim,
                    smooth=self.smooth,
                )(x)
            )
        return jnp.concatenate(results, axis=1)


class HashTableEncoding(nn.Module):
    """
    Encode real-valued spatial coordinates using a hash table over a fixed-size
    grid of coordinates.
    """

    table_size: int
    grid_size: int
    bbox_min: jnp.ndarray
    bbox_max: jnp.ndarray
    feature_dim: int = 2
    smooth: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute (interpolated) table entries for the coordinates.
        """
        frac = jnp.clip(
            (x - self.bbox_min) / (self.bbox_max - self.bbox_min), a_min=0, a_max=1
        )
        if self.smooth:
            # Shift by half a grid cell so that grid boundaries do not coincide
            # at different levels, since boundaries have zero derivatives.
            fractional_index = 0.5 + (self.grid_size - 2) * frac
        else:
            fractional_index = (self.grid_size - 1) * frac
        floored = jnp.floor(fractional_index)

        # Avoid out-of-bounds when adding 1 to floor(x) to get ceil(x).
        floored = jnp.clip(floored, a_max=self.grid_size - 2)

        ceil_frac = fractional_index - floored
        if self.smooth:
            ceil_frac = (ceil_frac ** 2) * (3 - 2 * ceil_frac)

        floored = floored.astype(jnp.uint32)

        all_coords = []
        all_weights = []
        for x_offset in [0, 1]:
            for y_offset in [0, 1]:
                for z_offset in [0, 1]:
                    offset = jnp.array(
                        [x_offset, y_offset, z_offset], dtype=floored.dtype
                    )
                    all_coords.append(floored + offset)
                    all_weights.append(
                        jnp.prod(
                            1
                            + (2 * ceil_frac - 1) * offset.astype(ceil_frac.dtype)
                            - ceil_frac,
                            axis=-1,
                            keepdims=True,
                        )
                    )

        if self.grid_size ** 3 > self.table_size:
            table = self.param(
                "table",
                lambda key: 1e-4
                * (
                    jax.random.uniform(key, (self.table_size, self.feature_dim)) * 2 - 1
                ),
            )
            all_lookup_results = jnp.concatenate(all_weights) * hash_table_lookup(
                table, jnp.concatenate(all_coords)
            )
        else:
            table = self.param(
                "table",
                lambda key: 1e-4
                * (
                    jax.random.uniform(key, (self.grid_size ** 3, self.feature_dim)) * 2
                    - 1
                ),
            )
            coords = jnp.concatenate(all_coords)
            indices = coords[:, 0] + self.grid_size * (
                coords[:, 1] + self.grid_size * coords[:, 2]
            )
            all_lookup_results = (
                jnp.concatenate(all_weights) * jnp.array(table)[indices]
            )

        return jnp.sum(
            all_lookup_results.reshape([8, x.shape[0], self.feature_dim]), axis=0
        )


def hash_table_lookup(table: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Lookup the integer coordinates in a hash table.

    :param table: a [T x F] table of T entries.
    :param coords: an [N x 3] batch of 3D integer coordinates.
    :return: an [N x F] batch of table lookup results.
    """
    coords = coords.astype(jnp.uint32)
    # Decorrelate the dimensions with a linear congruential permutation.
    indices = (
        coords[:, 0] ^ (19_349_663 * coords[:, 1]) ^ (83_492_791 * coords[:, 2])
    ) % table.shape[0]
    return jnp.array(table)[indices]
