"""
Primitives and helpers for Ref-NeRF: https://arxiv.org/abs/2112.03907.
"""

from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .model import ModelBase

HARMONIC_COUNTS = [1, 3, 5, 7, 9, 11, 13, 15]
REF_NERF_OUT_DIM = 9


class RefNERFBase(ModelBase):
    """
    A NeRF model that utilizes a multilevel hash table.
    """

    sh_degree: int

    def spatial_block(self, x: jnp.ndarray) -> jnp.ndarray:
        _ = x
        raise NotImplementedError

    def directional_block(self, x: jnp.ndarray) -> jnp.ndarray:
        _ = x
        raise NotImplementedError

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, d: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        def spatial_fn(x):
            out = self.spatial_block(x)
            return out[:, 0].sum(), out

        real_normal, spatial_out = jax.grad(spatial_fn, has_aux=True)(x)
        real_normal = real_normal / (
            jnp.linalg.norm(real_normal, axis=-1, keepdims=True) + 1e-8
        )

        density, diffuse_color, spectral, roughness, normal, bottleneck = jnp.split(
            spatial_out, np.cumsum([1, 3, 1, 1, 3]).tolist(), axis=-1
        )
        density = jnp.exp(density)
        diffuse_color = nn.sigmoid(diffuse_color)
        spectral = nn.sigmoid(spectral)
        roughness = nn.softplus(roughness)
        normal = normal / (jnp.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)

        reflection = d - 2 * normal * jnp.sum(d * normal, axis=-1, keepdims=True)
        reflection_enc = integrated_directional_encoding(
            self.sh_degree, reflection, density
        )
        normal_dot = jnp.sum(-d * normal, axis=-1, keepdims=True)
        dir_input = jnp.concatenate([bottleneck, reflection_enc, normal_dot], axis=1)
        dir_output = self.directional_block(dir_input)
        spectral_color = nn.sigmoid(dir_output)

        full_color = (
            linear_rgb_to_srgb(
                spectral_color * spectral + diffuse_color * (1 - spectral)
            )
            * 2
            - 1
        )
        aux_losses = dict(
            normal_mse=jnp.sum((normal - real_normal) ** 2, axis=-1),
            neg_normal=jnp.maximum(0.0, jnp.sum(normal * d, axis=-1)) ** 2,
        )

        return density, full_color, aux_losses


def linear_rgb_to_srgb(colors: jnp.ndarray):
    """
    Perform Gamma compression to convert linear RGB colors to sRGB.
    """
    return jnp.where(
        colors <= 0.0031308, 12.92 * colors, 1.055 * (colors ** (1 / 2.4)) - 0.055
    )


def integrated_directional_encoding(
    sh_degree: int, coords: jnp.ndarray, roughness: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute the integrated directional encoding for the 3D coordinates given
    the corresponding roughness parameter. Intuitively, higher roughness
    parameters ignore the coordinates more.

    :param sh_degree: the degree of the harmonics. Should be in range [1, 8].
    :param coords: an [N x 3] array of normalized coordinates.
    :param roughness: an [N x 1] array of densities.
    :return: an [N x D] array of integrated directional encodings.
    """
    assert len(roughness.shape) == 2 and roughness.shape[1] == 1
    assert len(coords.shape) == 2 and coords.shape[1] == 3

    levels = jnp.array(
        [x for i, y in enumerate(HARMONIC_COUNTS[:sh_degree]) for x in [i] * y],
        dtype=roughness.dtype,
    )
    attenuation = jnp.exp(-roughness * (levels * (levels + 1)) / 2)
    harmonics = spherical_harmonic(sh_degree, coords)
    return harmonics * attenuation


def spherical_harmonic(sh_degree: int, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the spherical harmonic encoding of the 3D coordinates.

    :param sh_degree: the degree of the harmonics. Should be in range [1, 8].
    :param coords: an [N x 3] array of normalized coordinates.
    :return: an [N x D] array of harmonic encodings.
    """
    assert sh_degree >= 1 and sh_degree <= 8
    # Based on https://github.com/NVlabs/tiny-cuda-nn/blob/8575542682cb67cddfc748cc3d3cfc12593799aa/include/tiny-cuda-nn/encodings/spherical_harmonics.h#L76
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    xy = x * y
    xz = x * z
    yz = y * z
    x2 = x * x
    y2 = y * y
    z2 = z * z
    x4 = x2 * x2
    y4 = y2 * y2
    z4 = z2 * z2
    x6 = x4 * x2
    y6 = y4 * y2
    z6 = z4 * z2

    out = [None] * 64

    def populate():
        out[0] = jnp.broadcast_to(jnp.array(0.28209479177387814), x.shape)
        if sh_degree <= 1:
            return
        out[1] = -0.48860251190291987 * y
        out[2] = 0.48860251190291987 * z
        out[3] = -0.48860251190291987 * x
        if sh_degree <= 2:
            return
        out[4] = 1.0925484305920792 * xy
        out[5] = -1.0925484305920792 * yz
        out[6] = 0.94617469575755997 * z2 - 0.31539156525251999
        out[7] = -1.0925484305920792 * xz
        out[8] = 0.54627421529603959 * x2 - 0.54627421529603959 * y2
        if sh_degree <= 3:
            return
        out[9] = 0.59004358992664352 * y * (-3.0 * x2 + y2)
        out[10] = 2.8906114426405538 * xy * z
        out[11] = 0.45704579946446572 * y * (1.0 - 5.0 * z2)
        out[12] = 0.3731763325901154 * z * (5.0 * z2 - 3.0)
        out[13] = 0.45704579946446572 * x * (1.0 - 5.0 * z2)
        out[14] = 1.4453057213202769 * z * (x2 - y2)
        out[15] = 0.59004358992664352 * x * (-x2 + 3.0 * y2)
        if sh_degree <= 4:
            return
        out[16] = 2.5033429417967046 * xy * (x2 - y2)
        out[17] = 1.7701307697799304 * yz * (-3.0 * x2 + y2)
        out[18] = 0.94617469575756008 * xy * (7.0 * z2 - 1.0)
        out[19] = 0.66904654355728921 * yz * (3.0 - 7.0 * z2)
        out[20] = (
            -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293
        )
        out[21] = 0.66904654355728921 * xz * (3.0 - 7.0 * z2)
        out[22] = 0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0)
        out[23] = 1.7701307697799304 * xz * (-x2 + 3.0 * y2)
        out[24] = (
            -3.7550144126950569 * x2 * y2
            + 0.62583573544917614 * x4
            + 0.62583573544917614 * y4
        )
        if sh_degree <= 5:
            return
        out[25] = 0.65638205684017015 * y * (10.0 * x2 * y2 - 5.0 * x4 - y4)
        out[26] = 8.3026492595241645 * xy * z * (x2 - y2)
        out[27] = -0.48923829943525038 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0)
        out[28] = 4.7935367849733241 * xy * z * (3.0 * z2 - 1.0)
        out[29] = 0.45294665119569694 * y * (14.0 * z2 - 21.0 * z4 - 1.0)
        out[30] = 0.1169503224534236 * z * (-70.0 * z2 + 63.0 * z4 + 15.0)
        out[31] = 0.45294665119569694 * x * (14.0 * z2 - 21.0 * z4 - 1.0)
        out[32] = 2.3967683924866621 * z * (x2 - y2) * (3.0 * z2 - 1.0)
        out[33] = -0.48923829943525038 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0)
        out[34] = 2.0756623148810411 * z * (-6.0 * x2 * y2 + x4 + y4)
        out[35] = 0.65638205684017015 * x * (10.0 * x2 * y2 - x4 - 5.0 * y4)
        if sh_degree <= 6:
            return
        out[36] = 1.3663682103838286 * xy * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4)
        out[37] = 2.3666191622317521 * yz * (10.0 * x2 * y2 - 5.0 * x4 - y4)
        out[38] = 2.0182596029148963 * xy * (x2 - y2) * (11.0 * z2 - 1.0)
        out[39] = -0.92120525951492349 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0)
        out[40] = 0.92120525951492349 * xy * (-18.0 * z2 + 33.0 * z4 + 1.0)
        out[41] = 0.58262136251873131 * yz * (30.0 * z2 - 33.0 * z4 - 5.0)
        out[42] = (
            6.6747662381009842 * z2
            - 20.024298714302954 * z4
            + 14.684485723822165 * z6
            - 0.31784601133814211
        )
        out[43] = 0.58262136251873131 * xz * (30.0 * z2 - 33.0 * z4 - 5.0)
        out[44] = (
            0.46060262975746175
            * (x2 - y2)
            * (11.0 * z2 * (3.0 * z2 - 1.0) - 7.0 * z2 + 1.0)
        )
        out[45] = -0.92120525951492349 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0)
        out[46] = 0.50456490072872406 * (11.0 * z2 - 1.0) * (-6.0 * x2 * y2 + x4 + y4)
        out[47] = 2.3666191622317521 * xz * (10.0 * x2 * y2 - x4 - 5.0 * y4)
        out[48] = (
            10.247761577878714 * x2 * y4
            - 10.247761577878714 * x4 * y2
            + 0.6831841051919143 * x6
            - 0.6831841051919143 * y6
        )
        if sh_degree <= 7:
            return
        out[49] = (
            0.70716273252459627 * y * (-21.0 * x2 * y4 + 35.0 * x4 * y2 - 7.0 * x6 + y6)
        )
        out[50] = 5.2919213236038001 * xy * z * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4)
        out[51] = (
            -0.51891557872026028
            * y
            * (13.0 * z2 - 1.0)
            * (-10.0 * x2 * y2 + 5.0 * x4 + y4)
        )
        out[52] = 4.1513246297620823 * xy * z * (x2 - y2) * (13.0 * z2 - 3.0)
        out[53] = (
            -0.15645893386229404
            * y
            * (3.0 * x2 - y2)
            * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0)
        )
        out[54] = 0.44253269244498261 * xy * z * (-110.0 * z2 + 143.0 * z4 + 15.0)
        out[55] = (
            0.090331607582517306 * y * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0)
        )
        out[56] = (
            0.068284276912004949 * z * (315.0 * z2 - 693.0 * z4 + 429.0 * z6 - 35.0)
        )
        out[57] = (
            0.090331607582517306 * x * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0)
        )
        out[58] = (
            0.07375544874083044
            * z
            * (x2 - y2)
            * (143.0 * z2 * (3.0 * z2 - 1.0) - 187.0 * z2 + 45.0)
        )
        out[59] = (
            -0.15645893386229404
            * x
            * (x2 - 3.0 * y2)
            * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0)
        )
        out[60] = (
            1.0378311574405206 * z * (13.0 * z2 - 3.0) * (-6.0 * x2 * y2 + x4 + y4)
        )
        out[61] = (
            -0.51891557872026028
            * x
            * (13.0 * z2 - 1.0)
            * (-10.0 * x2 * y2 + x4 + 5.0 * y4)
        )
        out[62] = 2.6459606618019 * z * (15.0 * x2 * y4 - 15.0 * x4 * y2 + x6 - y6)
        out[63] = (
            0.70716273252459627 * x * (-35.0 * x2 * y4 + 21.0 * x4 * y2 - x6 + 7.0 * y6)
        )

    populate()
    return jnp.stack([x for x in out if x is not None], axis=1)
