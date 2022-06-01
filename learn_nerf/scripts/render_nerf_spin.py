"""
Spin around the y axis from a fixed camera view.
"""

import math

import jax.numpy as jnp
import numpy as np
from learn_nerf.dataset import CameraView
from learn_nerf.scripts.render_nerf import RenderSession, argparser


def main():
    parser = argparser()
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("view_json", type=str)
    parser.add_argument("output_png", type=str)
    args = parser.parse_args()

    rs = RenderSession(args)

    view = CameraView.from_json(args.view_json)
    x, z = np.array(view.x_axis), np.array(view.camera_direction)

    for i in range(args.frames):
        theta = 2 * math.pi * i / args.frames
        sin, cos = math.sin(theta), math.cos(theta)
        view.x_axis, view.camera_direction = tuple(cos * x + sin * z), tuple(
            -sin * x + cos * z
        )
        rs.render_view(view)

    rs.save(args.output_png)


if __name__ == "__main__":
    main()
