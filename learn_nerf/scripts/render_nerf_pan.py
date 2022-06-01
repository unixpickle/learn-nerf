"""
Render a panning view of a NeRF model.
"""

import math

import jax.numpy as jnp
import numpy as np
from learn_nerf.dataset import CameraView
from learn_nerf.scripts.render_nerf import RenderSession, argparser


def main():
    parser = argparser()
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--random_axis", action="store_true")
    parser.add_argument("output_png", type=str)
    args = parser.parse_args()

    rs = RenderSession(args)

    scale = float(jnp.linalg.norm(rs.renderer.bbox_min - rs.renderer.bbox_max))
    center = np.array((rs.renderer.bbox_min + rs.renderer.bbox_max) / 2)

    rot_axis = np.array([0.0, 0.0, -1.0])
    rot_basis_1 = np.array([1.0, 0.0, 0.0])
    if args.random_axis:
        rot_axis = np.random.normal(size=(3,))
        rot_axis /= np.linalg.norm(rot_axis)
        rot_basis_1 = np.array([-rot_axis[2], 0.0, rot_axis[0]])
        rot_basis_1 /= np.linalg.norm(rot_basis_1)
    rot_basis_2 = np.cross(rot_axis, rot_basis_1)

    for frame in range(args.frames):
        print(f"sampling frame {frame}...")
        theta = (frame / args.frames) * math.pi * 2
        direction = np.cos(theta) * rot_basis_1 + np.sin(theta) * rot_basis_2
        rs.render_view(
            CameraView(
                camera_direction=tuple(direction),
                camera_origin=tuple(-direction * scale * args.distance + center),
                x_axis=tuple(
                    np.cos(theta + np.pi / 2) * rot_basis_1
                    + np.sin(theta + np.pi / 2) * rot_basis_2
                ),
                y_axis=tuple(rot_axis),
                x_fov=60.0 * math.pi / 180,
                y_fov=60.0 * math.pi / 180,
            )
        )

    rs.save(args.output_png)


if __name__ == "__main__":
    main()
