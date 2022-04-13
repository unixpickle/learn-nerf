"""
Convert a blender dataset from the original NeRF repo into the format used by
this repository.
"""

import argparse
import json
import math
import os
import shutil

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        raise FileExistsError(f"output path exists: {args.output_dir}")
    os.mkdir(args.output_dir)

    json_path = os.path.join(args.input_dir, f"transforms_{args.split}.json")
    with open(json_path, "r") as f:
        info = json.load(f)

    x_fov = info["camera_angle_x"]
    for i, frame in enumerate(info["frames"]):
        img_path = os.path.join(args.input_dir, frame["file_path"] + ".png")
        img_width, img_height = Image.open(img_path).size

        matrix = np.array(frame["transform_matrix"])
        origin = matrix[:3, -1]
        rot = matrix[:3, :3]
        x = rot @ np.array([1.0, 0.0, 0.0])
        y = rot @ np.array([0.0, -1.0, 0.0])
        z = rot @ np.array([0.0, 0.0, -1.0])
        y_fov = 2 * math.atan(math.tan(x_fov / 2) * img_height / img_width)

        out_base = os.path.join(args.output_dir, f"{i:04}")
        with open(out_base + ".json", "w") as f:
            json.dump(
                dict(
                    origin=origin.tolist(),
                    x_fov=x_fov,
                    y_fov=y_fov,
                    x=x.tolist(),
                    y=y.tolist(),
                    z=z.tolist(),
                ),
                f,
            )
        shutil.copyfile(img_path, out_base + ".png")

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(dict(min=[-1.0] * 3, max=[1.0] * 3), f)


if __name__ == "__main__":
    main()
