"""
Decode an LLFF dataset.
"""

import argparse
import json
import os
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    img_dir = os.path.join(args.input_dir, "images")
    img_paths = [
        os.path.join(img_dir, x)
        for x in sorted(os.listdir(img_dir))
        if os.path.splitext(x)[1].lower() in [".jpg", ".jpeg", ".png"]
    ]

    pose_path = os.path.join(args.input_dir, "poses_bounds.npy")
    pose_bounds = np.load(pose_path)
    assert len(pose_bounds) == len(img_paths), "image count must match pose count"

    os.makedirs(args.output_dir, exist_ok=True)
    bbox_min, bbox_max = None, None
    with ThreadPool(8) as p:
        for local_min, local_max in tqdm(
            p.imap_unordered(
                partial(process_img, args.output_dir, args.factor),
                enumerate(zip(pose_bounds, img_paths)),
            )
        ):
            if bbox_min is None:
                bbox_min, bbox_max = local_min, local_max
            else:
                bbox_min = np.minimum(bbox_min, local_min)
                bbox_max = np.maximum(bbox_max, local_max)

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        bbox_info = dict(min=bbox_min.tolist(), max=bbox_max.tolist())
        json.dump(bbox_info, f)


def process_img(
    output_dir: str, factor: float, item: Tuple[int, Tuple[np.ndarray, str]]
):
    i, (pose_bound, img_path) = item
    info = pose_bound[:15].reshape([3, 5])
    x, y, z, pos, hwf = info.T
    h, w, focal = hwf
    z_near, z_far = pose_bound[15:]
    _ = z_near

    # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L250
    x, y, z = y, -x, z

    # Same changes as in blender.py
    y = -y
    z = -z

    local_min = pos - z_far
    local_max = pos + z_far

    info = dict(
        origin=pos.tolist(),
        x_fov=float(2 * np.arctan(w / (2 * focal))),
        y_fov=float(2 * np.arctan(h / (2 * focal))),
        x=x.tolist(),
        y=y.tolist(),
        z=z.tolist(),
    )
    with open(os.path.join(output_dir, f"{i:05}.json"), "w") as f:
        json.dump(info, f)
    img_path_out = os.path.join(output_dir, f"{i:05}.png")
    new_img = Image.open(img_path).convert("RGB")
    if factor != 1.0:
        old_w, old_h = new_img.size
        new_img = new_img.resize((round(old_w * factor), round(old_h * factor)))
    new_img.save(img_path_out)

    return local_min, local_max


if __name__ == "__main__":
    main()
