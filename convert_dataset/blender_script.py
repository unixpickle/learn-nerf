"""
Save Blender animations as a NERF dataset. Each frame of the animation is
saved as an image with an accompanying metadata file.

To use this script, set the OUTPUT_DIR variable near the top of the file
to point to a directory where the dataset should be saved, then run the
script inside of the "Scripting" tab in Blender.
"""

import json
import math
import os

import bpy

OUTPUT_DIR = None
assert OUTPUT_DIR is not None, "must set OUTPUT_DIR"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def scene_bbox():
    large = 100000.0
    bbox_min = (large,) * 3
    bbox_max = (-large,) * 3
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Camera, bpy.types.Light)):
            continue
        for coord in obj.bound_box:
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    return dict(min=bbox_min, max=bbox_max)


def scene_fov():
    x_fov = scene.camera.data.angle_x
    y_fov = scene.camera.data.angle_y
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    if scene.camera.data.angle == x_fov:
        y_fov = 2 * math.atan(math.tan(x_fov / 2) * height / width)
    else:
        x_fov = 2 * math.atan(math.tan(y_fov / 2) * width / height)
    return x_fov, y_fov


with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(scene_bbox(), f)

scene = bpy.context.scene
backup_path = scene.render.filepath
backup_format = scene.render.image_settings.file_format
try:
    scene.render.image_settings.file_format = "PNG"
    for i, frame in enumerate(range(scene.frame_start, scene.frame_end)):
        scene.frame_set(frame)
        scene.render.filepath = os.path.join(OUTPUT_DIR, f"{i:05}")

        x_fov, y_fov = scene_fov()
        matrix = scene.camera.matrix_world
        with open(scene.render.filepath + ".json", "w") as f:
            json.dump(
                dict(
                    origin=list(matrix.col[3])[:3],
                    x_fov=x_fov,
                    y_fov=y_fov,
                    x=list(matrix.col[0])[:3],
                    y=list(-matrix.col[1])[:3],
                    z=list(-matrix.col[2])[:3],
                ),
                f,
            )

        bpy.ops.render.render(write_still=True)
finally:
    scene.render.filepath = backup_path
    scene.render.image_settings.file_format = backup_format
