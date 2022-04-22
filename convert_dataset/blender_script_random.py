"""
Save random views of objects in Blender as a NERF dataset. This is similar to
blender_script.py, but uses random views rather than animation views.
"""

import json
import math
import os

import bpy
from mathutils import Vector
from mathutils.noise import random_unit_vector

NUM_FRAMES = 100
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


def scene_center():
    bbox = scene_bbox()
    return (Vector(bbox["min"]) + Vector(bbox["max"])) / 2


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
backup_matrix = scene.camera.matrix_world.copy()
camera_dist = (backup_matrix.to_translation() - scene_center()).length
backup_path = scene.render.filepath
backup_format = scene.render.image_settings.file_format
try:
    scene.render.image_settings.file_format = "PNG"
    for i in range(NUM_FRAMES):
        scene.render.filepath = os.path.join(OUTPUT_DIR, f"{i:05}")

        x_fov, y_fov = scene_fov()

        direction = random_unit_vector()
        camera_pos = scene_center() - camera_dist * direction
        scene.camera.location = camera_pos

        # https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
        rot_quat = direction.to_track_quat("-Z", "Y")
        scene.camera.rotation_euler = rot_quat.to_euler()

        bpy.context.view_layer.update()
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
    scene.camera.matrix_world = backup_matrix
    bpy.context.view_layer.update()
    scene.render.filepath = backup_path
    scene.render.image_settings.file_format = backup_format
