import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL

from torchvision import transforms

parser = argparse.ArgumentParser(description="preprocess scannet dataset to sdfstudio dataset")

parser.add_argument("--input_path", dest="input_path", help="path to scannet scene")
parser.set_defaults(im_name="NONE")

args = parser.parse_args()

long_input_path = os.path.join(args.input_path, "long_capture", "sdf_dataset_all", "meta_data.json")
short_input_path = os.path.join(args.input_path, "short_capture")

short_input_path = Path(short_input_path) 

json_file = json.load(open(short_input_path/ "transformations_colmap.json"))




color_paths = []
poses = []

TRANSFORM_CAM = np.array([
    [1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,1],
    ])

for frame in json_file["frames"]:
    color_paths.append(frame["file_path"])
    c2w = np.array(frame["transform_matrix"])
    c2w = np.matmul(c2w, TRANSFORM_CAM)
    
    poses.append(c2w)


poses = np.array(poses)


cx, cy, fx, fy = json_file["cx"], json_file["cy"], json_file["fl_x"], json_file["fl_y"]


K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

long_transformation = json.load(open(long_input_path))
center = long_transformation["center"]
scale = long_transformation["scale"]
  
poses[:, :3, 3] -= center
poses[:, :3, 3] *= scale

out_index = 0
frames = []



for i, image_path in enumerate(color_paths):
    
    image_name = os.path.basename(image_path)
    
    pose = poses[i]

    frame = {
        "rgb_path": "images/" + image_name,
        "camtoworld": pose.tolist(),
        "intrinsics": K.tolist(),
        "sensor_depth_path": "depth/" + image_name,
    }
    frames.append(frame)
    out_index += 1

# scene bbox for the scannet scene
scene_box = {
    "aabb": [[-1, -1, -1], [1, 1, 1]],
    "near": 0.00,
    "far": 10,
    "radius": 1.0,
    "collider_type": "box",
}

# meta data
output_data = {
    "scale": scale,
    "center": center,
    "camera_model": "OPENCV",
    "height": 720,
    "width": 1280,
    "has_mono_prior": False,
    "has_sensor_depth": False,
    "pairs": None,
    "scene_box": scene_box,
}

output_data["frames"] = frames

# save as json
with open(short_input_path / "meta_data_align.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)




