import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description="preprocess scannet dataset to sdfstudio dataset")

parser.add_argument("--input_path", dest="input_path", help="path to scannet scene")
parser.set_defaults(im_name="NONE")

parser.add_argument("--output_path", dest="output_path", help="path to output")
parser.set_defaults(store_name="NONE")
parser.add_argument(
    "--type",
    dest="type",
    default="mono_prior",
    choices=["mono_prior", "sensor_depth"],
    help="mono_prior to use monocular prior, sensor_depth to use depth captured with a depth sensor (gt depth)",
)

parser.set_defaults(store_name="NONE")

args = parser.parse_args()

trans_totensor = transforms.Compose(
    [
        transforms.Resize([720, 1280], interpolation=PIL.Image.BILINEAR),
    ]
)

depth_trans_totensor = transforms.Compose(
    [
        transforms.Resize([720, 1280], interpolation=PIL.Image.NEAREST),
    ]
)

output_path = Path(args.output_path)  
input_path = Path(args.input_path) 

output_path.mkdir(parents=True, exist_ok=True)

# load color
json_file = json.load(open(input_path / "transformations_colmap.json"))
poses = []
color_paths = []
depth_paths = []

TRANSFORM_CAM = np.array([
    [1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,1],
    ])

for frame in json_file["frames"]:
    image_path = input_path / frame["file_path"]
    color_paths.append(image_path)
    depth_path = (input_path / (frame["depth_file_path"]) )

    depth_path = str(depth_path).replace("depth", "depth_complete_all")
    depth_paths.append(depth_path)
    
    c2w = np.array(frame["transform_matrix"])
    c2w = np.matmul( c2w, TRANSFORM_CAM)
    poses.append(c2w)
    
cx, cy, fx, fy = json_file["cx"], json_file["cy"], json_file["fl_x"], json_file["fl_y"]

poses = np.array(poses)
num_images = poses.shape[0]
i_all = np.arange(num_images)
index = i_all

min_vertices = poses[index][:, :3, 3].min(axis=0)
max_vertices = poses[index][:, :3, 3].max(axis=0)
center = (min_vertices + max_vertices) / 2.0

scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)


# we should normalize pose to unit cube
poses[:, :3, 3] -= center
poses[:, :3, 3] *= scale

# inverse normalization
scale_mat = np.eye(4).astype(np.float32)
scale_mat[:3, 3] -= center
scale_mat[:3] *= scale
scale_mat = np.linalg.inv(scale_mat)

# copy image
H, W = 720, 1280

K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

frames = []
out_index = 0
for idx, (pose, image_path, depth_path) in enumerate(zip(poses, color_paths, depth_paths)):

    target_image = output_path / f"{out_index:06d}_rgb.png"
    
    img = Image.open(image_path)
    img_tensor = trans_totensor(img)
    img_tensor.save(target_image)

    # load depth
    target_depth_image = output_path / f"{out_index:06d}_sensor_depth.png"
    
    depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000.0

    depth_PIL = Image.fromarray(depth)
    new_depth = depth_trans_totensor(depth_PIL)
    new_depth = np.asarray(new_depth)
    # scale depth as we normalize the scene to unit box
    new_depth = new_depth * scale
    plt.imsave(target_depth_image, new_depth, cmap="viridis")
    np.save(str(target_depth_image).replace(".png", ".npy"), new_depth)

    rgb_path = str(target_image.relative_to(output_path))
    frame = {
        "rgb_path": rgb_path,
        "camtoworld": pose.tolist(),
        "intrinsics": K.tolist(),
        "sensor_depth_path": rgb_path.replace("_rgb.png", "_sensor_depth.npy"),
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
    "center": center.tolist(),
    "camera_model": "OPENCV",
    "height": 720,
    "width": 1280,
    "has_mono_prior": False,
    "has_sensor_depth": True,
    "pairs": None,
    "worldtogt": scale_mat.tolist(),
    "scene_box": scene_box,
}

output_data["frames"] = frames

# save as json
with open(output_path / "meta_data.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)
