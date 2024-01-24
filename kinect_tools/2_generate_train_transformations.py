import os
import numpy as np
import json
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to folder with session to process")
args = parser.parse_args()


input_path = args.input
images_folder = os.path.join(input_path, "images")
depth_folder = os.path.join(input_path, "depth")
pose_folder = os.path.join(input_path, "pose")
intrinsics_folder = os.path.join(input_path, "intrinsic")
intrinsics_color = np.loadtxt(os.path.join(intrinsics_folder, "intrinsic_color.txt")).reshape(4, 4)
pose_list = list(sorted(os.listdir(pose_folder), key=lambda x: int(x.split(".")[0])))
name_list = [name.split(".")[0] for name in pose_list]

TRANSFORM_CAM = np.array([
 [1,0,0,0],
 [0,-1,0,0],
 [0,0,-1,0],
 [0,0,0,1],
])


metadata = {
    "fl_x" : intrinsics_color[0, 0],
    "fl_y" : intrinsics_color[1, 1],
    "cx" : intrinsics_color[0, 2],
    "cy" : intrinsics_color[1, 2],
    "w" : int(intrinsics_color[0, 2])*2,
    "h" : int(intrinsics_color[1, 2])*2,
    "angle_x" : math.atan(int(intrinsics_color[0, 2])*2 / (intrinsics_color[0, 0] * 2)) * 2,
    "angle_y" : math.atan(int(intrinsics_color[1, 2])*2 / (intrinsics_color[1, 1] * 2)) * 2,
    "frames" : []
}

for name in name_list:
    metadata["frames"].append({
        "file_path" : os.path.join("images", name + ".png"),
        "transform_matrix" : np.matmul(np.loadtxt(os.path.join(pose_folder, name + ".txt")).reshape(4, 4), TRANSFORM_CAM).tolist(),
        "depth_file_path" : os.path.join("depth", name + ".png"),
    })

save_path = os.path.join(input_path, "transformations.json")
with open(save_path, 'w') as outfile:
    json.dump(metadata, outfile, indent=4)

