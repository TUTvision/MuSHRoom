import argparse
import json
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import math

def main(args):
    """
    given data that follows the nerfstduio format such as the output from colmap or polycam,
    convert to a format that sdfstudio will ingest
    """
    # output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)
    long_input_dir = input_dir / "long_capture" / "sdf_dataset_all"
    short_input_dir = input_dir / "short_capture"


    """ load long sequence pose """
    long_cam_params = json.load(open(long_input_dir / "meta_data.json"))
    # === load camera intrinsics and poses ===
    scene_center = np.array(long_cam_params["scene_center"])
    scene_scale = long_cam_params["scene_scale"]






    """ load short sequence pose """
    short_cam_params = json.load(open(short_input_dir / "transformations_colmap.json"))
    # short_to_long_tfm = np.array(json.load(open(os.path.join(args.input_dir, "short_to_long_transformation_icp.json")))["gt_transformation"]).reshape(4, 4) 
    
    short_cam_intrinsics = []

    # === load camera intrinsics and poses ===

    short_frames = short_cam_params["frames"]
    short_poses = []
    short_image_paths = []
    for frame in short_frames:
        # load intrinsics from polycam
        if "fl_x" in frame:
            short_cam_intrinsics.append(
                np.array([[frame["fl_x"], 0, frame["cx"]], [0, frame["fl_y"], frame["cy"]], [0, 0, 1]])
            )
        name = frame["file_path"]
        if os.path.exists(os.path.join(short_input_dir, name[2:])) == False:
            print(os.path.join(short_input_dir, name[2:]))
            continue
        c2w = np.array(frame["transform_matrix"]).reshape(4, 4)
        c2w[0:3, 1:3] *= -1
        # c2w = np.matmul(short_to_long_tfm, c2w)
        short_poses.append(c2w)

        # load images
        file_path = Path(frame["file_path"])
        img_path =  "images/" + file_path.name
        # assert img_path.exists()
        short_image_paths.append(img_path)
    
    short_poses = np.array(short_poses)



    # === Normalize the scene ===
    if args.scene_type in ["indoor", "object"]:
        # Enlarge bbox by 1.05 for object scene and by 5.0 for indoor scene
        # TODO: Adaptively estimate `scene_scale_mult` based on depth-map or point-cloud prior
        if not args.scene_scale_mult:
            args.scene_scale_mult = 1.05 if args.scene_type == "object" else 5.0
        
        short_poses[:, :3, 3] -= scene_center
        short_poses[:, :3, 3] *= scene_scale
        # calculate scale matrix
        scale_mat = np.eye(4).astype(np.float32)
        scale_mat[:3, 3] -= scene_center
        scale_mat[:3] *= scene_scale
        scale_mat = np.linalg.inv(scale_mat)



    # === Construct the scene box ===
    if args.scene_type == "indoor":
        scene_box = {
            "aabb": [[-1, -1, -1], [1, 1, 1]],
            "near": 0.05,
            "far": 2.5,
            "radius": 1.0,
            "collider_type": "box",
        }
    elif args.scene_type == "object":
        scene_box = {
            "aabb": [[-1, -1, -1], [1, 1, 1]],
            "near": 0.05,
            "far": 2.0,
            "radius": 1.0,
            "collider_type": "near_far",
        }
    elif args.scene_type == "unbound":
        # TODO: case-by-case near far based on depth prior
        #  such as colmap sparse points or sensor depths
        scene_box = {
            "aabb": [min_vertices.tolist(), max_vertices.tolist()],
            "near": 0.05,
            "far": 2.5 * np.max(max_vertices - min_vertices),
            "radius": np.min(max_vertices - min_vertices) / 2.0,
            "collider_type": "box",
        }


    # === Construct the frames in the meta_data.json ===
    frames = []
    out_index = 0
    for idx, ( pose, image_path) in enumerate(tqdm(zip( short_poses, short_image_paths))):


        frame = {
            "rgb_path": image_path,
            "camtoworld": pose.tolist(),
            "intrinsics": short_cam_intrinsics[idx].tolist(),
        }


        frames.append(frame)
        out_index += 1

    # === Construct and export the metadata ===
    meta_data = {
        "camera_model": "OPENCV",
        "height": 994,
        "width": 738,
        "has_mono_prior": False,
        "has_sensor_depth": False,
        "has_foreground_mask": False,
        "pairs": None,
        "worldtogt": scale_mat.tolist(),
        "scene_box": scene_box,
        "frames": frames,
    }
    with open(short_input_dir / "meta_data_align.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="preprocess nerfstudio dataset to sdfstudio dataset, " "currently support colmap and polycam"
    )

    parser.add_argument("--input_dir", required=True, help="path to nerfstudio data directory")


    parser.add_argument(
        "--scene-type",
        dest="scene_type",
        required=True,
        choices=["indoor", "object", "unbound"],
        help="The scene will be normalized into a unit sphere when selecting indoor or object.",
    )
    parser.add_argument(
        "--scene-scale-mult",
        dest="scene_scale_mult",
        type=float,
        default=None,
        help="The bounding box of the scene is firstly calculated by the camera positions, "
        "then mutiply with scene_scale_mult",
    )



    args = parser.parse_args()

    main(args)
