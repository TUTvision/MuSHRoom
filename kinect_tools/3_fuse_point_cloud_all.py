
import open3d as o3d
import os
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to folder with session to process")
args = parser.parse_args()

input = args.input

input_path = os.path.join(input, "PointCloud")
pose = os.path.join(input, "pose")



points_list = []
colors_list = []
normals_list = []

"""
read a line of numbers from test.txt
"""
TRANSFORM_CAM = np.array([
    [1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,1],
    ])

frames = json.load(open(os.path.join(input, "transformations_colmap.json")))["frames"]
for i, frame in enumerate(frames):
    print(i)

    name = frame["file_path"].split("/")[-1].split(".")[0]
    pcd = o3d.io.read_point_cloud(os.path.join(input_path, name + ".ply"))
    original_pose = np.loadtxt(os.path.join(pose, name + ".txt")).reshape(4, 4)
    # pcd = pcd.transform(original_pose)

    # pcd = pcd.transform(np.linalg.inv(original_pose))

    new_pose = frame["transform_matrix"]
    new_pose = np.matmul(np.array(new_pose), TRANSFORM_CAM) 
    pcd = pcd.transform(new_pose) 

    
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    points = pcd.points
    color = pcd.colors
    normal = pcd.normals
    points_list.append(np.asarray(points))
    colors_list.append(np.asarray(color))
    normals_list.append(np.asarray(normal))

cloud = o3d.geometry.PointCloud()

points = o3d.utility.Vector3dVector(np.vstack(points_list))
colors = o3d.utility.Vector3dVector(np.vstack(colors_list))  
normals = o3d.utility.Vector3dVector(np.vstack(normals_list))

cloud.points = points
cloud.colors = colors
cloud.normals = normals
cloud = cloud.voxel_down_sample(voxel_size=0.01)
o3d.io.write_point_cloud(os.path.join(input, "pointcloud_colmap.ply"), cloud)