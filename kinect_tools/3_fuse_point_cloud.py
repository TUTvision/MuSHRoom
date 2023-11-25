# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import open3d as o3d
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to folder with session to process")
parser.add_argument(
    "--pose_scale", help="use training data or all data to rescale pose",
)
args = parser.parse_args()

if __name__ == "__main__":
    input_path = os.path.join(args.input, "PointCloud")
    print(args.input)
    # input_path = args.input
    point_path = list(sorted(os.listdir(input_path), key=lambda x: int(x.split(".")[0])))
    frame_names = [num.split(".")[0] for num in point_path]

    num = len(point_path)

    points_list = []
    colors_list = []
    normals_list = []

    """
    read a line of numbers from test.txt
    """
    i_all = np.arange(num)
    
    if args.pose_scale == "train":
        with open(os.path.join(args.input, "test.txt")) as f:
            lines = f.readlines()
        i_eval_name = [num.split("\n")[0] for num in lines]
        i_eval = [frame_names.index(name) for name in i_eval_name]
    
    i_train = np.setdiff1d(i_all, i_eval)

    
    
    for i in range(num):
        if args.pose_scale == "train":
            if i in i_eval:
                continue
        
        pcd = o3d.io.read_point_cloud(os.path.join(input_path, str(i) + ".ply"))

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
    o3d.io.write_point_cloud(os.path.join(args.input, "pointcloud_{}.ply".format(args.pose_scale)), cloud)
