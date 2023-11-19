#!/usr/bin/env python
#
# Replay existing session and convert output to format used by instant-ngp
#
# Use output with: https://github.com/NVlabs/instant-ngp

import argparse
import cv2
import json
import os
import shutil
import math
import numpy as np
import open3d as o3d


parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to folder with session to process")

args = parser.parse_args()

TRANSFORM_CAM = np.array([
 [1,0,0,0],
 [0,-1,0,0],
 [0,0,-1,0],
 [0,0,0,1],
])

def closestPointBetweenTwoLines(oa, da, ob, db):
    normal = np.cross(da, db)
    denom = np.linalg.norm(normal)**2
    t = ob - oa
    ta = np.linalg.det([t, db, normal]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, normal]) / (denom + 1e-10)
    if ta > 0: ta = 0
    if tb > 0: tb = 0
    return ((oa + ta * da + ob + tb * db) * 0.5, denom)


def resizeToUnitCube(frames, i_train, i_eval):
    weight = 0.0
    centerPos = np.array([0.0, 0.0, 0.0])
    for i in i_train:
        f = frames[i]
        mf = f["transform_matrix"][0:3,:]
        for j in i_train:
            g = frames[j]
            mg = g["transform_matrix"][0:3,:]
            p, w = closestPointBetweenTwoLines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.00001:
                centerPos += p * w
                weight += w
    if weight > 0.0: centerPos /= weight # [-1.8993619047588872, 0.503686679851153, 0.43740777174572115]

    scale = 0.

    for i, f in enumerate(frames):
        f["transform_matrix"][0:3,3] -= centerPos
        if i in i_train:
            scale += np.linalg.norm(f["transform_matrix"][0:3,3]) # 326.8

    scale = 4.0 / (scale / len(i_train)) # 2.264


    for f in frames: f["transform_matrix"][0:3,3] *= scale

    return frames, centerPos, scale


if __name__ == "__main__":
    input_path = args.input

    long_transformations = json.load(open(os.path.join(input_path,  "transformations.json")))


    with open(os.path.join(args.input, "test.txt")) as f:
        lines = f.readlines()[0]
    i_eval = [int(num) for num in lines.split(" ")] 
    

    long_pose_path = os.path.join(input_path, "pose")
    long_pose_files = list(sorted(os.listdir(long_pose_path), key=lambda x: int(x.split(".")[0])))
    long_frames = long_transformations["frames"]
    
    i_all = np.arange(len(long_pose_files))
    i_train = np.setdiff1d(i_all, i_eval)
    i_train = i_train.tolist()


    new_long_frames = []
    for i, pose in enumerate(long_pose_files):
        pose = np.loadtxt(os.path.join(long_pose_path, pose)).reshape(4, 4)
        pose = np.matmul(pose, TRANSFORM_CAM)
        frame = long_frames[i]    
        frame["transform_matrix"] = pose
        new_long_frames.append(frame)

    
    new_long_frames, centerPos, scale = resizeToUnitCube(new_long_frames, i_train, i_eval)
    
    for f in new_long_frames: f["transform_matrix"] = f["transform_matrix"].tolist()


    long_transformations["frames"] = new_long_frames
    print(centerPos, scale)
    long_transformations["center"] = centerPos.tolist()
    long_transformations["scale"] = scale

    
    with open(os.path.join(input_path,  "transformations_train.json"), "w") as outFile:
        json.dump(long_transformations, outFile, indent=2)

    