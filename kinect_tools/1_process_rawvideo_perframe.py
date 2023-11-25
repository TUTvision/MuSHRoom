#!/usr/bin/env python
#
# Replay existing session and convert output to format used by instant-ngp
#
# Use output with: https://github.com/NVlabs/instant-ngp

import argparse
import spectacularAI
import cv2
import json
import os
import shutil
import math
import numpy as np
from spectacularAI.mapping import PointCloud
import open3d as o3d


parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to folder with session to process")
parser.add_argument("--output", help="Path to output folder")
parser.add_argument("--scale", help="Scene scale, exponent of 2", type=int, default=128)
parser.add_argument("--preview", help="Show latest primary image as a preview", action="store_true")
args = parser.parse_args()

# Globals
savedKeyFrames = {}
frameWidth = -1
frameHeight = -1
intrinsics = None

TRANSFORM_CAM = np.array([
 [1,0,0,0],
 [0,-1,0,0],
 [0,0,-1,0],
 [0,0,0,1],
])

TRANSFORM_WORLD = np.array([
 [0,1,0,0],
 [-1,0,0,0],
 [0,0,1,0],
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


def resizeToUnitCube(frames):
    weight = 0.0
    centerPos = np.array([0.0, 0.0, 0.0])
    for f in frames:
        mf = f["transform_matrix"][0:3,:]
        for g in frames:
            mg = g["transform_matrix"][0:3,:]
            p, w = closestPointBetweenTwoLines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.00001:
                centerPos += p * w
                weight += w
    if weight > 0.0: centerPos /= weight # [-1.8993619047588872, 0.503686679851153, 0.43740777174572115]

    scale = 0.
    for f in frames:
        f["transform_matrix"][0:3,3] -= centerPos
        scale += np.linalg.norm(f["transform_matrix"][0:3,3]) # 326.8

    scale = 4.0 / (scale / len(frames)) # 2.264
    for f in frames: f["transform_matrix"][0:3,3] *= scale
    return centerPos, scale


def sharpness(path):
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return cv2.Laplacian(img, cv2.CV_64F).var()

def getKeyFramePointCloud(keyFrame):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(keyFrame.pointCloud.getPositionData())

    if keyFrame.pointCloud.hasColors():
        
        colors = keyFrame.pointCloud.getRGB24Data() * 1./255
        cloud.colors = o3d.utility.Vector3dVector(colors)

    if keyFrame.pointCloud.hasNormals():
        cloud.normals = o3d.utility.Vector3dVector(keyFrame.pointCloud.getNormalData())

    
    # cloud.transform(poseCToW)

    return cloud


def onMappingOutput(output):
    global savedKeyFrames
    global frameWidth
    global frameHeight
    global intrinsics

    if not output.finalMap:
        # New frames, let's save the images to disk
        for frameId in output.updatedKeyFrames:
            keyFrame = output.map.keyFrames.get(frameId)
            if not keyFrame or savedKeyFrames.get(keyFrame):
                continue
            savedKeyFrames[keyFrame] = True
            frameSet = keyFrame.frameSet
            if not frameSet.rgbFrame or not frameSet.rgbFrame.image:
                continue

            if frameWidth < 0:
                frameWidth = frameSet.rgbFrame.image.getWidth()
                frameHeight = frameSet.rgbFrame.image.getHeight()

            undistortedFrame = frameSet.getUndistortedFrame(frameSet.rgbFrame)
            if intrinsics is None: intrinsics = undistortedFrame.cameraPose.camera.getIntrinsicMatrix()
            img = undistortedFrame.image.toArray()
            bgrImage = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            depthFrame = frameSet.getAlignedDepthFrame(undistortedFrame)
            depthFrame = depthFrame.image.toArray()

            RGBfileName = args.output + "/tmp_images/" + f'{frameId}' + ".png"
            DepthfileName = args.output + "/tmp_depth/" + f'{frameId}' + ".png"
            
            cv2.imwrite(RGBfileName, bgrImage)
            cv2.imwrite(DepthfileName, depthFrame)

            if args.preview:
                cv2.imshow("Frame", bgrImage)
                cv2.setWindowTitle("Frame", "Frame #{}".format(frameId))
                cv2.waitKey(1)
    else:
        # Final optimized poses
        frames = []
        index = 0

        up = np.zeros(3)
        for frameId in output.map.keyFrames:
            keyFrame = output.map.keyFrames.get(frameId)
            oldImgName = args.output + "/tmp_images/" + f'{frameId}' + ".png"
            newImgName = args.output + "/images/" + f'{index}' + ".png"
            os.rename(oldImgName, newImgName)
            oldImgName = args.output + "/tmp_depth/" + f'{frameId}' + ".png"
            newImgName = args.output + "/depth/" + f'{index}' + ".png"
            os.rename(oldImgName, newImgName)
            # cameraPose = keyFrame.frameSet.rgbFrame.cameraPose
            cameraPose = keyFrame.frameSet.rgbFrame.cameraPose

            # Converts Spectacular AI camera to coordinate system used by instant-ngp
            # cameraToWorld = np.matmul(TRANSFORM_WORLD, np.matmul(cameraPose.getCameraToWorldMatrix(), TRANSFORM_CAM))
            camera_rgb = cameraPose.getCameraToWorldMatrix()
            cameraToWorld = np.matmul(camera_rgb, TRANSFORM_CAM)
            
            txt_cameraToWorld = camera_rgb.tolist()
            txtfile = args.output + "/pose"
            f2 = open(os.path.join(txtfile, f'{index}' + ".txt"), "w")
            for item1 in txt_cameraToWorld:
                for item2 in item1:
                    f2.write(str(item2) + " ")
                    
                f2.write("\n")
            f2.close()
            
            # pointCloud = keyFrame.pointCloud
            # poseCToW = keyFrame.frameSet.rgbFrame.cameraPose.getCameraToWorldMatrix()
            pointCloud = getKeyFramePointCloud(keyFrame)
            

            posefilename = args.output + "/PointCloud/" + f'{index}' + ".ply"
            o3d.io.write_point_cloud(posefilename, pointCloud)

            # up += cameraToWorld[0:3,1]
            # frame = {
            #     "file_path": "images/" + f'{index}' + ".png",
            #     "sharpness": sharpness(newImgName),
            #     "transform_matrix": cameraToWorld,
            #     "depth_file_path": "depth/" + f'{index}' + ".png",
            # }
            # frames.append(frame)
            index += 1

            
        
        # centerPos, scale = resizeToUnitCube(frames)

        # for f in frames: f["transform_matrix"] = f["transform_matrix"].tolist()

        # if frameWidth < 0 or frameHeight < 0: raise Exception("Unable get image dimensions, zero images received?")

        # fl_x = intrinsics[0][0]
        # fl_y = intrinsics[1][1]
        # cx = intrinsics[0][2]
        # cy = intrinsics[1][2]
        # angle_x = math.atan(frameWidth / (fl_x * 2)) * 2
        # angle_y = math.atan(frameHeight / (fl_y * 2)) * 2

        # transformationsJson = {
        #     "center": centerPos.tolist(),
        #     "scale": scale,
        #     "camera_angle_x": angle_x,
        #     "camera_angle_y": angle_y,
        #     "fl_x": fl_x,
        #     "fl_y": fl_y,
        #     "k1": 0.0,
        #     "k2": 0.0,
        #     "p1": 0.0,
        #     "p2": 0.0,
        #     "cx": cx,
        #     "cy": cy,
        #     "w": frameWidth,
        #     "h": frameHeight,
        #     "aabb_scale": args.scale,
        #     "frames": frames
        # }

        # with open(args.output + "/transformations.json", "w") as outFile:
        #     json.dump(transformationsJson, outFile, indent=2)


def main():
    os.makedirs(args.output + "/images", exist_ok=True)
    os.makedirs(args.output + "/tmp_images", exist_ok=True)
    os.makedirs(args.output + "/tmp_depth", exist_ok=True)
    os.makedirs(args.output + "/depth", exist_ok=True)
    os.makedirs(args.output + "/pose", exist_ok=True)
    os.makedirs(args.output + "/PointCloud/", exist_ok= True)
    print("Processing")
    replay = spectacularAI.Replay(args.input, onMappingOutput)

    replay.runReplay()

    shutil.rmtree(args.output + "/tmp_images")
    shutil.rmtree(args.output + "/tmp_depth")

    print("Done!")
    print("")
    print("You can now run instant-ngp nerfs using following command:")
    print("")
    print("    ./build/testbed --mode nerf --scene {}/transformations.json".format(args.output))


if __name__ == '__main__':
    main()