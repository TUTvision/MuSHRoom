import os
import math
import json
import numpy as np

item_list = ["activity", "classroom",  "computer", "honka", "koivu", "kokko", "olohuone", "sauna", "vr_room" ]
for item in item_list:
    path = "room_datasets/{}/iphone_old/long_capture/".format(item)
    transformations = json.load(open(os.path.join(path, "transformations.json")))
    frames = transformations["frames"]

    num_frames = len(frames)
    i_all = np.arange(num_frames)

    # read test.txt
    with open(os.path.join("room_datasets/{}/iphone_old/long_capture/".format(item), "test.txt"), "r") as f:
        lines = f.readlines()
    i_eval = [int(num.split("\n")[0]) for num in lines]




    frame_name = [frame["file_path"].split("/")[-1] for frame in frames]

    test_frame_name = [frame_name[i].split(".")[0] for i in i_eval]

    # save test_frame_name to test.txt  

    with open(os.path.join("room_datasets/{}/iphone/long_capture/".format(item), "test.txt"), "w") as f:
        for i in test_frame_name:
            f.write(str(i) + "\n")


# with open(os.path.join("room_datasets/activity/iphone/long_capture/", "test.txt"), "r") as f:
#     lines = f.readlines()