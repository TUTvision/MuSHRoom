import torch
from rasterizer.rasterizer import rasterize
from zbuf.dataset import ScanDataset
import os
import numpy as np
import time
import cv2
import configargparse

if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    
    parser.add_argument("--datadir", type=str, help='data directory')
    # parser.add_argument("--interp", help='interplate or not', action="store_true")
    parser.add_argument(
    "--pose_scale", help="use training data or all data to rescale pose",
    )
    parser.add_argument("--radius", type=float, default=0.015, help='the radius of points when rasterizing')
    parser.add_argument("--H", type=int, default=720)
    parser.add_argument("--W", type=int, default=1280)
    parser.add_argument("--interp_num", type=str, default = "0", help='data directory')
    args = parser.parse_args()
    
    train_set = ScanDataset(args, args.pose_scale, 'rasterize')


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)

    
    begin = time.time()

    pc = train_set.get_pc()


    # train set 
    if int(args.interp_num) > 0:
        os.makedirs(os.path.join(args.datadir, "interplated_depth_" + args.interp_num), exist_ok=True)
    else:
        os.makedirs(os.path.join(args.datadir, "depth_zbuf_{}".format(args.pose_scale)), exist_ok=True)

    for i, batch in enumerate(train_loader):
        pose = batch['w2c'][0]
        xyz_ndc = pc.get_ndc(pose)
        id, zbuf = rasterize(xyz_ndc, (args.H, args.W), args.radius)
        
        zbuf_render = (zbuf.cpu().numpy())[0,:,:,:]
        zbuf_render[zbuf_render<0] = 0

        zbuf_render = (zbuf_render * 1000).astype(np.uint16)

        # zbuf_render = cv2.applyColorMap(cv2.convertScaleAbs(zbuf_render/1000, alpha=15),cv2.COLORMAP_PINK)          
        # cv2.imwrite("test.png", zbuf_render)
        # exit()

        if int(args.interp_num) > 0:
            cv2.imwrite(os.path.join(args.datadir, "interplated_depth_{}/{}.png".format(args.interp_num, str(i))), zbuf_render)
        else:
            cv2.imwrite(os.path.join(args.datadir, "depth_zbuf_{}/{}.png".format(args.pose_scale, str(i))), zbuf_render)

        
        
        if i % 20 == 0:
            print('train', i)


    end = time.time()
    print(f'time cost: {end-begin} s')
    

