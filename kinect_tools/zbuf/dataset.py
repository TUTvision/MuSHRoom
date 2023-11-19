import torch
import torch.utils.data as data
import os
import numpy as np
import json
from PIL import Image
from .dataset_utils import load_pc, get_rays, PointCloud
from torchvision import transforms as T

class ScanDataset(data.Dataset):

    @staticmethod
    def CameraRead(dir, line_num):
        camera_para = []
        f = open(dir)
        for i in range(line_num):
            line = f.readline()
            tmp = line.split()
            camera_para.append(tmp)
        camera_para = np.array(camera_para, dtype=np.float32)
        f.close()
        return torch.tensor(camera_para, dtype=torch.float32)

    def __init__(self, args, split, mode):
        self.img_size = (args.H, args.W)
        self.device = "cuda"
        self.mode = mode
        self.id_list = []
        self.args = args
        self.pose_scale = args.pose_scale
        
        if int(args.interp_num) > 0:
            pose_path = os.path.join(args.datadir, 'interplated_pose_' + args.pose_scale + "_" + args.interp_num)
        else:
            pose_path = os.path.join(args.datadir, 'pose')
        total = len(os.listdir(pose_path))
            
        
        train_id = list(range(total))
        self.id_list = [str(k) for k in train_id]
            
        print(self.id_list)

        intrinsic_path = os.path.join(args.datadir, 'intrinsic', 'intrinsic_color.txt')
        intrinsic = self.CameraRead(intrinsic_path, 4)[:3,:3]
        

        scale1_c = (args.W - 1) / (self.img_size[1] - 1)
        scale2_c = (args.H - 1) / (self.img_size[0] - 1)

        intrinsic[0:1, :] = intrinsic[0:1, :] / scale1_c
        intrinsic[1:2, :] = intrinsic[1:2, :] / scale2_c
        self.intrinsic = intrinsic

        
        self.transform = T.ToTensor()
        self.img_list = []
        self.w2c_list = []
 
        for idx in self.id_list:
        # load pose
            c2w = self.CameraRead(os.path.join(pose_path, idx + '.txt'), 4)
            w2c = torch.inverse(c2w)
            self.w2c_list.append(w2c)

        


    def get_pc(self):
        print(os.path.join(self.args.datadir, "pointcloud_{}.ply".format(self.pose_scale)))
        pc_xyz = load_pc(os.path.join(self.args.datadir, "pointcloud_{}.ply".format(self.pose_scale)), self.device, down=4)  # n,3
        pc = PointCloud(pc_xyz, self.intrinsic, self.device, (self.args.W, self.args.H))
        return pc


    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        """
        Returns:
            data dict {"img.rgb": rgb (H W C),
                       "img.mask": mask (H,W 1),
                       "camera_mat": camera_mat (4,4)
        """
        pass
        idx = idx % self.__len__()
        id = self.id_list[idx]
        w2c = self.w2c_list[idx]
      

        return {"idx": str(idx).rjust(3,'0'),
                "file_id": id,
                "w2c": w2c.to(self.device)}

