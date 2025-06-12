import os
import re
from typing import Tuple, List
import numpy as np
import glob
from PIL import Image
from tqdm.auto import tqdm
from scipy.spatial import cKDTree
import imageio as iio
import torch
from torchvision import transforms
import lightning.pytorch as pl
import timm
import potpourri3d as pp3d

from transformers import CLIPImageProcessor

from gecco_torch.structs import Context3d, Example
from gecco_torch.data.samplers import FixedSampler


class Building:
    def __init__(
        self,
        pcd_path: str,
        rgb_path: str,
        input_type: str,
        batch_size: int = 48,
    ):
        self.data_list = []
        self.pcd_path = pcd_path
        self.rgb_path = rgb_path
        self.input_type = input_type
        self.max_num_verts = 10000
        
        if self.input_type == 'mesh':        
            self.raw_v_filelist = glob.glob(os.path.join(self.pcd_path, '*_ori.obj'))
        elif self.input_type == 'pcd':
            self.raw_v_filelist = glob.glob(os.path.join(self.pcd_path, '*.ply'))
        else:
            self.raw_v_filelist = glob.glob(os.path.join(self.pcd_path, '*_v.txt'))
        
        self.clip_image_processor = CLIPImageProcessor()
        self.process()

    
    def farthest_point_sample(self, xyz, npoint):
        device = xyz.device
        if len(xyz.shape) < 3:
            xyz = xyz.unsqueeze(0)
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.full((B, N), 1e10, dtype=torch.float32, device=device)
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.norm(xyz - centroid, dim=2, p=2) ** 2
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids
    
    
    def process(self):
        self.K = np.array([[1422.2222222222222, 0, 512],
                         [0, 2133.3333333333335, 512],
                         [0, 0, 1]]).astype(np.float32)
        
        i = 0
        
        if len(self.raw_v_filelist) > 36:            
            self.raw_v_filelist = self.raw_v_filelist[:36]
        
        for v_filename in tqdm(self.raw_v_filelist):
            
            if self.input_type == 'mesh':        
                v, f = pp3d.read_mesh(v_filename)
                shape_name = v_filename.split('/')[-1].split('_')[0]  
            elif self.input_type == 'pcd':
                v = pp3d.read_point_cloud(v_filename)
                shape_name = v_filename.split('/')[-1].split('.')[0]  
            else:
                v = np.loadtxt(v_filename)
                shape_name = v_filename.split('/')[-1].split('_')[0]      
      
            # image_path = os.path.join(self.rgb_path, f'{shape_name}.png')
            image_path = os.path.join(self.rgb_path, f'{shape_name}', '001.png')

            # CLIP single image
            raw_image = Image.open(image_path)
            image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values[0]
    
            shape_name = int(shape_name)

            v = torch.from_numpy(v).float()
            if v.shape[0] > self.max_num_verts:
                fps_idx = self.farthest_point_sample(v[:, 0:3].cuda(), self.max_num_verts).squeeze().detach().cpu()
                v = v[fps_idx]            
            
            if v.shape[0] == 10000:
                i += 1
            else:
                print(shape_name)
                continue
            
            self.data_list.append({
                    'pointcloud': v[:, 0:3],
                    'name': shape_name,
                    'image': image,
                    'camera': self.K,
                })
        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image = self.data_list[index]['image']
        K = self.data_list[index]['camera']
        
        gaussian_pcd = self.data_list[index]['pointcloud'].detach().cpu().numpy()
        shape_name = np.array(self.data_list[index]['name'])
        
        gaussian_pcd = gaussian_pcd.astype(np.float32)

        return Example(
            data=gaussian_pcd,
            ctx=Context3d(
                image=image,
                K=K,
                gaussian_pcd=gaussian_pcd,
                shape_name=shape_name,
            ),
        )
    

class Taskonomy(torch.utils.data.ConcatDataset):
    def __init__(self, pcd_path: str, rgb_path: str, input_type: str):
        self.pcd_path = pcd_path
        self.rgb_path = rgb_path
        self.input_type = input_type
        
        buildings = []
        buildings.append(
            Building(self.pcd_path, self.rgb_path, self.input_type)
        )

        super().__init__(buildings)

    def __repr__(self):
        return f"Taskonomy(n_buildings={len(buildings)}, len={len(self)})"


class TaskonomyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_root: str,
        test_root: str,
        rgb_root: str = None,
        n_points: int = 100,
        batch_size: int = 48,
        num_workers: int = 8,
        epoch_size: int | None = 10_000,
        val_size: int | None = 10_000,
        input_type: str = None,
    ):
        super().__init__()

        self.train_root = train_root
        self.test_root = test_root
        self.rgb_root = rgb_root
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size
        self.val_size = val_size
        self.input_type = input_type

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train = Taskonomy(self.train_root, self.rgb_root, self.input_type)
            self.val = Taskonomy(self.test_root, self.rgb_root, self.input_type)
        elif stage == "test":
            self.test = Taskonomy(self.test_root, self.rgb_root, self.input_type)
        else:
            raise ValueError(f"Unknown stage {stage}")

    def val_dataloader(self):
        if self.val_size is None:
            sampler = None
        else:
            sampler = FixedSampler(self.val, length=self.val_size)

        return torch.utils.data.DataLoader(
            self.val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            sampler=sampler,
        )
