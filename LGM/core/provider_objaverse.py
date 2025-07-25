import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

from PIL import Image
import json
import tarfile

DEBUG = False

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)




class ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training
        
        # Replace this placeholder with actual code to load your dataset
        self.dataset_dir = self.opt.dataset_dir
        self.items = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]


        # TODO: remove this barrier
        # self._warn()

        # TODO: load the list of objects for training
        dataset_list_path = self.opt.dataset_list_path
        self.items = []
        with open(dataset_list_path, 'r') as f:
            for line in f.readlines():
                # print('line', line )
                self.items.append(line.strip())

        print('self.items.__len__()=',self.items.__len__()) 
        
        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        # print('len(self.items)=',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print('in __getitem__=', idx)
        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # # TODO: choose views, based on your rendering settings
        if self.training:
            # input views are in (36, 72), other views are randomly selected
            # vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
            
            
            vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
            # print('vids 1 = ', vids)
            
            
        else:
            # fixed views
            vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()
            # print('vids 2 = ', vids)


        # #######################################################
        # # 自定义训练, 本地数据集, 从Gobj导入的相机参数
        for vid in vids:

            try:
                image_path = os.path.join(self.dataset_dir, uid, f'{vid:03d}.png')
                camera_path = os.path.join(self.dataset_dir, uid, f'{vid:03d}_camera.json')
                # print('image_path=', image_path)
                with open(image_path, 'rb') as f:
                    image = np.frombuffer(f.read(), np.uint8)

                # print('meta_path=', camera_path)
                with open(camera_path, 'r') as f:
                    meta = json.load(f)

                image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                
                camera_matrix = torch.eye(4)
                camera_matrix[:3, 0] = torch.tensor(meta["x"])
                camera_matrix[:3, 1] = -torch.tensor(meta["y"])
                camera_matrix[:3, 2] = -torch.tensor(meta["z"])
                camera_matrix[:3, 3] = torch.tensor(meta["origin"])
                c2w = camera_matrix
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]

            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg

            image = image[[2,1,0]].contiguous() # bgr to rgb

            # normal = normal.permute(2, 0, 1) # [3, 512, 512]
            # normal = normal * mask # to [0, 0, 0] bg

            image = TF.resize(image, (512, 512))
            mask = TF.resize(mask, (512, 512))
            
            images.append(image)
            # normals.append(normal)
            # depths.append(depth)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            print(len(images))
            images = images + [images[-1]] * n
            # normals = normals + [normals[-1]] * n
            # depths = depths + [depths[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, 3, H, W]
        # normals = torch.stack(normals, dim=0) # [V, 3, H, W]
        # depths = torch.stack(depths, dim=0) # [V, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.opt.cam_radius / radius
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        
        #######################################################
        
        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results


class Head_ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training
        
        # Replace this placeholder with actual code to load your dataset
        self.dataset_dir = self.opt.dataset_dir
        self.items = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]


        # TODO: remove this barrier
        # self._warn()

        # TODO: load the list of objects for training
        dataset_list_path = self.opt.dataset_list_path
        self.items = []
        with open(dataset_list_path, 'r') as f:
            for line in f.readlines():
                # print('line', line )
                self.items.append(line.strip())

        print(self.items.__len__()) 
        
        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        # print('len(self.items)=',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print('in __getitem__=', idx)
        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # # TODO: choose views, based on your rendering settings
        if self.training:
            # input views are in (36, 72), other views are randomly selected
            vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(200).tolist()
            # print('vids 1 = ', vids)
        else:
            # fixed views
            vids = np.arange(36, 73, 4).tolist() + np.arange(200).tolist()
            # print('vids 2 = ', vids)
        

        # #######################################################
        # # 自定义训练, 本地数据集, 从Gobj导入的相机参数
        for vid in vids:

            try:
                # image_path = os.path.join(self.dataset_dir, uid, f'{vid:03d}.png')
                # camera_path = os.path.join(self.dataset_dir, uid, f'{vid:03d}_camera.json')
                # print('image_path=', image_path)
                # with open(image_path) as f:
                #     image = np.frombuffer(f.read(), np.uint8)

                
                # print('meta_path=', camera_path)
                # with open(camera_path, 'r') as f:
                #         meta = json.load(f)

                image_path = os.path.join(self.dataset_dir, uid, f'{vid:03d}.png')
                camera_path = os.path.join(self.dataset_dir, uid, f'{vid:03d}_camera.json')
                # print('image_path=', image_path)
                with open(image_path, 'rb') as f:
                    image = np.frombuffer(f.read(), np.uint8)
                    image = image_data.reshape((1024, 1024, 4))  # 重塑为三维数组
                    upper_half = image[:512, :, :]  # 选择上半部分512行
                    # 由于整个宽度是1024，我们需要从256开始取512宽，结束于768（256 + 512）
                    center_part = upper_half[:, 256:768, :]
                    image = image

                with open(camera_path, 'r') as f:
                    meta = json.load(f)

                image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                
                camera_matrix = torch.eye(4)
                camera_matrix[:3, 0] = torch.tensor(meta["x"])
                camera_matrix[:3, 1] = -torch.tensor(meta["y"])
                camera_matrix[:3, 2] = -torch.tensor(meta["z"])
                camera_matrix[:3, 3] = torch.tensor(meta["origin"])
                c2w = camera_matrix
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]

            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg

            image = image[[2,1,0]].contiguous() # bgr to rgb

            # normal = normal.permute(2, 0, 1) # [3, 512, 512]
            # normal = normal * mask # to [0, 0, 0] bg

            images.append(image)
            # normals.append(normal)
            # depths.append(depth)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            print(len(images))
            images = images + [images[-1]] * n
            # normals = normals + [normals[-1]] * n
            # depths = depths + [depths[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, 3, H, W]
        # normals = torch.stack(normals, dim=0) # [V, 3, H, W]
        # depths = torch.stack(depths, dim=0) # [V, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.opt.cam_radius / radius
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        
        
        
        
        #######################################################
        
        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results


class Ori_ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        # Replace this placeholder with actual code to load your dataset
        self.dataset_dir = self.opt.dataset_dir
        self.items = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]

        # TODO: remove this barrier
        # self._warn()

        # TODO: load the list of objects for training
        dataset_list_path = self.opt.dataset_list_path
        self.items = []
        with open(dataset_list_path, 'r') as f:
            for line in f.readlines():
                # print('line', line )
                self.items.append(line.strip())

        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        if self.training:
            # input views are in (36, 72), other views are randomly selected
            vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
            
            print(f'self.training vids = {vids}')
        else:
            # fixed views
            vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()
        
        for vid in vids:
            try:
                # # TODO: load data (modify self.client here)
                # image = np.frombuffer(self.client.get(image_path), np.uint8)
                # image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                # c2w = [float(t) for t in self.client.get(camera_path).decode().strip().split(' ')]
                # c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
                
                # inside your dataset's item retrieval method:
                image_path = os.path.join(self.dataset_dir, uid, f'{vid:03d}.png')
                camera_path = os.path.join(self.dataset_dir, uid, f'{vid:03d}_camera.json')

                # Load and decode the image
                with open(image_path, 'rb') as f:
                    image = np.frombuffer(f.read(), np.uint8)
                # image_binary = self.client.get(image_path)  # Get the binary content of the image
                # image_np = np.frombuffer(image_binary, np.uint8)  # Convert binary to numpy array
                # image_tensor = torch.from_numpy(cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255)  # Convert numpy array to tensor
                with open(camera_path, 'r') as f:
                    camera_data = json.load(f)
                # # Load and decode the camera parameters
                # camera_json = self.client.get(camera_path).decode()  # Assuming this returns a string of JSON content
                # camera_data = json.loads(camera_json)  # Convert JSON string to Python dictionary


            except Exception as e:
                print(f'[Exception] dataset {uid} {vid}: {e}')
                continue
            

            # If the JSON represents a flattened 4x4 matrix
            if isinstance(camera_data, list) and len(camera_data) == 16:
                c2w = torch.tensor(camera_data, dtype=torch.float32).reshape(4, 4)
            # If the JSON represents structured camera parameters (and you need to convert it)
            # Here, you would need additional code to construct the matrix from components
            else:
                # Placeholder for conversion logic from structured format to 4x4 matrix
                c2w = torch.eye(4)  # Replace this with actual conversion logic
    
            # # TODO: you may have a different camera system
            # # blender world + opencv cam --> opengl world & cam
            # c2w[1] *= -1
            # c2w[[1, 2]] = c2w[[2, 1]]
            # c2w[:3, 1:3] *= -1 # invert up and forward direction

            # # scale up radius to fully use the [-1, 1]^3 space!
            # c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            print(len(images))
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results
    

class G_ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        # TODO: remove this barrier
        # self._warn()

        # TODO: load the list of objects for training
        dataset_list_path = self.opt.dataset_list_path

        self.items = []
        # with open(dataset_list_path, 'r') as f:
        #     self.items = [line.strip() for line in f.readlines()]

        with open(dataset_list_path, 'r') as f:
            for line in f.readlines():
                # print('line', line )
                self.items.append(line.strip())

        # print(self.__len__())
        
        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        # print('len(self.items)=',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print('in __getitem__=', idx)
        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # # # TODO: choose views, based on your rendering settings
        # if self.training:
        #     # input views are in (36, 72), other views are randomly selected
        #     vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
        #     # print('vids 1 = ', vids)
        # else:
        #     # fixed views
        #     vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()
        #     # print('vids 2 = ', vids)
        
        
        # 从Gobj的0-39个view里选择
        if self.training:
            # input views are in (36, 72), other views are randomly selected
            vids = np.random.permutation(np.arange(0, 40))[:self.opt.num_input_views].tolist() + np.random.permutation(40).tolist()
            if DEBUG: print('vids 1 = ', vids)
        else:
            # fixed views
            vids = np.arange(0, 40, 4).tolist() + np.arange(40).tolist()
            if DEBUG: print('vids 2 = ', vids)
            
        
        
        #######################################
        # 作者在gitthub上给的代码, 对应G obj的可以正常train LGM
        
        if DEBUG: print('vids=', vids)
        if DEBUG: print('uid=', uid)
        
        for vid in vids:
            
            try:
                tar_path = os.path.join(self.opt.dataset_dir, f"{uid}.tar")
                if DEBUG: print('tar_path=', tar_path)
                
                uid_last = uid.split('/')[1]
                
                if DEBUG: print('uid_last=',uid_last)

                image_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}.png")
                meta_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}.json")
                # albedo_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_albedo.png") # black bg...
                # mr_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_mr.png")
                # nd_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_nd.exr")
                
                with tarfile.open(tar_path, 'r') as tar:
                    with tar.extractfile(image_path) as f:
                        image = np.frombuffer(f.read(), np.uint8)
                    with tar.extractfile(meta_path) as f:
                        meta = json.loads(f.read().decode())
                    # with tar.extractfile(nd_path) as f:
                    #     nd = np.frombuffer(f.read(), np.uint8)

                image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]

                c2w = np.eye(4)
                c2w[:3, 0] = np.array(meta['x'])
                c2w[:3, 1] = np.array(meta['y'])
                c2w[:3, 2] = np.array(meta['z'])
                c2w[:3, 3] = np.array(meta['origin'])
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
                
                # nd = cv2.imdecode(nd, cv2.IMREAD_UNCHANGED).astype(np.float32) # [512, 512, 4] in [-1, 1]
                # normal = nd[..., :3] # in [-1, 1], bg is [0, 0, 1]
                # depth = nd[..., 3] # in [0, +?), bg is 0

                # # rectify normal directions
                # normal = normal[..., ::-1]
                # normal[..., 0] *= -1
                # normal = torch.from_numpy(normal.astype(np.float32)).nan_to_num_(0) # there are nans in gt normal... 
                # depth = torch.from_numpy(depth.astype(np.float32)).nan_to_num_(0)
                
            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]

            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg

            image = image[[2,1,0]].contiguous() # bgr to rgb

            # normal = normal.permute(2, 0, 1) # [3, 512, 512]
            # normal = normal * mask # to [0, 0, 0] bg

            images.append(image)
            # normals.append(normal)
            # depths.append(depth)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if DEBUG: print('images len=', len(images))
        
        
        if vid_cnt < self.opt.num_views:
            if DEBUG: print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            
            if not images:  # Check if the list is empty
                # Handle the empty case, maybe by adding a default image or skipping
                default_image = np.zeros((3, 224, 224))  # Example: Replace with appropriate default
                images = [default_image] * n  # Replace n with the desired number of images
            else:
                images = images + [images[-1]] * n


            # images = images + [images[-1]] * n
            
            
            
            # normals = normals + [normals[-1]] * n
            # depths = depths + [depths[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, 3, H, W]
        # normals = torch.stack(normals, dim=0) # [V, 3, H, W]
        # depths = torch.stack(depths, dim=0) # [V, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.opt.cam_radius / radius
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        
        # rotate normal!
        # V, _, H, W = normal_final.shape # [1, 3, h, w]
        # normal_final = (transform[:3, :3].unsqueeze(0) @ normal_final.permute(0, 2, 3, 1).reshape(-1, 3, 1)).reshape(V, H, W, 3).permute(0, 3, 1, 2).contiguous()

        #######################################################
        
        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results