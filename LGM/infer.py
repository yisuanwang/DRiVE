
import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import random
import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models import LGM

from sv3d.scripts.sampling.sv3d_pipeline import sv3d_pipe


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

opt = tyro.cli(AllConfigs)

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')

# device
device = torch.device('cuda')
model = model.half().to(device)
model.eval()

rays_embeddings = model.prepare_default_rays(device)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# load rembg
bg_remover = rembg.new_session()

import matplotlib.pyplot as plt
import cv2
import numpy as np

def replace_black_background_with_white(image):
    # 创建一个掩膜，将像素值都小于10的像素标记为真
    mask = np.all(image < 10, axis=-1)

    # 将满足掩膜条件的像素替换为白色
    image[mask] = [255, 255, 255]

    return image


# process function
def process(opt: Options, path):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)


    mv_image0 = sv3d_pipe(input_path = path, version='sv3d_p', seed=random.randint(0, 999999))
    

    for i, img in enumerate(mv_image0):
        out_path = os.path.join(opt.workspace, f'2-sv3d_{i}.png')
        imageio.imwrite(out_path, mv_image0[i], format='PNG')
        

    # 1100050704346030754
    # os.path.join(opt.workspace, f'2-sv3d_{i}.png')
    image_paths = [
        os.path.join(opt.workspace, f'2-sv3d_{0}.png'),
        os.path.join(opt.workspace, f'2-sv3d_{1}.png'),
        os.path.join(opt.workspace, f'2-sv3d_{2}.png'),
        os.path.join(opt.workspace, f'2-sv3d_{3}.png')
    ]
    
    
    
    mv_image0 = []


    for path in image_paths:
        
        img = imageio.imread(path)
        
        img_float32 = img.astype(np.float32)
        
        mv_image0.append(img_float32)
    
    
    
    
    mv_image = []
    
    for img in mv_image0:
        # img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Use IMREAD_UNCHANGED to ensure alpha channel is read if present
        if img is not None:
            print(img.shape)

            if img.shape[-1] == 4:  # Assuming the image has 4 channels, the last being the alpha
                # Create a white background
                background = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
                # Extract the alpha channel and convert it to a 3 channel mask
                alpha_channel = img[:, :, 3] / 255.0  # Normalize alpha values to range 0 to 1
                alpha_channel = np.stack([alpha_channel]*3, axis=-1)  # Expand alpha to 3 channels
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img = (img * alpha_channel + background * (1 - alpha_channel)).astype(np.uint8)
                
            img = cv2.resize(img, (512, 512))
            img = img.astype(np.float32) / 255.0
            mv_image.append(img)
            
    
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    
    
    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians
        model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '.ply'))

        # render 360 video 
        images = []
        elevation = 0

        if opt.fancy_video:

            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = min(azi / 360, 1)

                image = model.gs.render1024(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model.gs.render1024(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(os.path.join(opt.workspace, name + '.mp4'), images, fps=30)


assert opt.test_path is not None
if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
else:
    file_paths = [opt.test_path]
for path in file_paths:
    process(opt, path)

