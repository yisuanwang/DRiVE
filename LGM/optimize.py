
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
import cv2
from PIL import Image

import torchvision
import torchvision.transforms as transforms

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models import LGM
# from mvdream.pipeline_mvdream import MVDreamPipeline

def chamfer_distance(p1, p2):
    """
    Compute the Chamfer Distance between two point clouds.

    Args:
    p1 (torch.Tensor): Tensor of shape [B1, N, 3], first point cloud.
    p2 (torch.Tensor): Tensor of shape [B2, M, 3], second point cloud.

    Returns:
    float: The Chamfer Distance between the two point clouds.
    """
    # Expand point clouds to [B1, N, 1, 3] and [1, B2, M, 3]
    p1 = p1.unsqueeze(2)
    p2 = p2.unsqueeze(0)

    # Compute squared distances [B1, N, B2, M]
    distances = torch.sum((p1 - p2) ** 2, dim=-1)

    # Find the closest points and compute the distances for each point set
    min_distances_p1_to_p2 = torch.min(distances, dim=2)[0]  # Closest from p1 to p2
    min_distances_p2_to_p1 = torch.min(distances, dim=1)[0]  # Closest from p2 to p1

    # Compute the Chamfer Distance
    loss = torch.mean(min_distances_p1_to_p2) + torch.mean(min_distances_p2_to_p1)

    return loss


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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')
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

def rgba2rgb(img):
    new_img = np.ones([img.shape[0], img.shape[1], 3]) * 255
    mask = img[..., -1] > 255//2
    new_img[mask] = img[mask][:, :3]
    new_img = new_img.astype(np.uint8)    

    return new_img

# process function
def process(opt: Options, path, part):
    name = os.path.splitext(os.path.basename(path))[0]
    name = name + f'_{part}'
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)

    input_image = kiui.read_image(path, mode='uint8')
    imgs = sorted(os.listdir(os.path.dirname(path)))
    imgs = [i for i in imgs if not i.endswith('front_right.png')]
    imgs = [i for i in imgs if not i.endswith('front_left.png')]
    imgs = [i for i in imgs if not i.endswith('back_left.png')]
    imgs = [i for i in imgs if not i.endswith('back_right.png')]
    imgs = imgs[0+part*4:4+part*4]
    
    
    
    image_0 = rgba2rgb(cv2.imread(os.path.join(os.path.dirname(path), imgs[3]), cv2.IMREAD_UNCHANGED)) / 255.0
    image_1 = rgba2rgb(cv2.imread(os.path.join(os.path.dirname(path), imgs[1]), cv2.IMREAD_UNCHANGED)) / 255.0
    image_2 = rgba2rgb(cv2.imread(os.path.join(os.path.dirname(path), imgs[2]), cv2.IMREAD_UNCHANGED)) / 255.0
    image_3 = rgba2rgb(cv2.imread(os.path.join(os.path.dirname(path), imgs[0]), cv2.IMREAD_UNCHANGED)) / 255.0


    # # recenter
    mask = image_0[..., -1] > -1
    image_0 = recenter(image_0, mask, border_ratio=0.15)
    image_1 = recenter(image_1, mask, border_ratio=0.15)
    image_2 = recenter(image_2, mask, border_ratio=0.15)
    image_3 = recenter(image_3, mask, border_ratio=0.15)    
    mv_image = [image_0, image_1, image_2, image_3]

    # # bg removal
    # carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
    # mask = carved_image[..., -1] > 0


    
    # # generate mv
    # image = image.astype(np.float32) / 255.0

    # # rgba to rgb white bg
    # if image.shape[-1] == 4:
    #     image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])

    # mv_image = pipe('', image, guidance_scale=5.0, num_inference_steps=30, elevation=0)
    # save mv images
    cv2.imwrite(os.path.join(opt.workspace, name + '_0.png'), (mv_image[0] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(opt.workspace, name + '_1.png'), (mv_image[1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(opt.workspace, name + '_2.png'), (mv_image[2] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(opt.workspace, name + '_3.png'), (mv_image[3] * 255).astype(np.uint8))
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32

    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        import ipdb; ipdb.set_trace(context=10)
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

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
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

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(os.path.join(opt.workspace, name + '.mp4'), images[..., ::-1], fps=30)

import argparse


def save_and_render(gau, ply_save_path):
    # 保存合并后的点云为本地文件 去除脑袋的body
    pos = gau[..., 0:3] # [B, N, 3]
    opacity = gau[..., 3:4]
    scale = gau[..., 4:7]
    rotation = gau[..., 7:11]
    rgbs = gau[..., 11:]
    gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
    gaussians_with_extra_dim = gaussians.unsqueeze(0)  # 添加一个额外的维度
    model.gs.save_ply(gaussians_with_extra_dim, ply_save_path)

    print('save ply finish, render 360 video...')
    # render 360 video 
    images = []
    elevation = 0

    gaussians = gaussians_with_extra_dim
    gaussians = gaussians.to(device)
    # print(f'gaussians = {gaussians}')
    
    # 渲染图像
    azimuth = np.arange(0, 720, 5, dtype=np.int32)
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1


    for azi in tqdm.tqdm(azimuth):
        cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view = cam_view.to(device)  # 将cam_view移动到GPU
        proj_matrix = proj_matrix.to(device)  # 将proj_matrix也移动到GPU
        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        scale = 1
        image = model.gs.render1024(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
        images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

    images = np.concatenate(images, axis=0)
    imageio.mimwrite(ply_save_path.replace('.ply', '.mp4'), images, fps=30)



def main():
    # parser = argparse.ArgumentParser(description='Optimize the ICP process.')
    # parser.add_argument('--head-ply-path', type=str, required=True, help='Path to the head PLY file.')
    # parser.add_argument('--body-ply-path', type=str, required=True, help='Path to the body PLY file.')
    # parser.add_argument('--full-ply-path', type=str, required=True, help='Path to the full body PLY file.')
    # parser.add_argument('--full-image', type=str, required=True, help='Path to the full image.')
    # parser.add_argument('--out-path', type=str, required=True, help='Output path for the optimized results.')
    # args = parser.parse_args()

    # read initial 3D GS
    path = opt.workspace
    # body_gs = model.gs.load_ply(os.path.join(path, '/mnt/chenjh/Animatabler/ICP/full_gs_combined.ply')).to(device)
    body_gs = model.gs.load_ply(opt.body_ply_path).to(device)

    top_gs = model.gs.load_ply(opt.head_ply_path).to(device)
    # bottom_gs = model.gs.load_ply(os.path.join(path, 'part03_rgb_000_back_left_2.ply')).to(device)
    # shoes_gs = model.gs.load_ply(os.path.join(path, 'part03_rgb_000_back_left_3.ply')).to(device)
    full_gs = model.gs.load_ply(opt.full_ply_path).to(device)
    # define variables to optimize
    top_gs_translation = nn.Parameter( torch.tensor([0.0, 0.4, 0.0], device=device))
    # bottom_gs_translation = nn.Parameter(torch.zeros(3, device=device))
    # shoes_gs_translation = nn.Parameter(torch.tensor([0, body_gs[:,:3].min(dim=0)[0][1].item(), 0], device=device))
    scale_value = 320 / 1024
    top_gs_scale = nn.Parameter(torch.tensor([scale_value], device=device))
    # bottom_gs_scale = nn.Parameter(torch.ones(1, device=device))
    # shoes_gs_scale = nn.Parameter(0.25*torch.ones(1, device=device))

    # define optimizer
    # optimizer = torch.optim.Adam([top_gs_translation, bottom_gs_translation, shoes_gs_translation, top_gs_scale, bottom_gs_scale, shoes_gs_scale], lr=1e-2)
    optimizer = torch.optim.Adam([top_gs_translation, top_gs_scale], lr=1e-2)


    def deform_gs(gs, translation, scale):
        gs_new = gs.clone()
        gs_new[:, :3] = gs[:, :3] * scale + translation
        gs_new[:, 4:7] = gs[:, 4:7] * scale
        return gs_new

    elevation = 0
    azimuth = np.arange(0, 30, 30, dtype=np.int32)
    batch_cam_view = []
    batch_cam_view_proj = []
    batch_cam_pos = []
    for azi in azimuth:
        
        cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        batch_cam_view.append(cam_view)
        batch_cam_view_proj.append(cam_view_proj)
        batch_cam_pos.append(cam_pos)

    batch_cam_view = torch.stack(batch_cam_view)
    batch_cam_view_proj = torch.stack(batch_cam_view_proj)
    batch_cam_pos = torch.stack(batch_cam_pos)


    #stage 1:
    # for i in range(1000):
    #     optimizer.zero_grad()
    #     # construct full gs
    #     top_gs_new = deform_gs(top_gs, top_gs_translation, top_gs_scale)

    #     # calculate loss
    #     loss = chamfer_distance(top_gs_new.unsqueeze(0), body_gs.unsqueeze(0))
    #     loss.backward()
    #     optimizer.step()

    
    image_full = Image.open(opt.full_image)
    if image_full.mode != 'RGB':
        image_full = image_full.convert('RGB')

    # Define the transformation without normalization
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()  # No normalization step here
    ])
    image_full = transform(image_full).unsqueeze(0).to(device)

    image_full = image_full[:, [2, 1, 0], :, :]

    # optimize
    for i in range(300):
        optimizer.zero_grad()
        # construct full gs
        top_gs_new = deform_gs(top_gs, top_gs_translation, top_gs_scale)
        full_gs_combined = torch.cat([body_gs, top_gs_new], dim=0)

        # render 360 video 
        elevation = 0
        azimuth = np.arange(0, 360, 30, dtype=np.int32)
        image_combine = model.gs.render1024(full_gs_combined.repeat(batch_cam_view.shape[0], 1, 1), batch_cam_view, batch_cam_view_proj, batch_cam_pos, scale_modifier=1)['image']
        image_combine = image_combine.squeeze(1)  # Adjust this as necessary based on your output
        image_combine = image_combine[:, [2, 1, 0], :, :]

        # print(f'image_full shape = {image_full.shape}')
        
        # print(f'image_combine = {image_combine}')
        # print(f'image_full = {image_full}')
        
        # N = 512
        # image_full[:, :, :N, :N] = image_combine[:, :, :N, :N]
        
        # print(f'after image_combine = {image_combine}')
        # print(f'after image_full = {image_full}')

        # Ensure tensors are on the CPU for processing with item() if needed
        # image_combine_max = image_combine.max().item()
        # image_combine_min = image_combine.min().item()

        # image_full_max = image_full.max().item()
        # image_full_min = image_full.min().item()

        # Print the maximum and minimum values
        # print(f"image_combine - Max: {image_combine_max}, Min: {image_combine_min}")
        # print(f"image_full - Max: {image_full_max}, Min: {image_full_min}")


        # if image_full.mode != 'RGB':
        # image_full = image_full.convert('BGR')
        # transform = torchvision.transforms.Compose([
        #     torchvision.transforms.ToTensor(),  # Converts to a tensor and scales to [0, 1]
        #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],  # These are typical ImageNet mean/std
        #                                     std=[0.229, 0.224, 0.225])
        # ])
        # image_full = transform(image_full).unsqueeze(0).to(device)  # Adds a batch dimension at the beginning

        # print(f'image_full shape ={image_full.shape}')

        # calculate loss
        loss = F.mse_loss(image_combine, image_full)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'iter {i}, loss: {loss.item()}')
            # images = (image_combine.detach().squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
            # full_images = (image_full.detach().squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
            
            images = (image_combine.detach().squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
            full_images = (image_full.detach().squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
            
            # images = (image_combine.detach().squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
            # full_images = (image_full.detach().squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)

            # print(f'in save: images shape = {images.shape}')
            
            # print(f'in save: full_images shape = {full_images.shape}')

            save_img_path = os.path.join(opt.out_path, 'images')
            os.makedirs(save_img_path, exist_ok=True)
            two_img = np.concatenate([images[0], full_images[0]], axis=1)
            imageio.imwrite(os.path.join(save_img_path, f'image_{i}.jpg'), two_img[..., ::-1])
            
    # 保存ply
    full_gs_combined = full_gs_combined.unsqueeze(0)  # 添加一个额外的维度
    model.gs.save_ply(full_gs_combined, f'{opt.out_path}/full_gs_combined.ply')

    # save_and_render(full_gs_combined, f'{opt.out_path}/full_gs_combined.ply')

    # two_img = np.concatenate([images, full_images], axis=2)
    # imageio.mimwrite(os.path.join(save_img_path, f'render.mp4'), two_img[..., ::-1], fps=30)

if __name__ == '__main__':
    main()
    # 	CUDA_VISIBLE_DEVICES=5 python optimize.py big
