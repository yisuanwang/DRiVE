import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import gecco_torch
from gecco_torch.diffusion import Diffusion
from gecco_torch.structs import Example
from gecco_torch.utils import save_pcd_as_ply, symmetric_point_cloud_torch, remove_points_near_origin, sort_point_cloud



def sample_joints(config_root, config_name, ckpt_name, joints_num):
    config = gecco_torch.load_config(f'{config_root}/{config_name}') # load the model definition
    model: Diffusion = config.model

    state_dict = torch.load(f'{config_root}/{ckpt_name}', map_location='cpu')

    model.load_state_dict(state_dict['ema_state_dict'])
    model = model.eval()

    data = config.data
    data.setup() # PyTorch lightning data modules need to be setup before use
    batch: Example = next(iter(data.val_dataloader()))
    print(batch) # print the batch to see what it contains

    # find out the best backend
    if torch.cuda.is_available():
        map_device = lambda x: x.to(device='cuda')
    else:
        map_device = lambda x: x

    model: Diffusion = map_device(model).eval()
    context = batch.ctx.apply_to_tensors(map_device)

    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.float16):
            samples = model.sample_stochastic(
                            (29, joints_num, 3),
                            context=context,
                            with_pbar=True,
                        )

    return samples, batch, context
        

if __name__ == "__main__":   
    to_pil_image = transforms.ToPILImage()            
    config_root = '/cpfs04/user/sunzeming/project/cvpr25/DRiVE/example_configs/'
    config_name = 'train_joints.py'
    # common_ckpt_path = 'lightning_logs/common_joints_global_3d/checkpoints/epoch=147-val_loss=51.811.ckpt'
    # special_ckpt_path = 'lightning_logs/special_joints_global_3d/checkpoints/epoch=148-val_loss=63.883.ckpt'
    common_ckpt_path = 'lightning_logs/common_joints_cross_att_3d_2d/checkpoints/epoch=148-val_loss=49.486.ckpt'
    special_ckpt_path = 'lightning_logs/special_joints_cross_att_3d_2d/checkpoints/epoch=168-val_loss=61.023.ckpt'
    
    common_samples, batch, context = sample_joints(config_root, config_name, common_ckpt_path, 25)
    special_samples, _, _ = sample_joints(config_root, config_name, special_ckpt_path, 75)
    
    # Save
    for i in tqdm(range(common_samples.shape[0])):
        image = context.image[i]
        gaussian_pcd = context.gaussian_pcd[i]
        shape_name = int(context.shape_name[i].detach().cpu().numpy())

        common_pcd_sample = symmetric_point_cloud_torch(common_samples[i]).float()
        common_pcd_sample = sort_point_cloud(common_pcd_sample)
        
        special_pcd_sample = special_samples[i].float()
        special_pcd_sample = symmetric_point_cloud_torch(special_pcd_sample)
        # try:
        #     special_pcd_sample = symmetric_point_cloud_torch(special_pcd_sample)
        # except:
        #     special_pcd_sample = special_pcd_sample
        special_pcd_sample = sort_point_cloud(special_pcd_sample)
        
        special_pcd_sample, mask = remove_points_near_origin(special_pcd_sample)
        
        pcd_sample = torch.cat([common_pcd_sample, special_pcd_sample], dim=0)

        save_pcd_as_ply(pcd_sample.detach().cpu().numpy(), os.path.join(config_root, 'results', 'open_source/joints_pcd/{:d}.ply'.format(shape_name)))
        save_pcd_as_ply(common_pcd_sample.detach().cpu().numpy(), os.path.join(config_root, 'results', 'open_source/joints_pcd/{:d}_common.ply'.format(shape_name)))
        save_pcd_as_ply(gaussian_pcd.detach().cpu().numpy(), os.path.join(config_root, 'results', 'open_source/gaussian_pcd/{:d}.ply'.format(shape_name)))



