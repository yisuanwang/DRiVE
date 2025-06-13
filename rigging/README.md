<h2> 
<a href="https://driveavatar.github.io/" target="_blank">DRiVE: Diffusion-based Rigging Empowers Generation of Versatile and Expressive Characters</a>
</h2>

This is the official PyTorch implementation of the following publication:


## 💻 Requirements
The code has been tested on:
- Ubuntu 20.04
- CUDA 12.3
- Python 3.10.12
- Pytorch 2.1.0
- GeForce RTX 4090.

## 🔧 Installation
For complete installation instructions, please see [INSTALL.md](INSTALL.md).

## 🚅 Pretrained model
VistaDream is training-free but utilizes pretrained models of several existing projects.
To download pretrained models for [Fooocus](https://github.com/lllyasviel/Fooocus), [Depth-Pro](https://github.com/apple/ml-depth-pro), 
[OneFormer](https://github.com/SHI-Labs/OneFormer), [SD-LCM](https://github.com/luosiallen/latent-consistency-model), run the following command:
```
bash download_weights.sh
```
The pretrained models of [LLaVA](https://github.com/haotian-liu/LLaVA) and [Stable Diffusion-1.5](https://github.com/CompVis/stable-diffusion) will be automatically downloaded from hugging face on the first running.

## 🔦 Demo (Single-View Generation)
Try VistaDream using the following commands:
```
python demo.py
```
Then, you should obtain:
- ```data/sd_readingroom/scene.pth```: the generated gaussian field;
- ```data/sd_readingroom/video_rgb(dpt).mp4```: the rgb(dpt) renderings from the scene.

## 🔦 Demo (Sparse-View Generation)
To use sparse views as input as [demo_here](https://github.com/WHU-USI3DV/VistaDream/issues/14), we need [Dust3r](https://github.com/naver/dust3r) to first reconstruct the input images to 3D as the scaffold (no zoom-out needed).

First download Dust3r [checkpoints](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) and place it at ```tools/Dust3r/checkpoints``` by the following command:
```
wget -P tools/Dust3r/checkpoints https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```
Then try VistaDream with sparse images as input using the following commands:
```
python demo_sparse.py
```
Then, you should obtain:
- ```data/bedroom/scene.pth```: the generated gaussian field;
- ```data/bedroom/video_rgb(dpt).mp4```: the rgb(dpt) renderings from the scene.

## 🔦 Generate your own scene (Single or Sparse views as input)
If you need to improve the reconstruction quality of your own images, please refer to [INSTRUCT.md](pipe/cfgs/INSTRUCT.md)

To visualize the generated gaussian field, you can use the following script:
```
import torch
from ops.utils import save_ply
scene = torch.load(f'data/vistadream/piano/refine.scene.pth')
save_ply(scene,'gf.ply')
```
and feed the ```gf.ply``` to [SuperSplat](https://playcanvas.com/supersplat/editor) for visualization.

## 🔦 ToDo List
- [x] Early check in generation.
- [x] Support more types of camera trajectory. Please follow [Here](ops/trajs/TRAJECTORY.MD) to define your trajectory. An example is given in this [issue](https://github.com/WHU-USI3DV/VistaDream/issues/11).
- [x] Support sparse-view-input (and no pose needed). An example is given in this [issue](https://github.com/WHU-USI3DV/VistaDream/issues/14).
- [ ] Interactive Demo.

## 💡 Citation
If you find this repo helpful, please give us a 😍 star 😍.
Please consider citing VistaDream if this program benefits your project
```
@article{wang2024vistadream,
  title={VistaDream: Sampling multiview consistent images for single-view scene reconstruction},
  author={Haiping Wang and Yuan Liu and Ziwei Liu and Zhen Dong and Wenping Wang and Bisheng Yang},
  journal={arXiv preprint arXiv:2410.16892},
  year={2024}
}
```

## 🔗 Related Projects
We sincerely thank the excellent open-source projects:
- [Fooocus](https://github.com/lllyasviel/Fooocus) for the wonderful inpainting quality;
- [LLaVA](https://github.com/haotian-liu/LLaVA) for the wonderful image analysis and QA ability;
- [Depth-Pro](https://github.com/apple/ml-depth-pro) for the wonderful monocular metric depth estimation accuracy;
- [OneFormer](https://github.com/SHI-Labs/OneFormer) for the wonderful sky segmentation accuracy;
- [StableDiffusion](https://github.com/CompVis/stable-diffusion) for the wonderful image generation/optimization ability.
