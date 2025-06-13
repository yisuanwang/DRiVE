<h2> 
<a href="https://driveavatar.github.io/" target="_blank">DRiVE: Diffusion-based Rigging Empowers Generation of Versatile and Expressive Characters</a>
</h2>

This is the official PyTorch implementation of the following publication:


## 💻 Requirements
The code has been tested on:
- Ubuntu 20.04
- CUDA 12.1
- Python 3.10.12
- Pytorch 2.2.2

## 🔧 Installation
This package can be installed with pip via pip install path/to/this/repository and used as import gecco_torch. Use pip install -e path/to/this/repository if you want your changes in this repository to be immediately reflected in import locations, otherwise you need to re-install the package after each modification.

## 🚅 Pretrained model
We provide pretrained checkpoints [here]

The pretrained models of [CLIP](https://github.com/openai/CLIP) can be downloaded from hugging face and We use CLIP-ViT-H-14-laion2B-s32B-b79K in our experiments.

## 🔦 Demo (Joints Generation from 3DGS)
Try DRiVE using the following commands:
```
python example_configs/infer_joints.py
```
Then, you should obtain:
- ```results/gaussian_pcd/id.ply```: the input 3DGS points;
- ```results/joints_pcd/id.ply```: the generated joints from the 3DGS.


## 🔦 ToDo List
- [x] Joints generation
- [ ] Bone connectivity and skinning geneation
- [ ] Training code


## 💡 Citation
If you find this repo helpful, please give us a 😍 star 😍.
Please consider citing DRiVE if this program benefits your project
```
@inproceedings{sun2025drive,
  title={Drive: Diffusion-based rigging empowers generation of versatile and expressive characters},
  author={Sun, Mingze and Chen, Junhao and Dong, Junting and Chen, Yurun and Jiang, Xinyu and Mao, Shiwei and Jiang, Puhua and Wang, Jingbo and Dai, Bo and Huang, Ruqi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={21170--21180},
  year={2025}
}
```

## 🔗 Related Projects
We sincerely thank the excellent open-source projects:
- [GECCO](https://github.com/cvlab-epfl/gecco/) for the wonderful point cloud generation quality;
- [RigNet](https://github.com/zhan-xu/RigNet) for the wonderful rigging ability;
