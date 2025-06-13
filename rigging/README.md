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
Try VistaDream using the following commands:
```
python infer_joints.py
```
Then, you should obtain:
- ```results/gaussian_pcd/id.ply```: the input 3DGS points;
- ```results/joints_pcd/id.ply```: the generated joints from the 3DGS.


## 🔦 ToDo List
- [x] Joints generation.
- [ ] Bone connectivity and skinning geneation
- [ ] Training code


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
- [GECCO](https://github.com/cvlab-epfl/gecco/) for the wonderful point cloud generation quality;
- [RigNet](https://github.com/zhan-xu/RigNet) for the wonderful rigging ability;
