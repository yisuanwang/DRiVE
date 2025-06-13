# [CVPR 2025] DRiVE: Diffusion-based Rigging Empowers Generation of Versatile and Expressive Characters

## 🔧 Dependencies and Installation
We have tested DRiVE on both A800 and A100 GPUs, and it runs successfully on both.


```
git clone https://github.com/yisuanwang/DRiVE.git
cd DRiVE

conda create -n drive python=3.10

# check your nvcc -V for cuda version

# tested on A800
# torch                       2.4.1+cu124
# torchaudio                  2.4.1+cu124
# torchvision                 0.19.1+cu124
pip install torch==2.4.1 torchvision torchaudio  xformers --index-url https://download.pytorch.org/whl/cu124

# tested on A100
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118 
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

cd sub
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
pip install git+https://github.com/NVlabs/nvdiffrast
```



## prepare [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
If you need to generate a T-pose anime image from text or image inputs, you should use [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
If you already have a normalized T-pose image or a 3DGS avatar ready for rigging and skinning, you can skip this step.



```
cd sub
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

bash webui.sh
```

## prepare pretrained weights
Please first edit the $sdwebui_path and $hf_token settings in the file.
```
bash ./src/scripts/download_ckpt.sh
```
If you plan to run the real-human Any-pose → T-pose pipeline, you’ll need to download the [Realism Engine SDXL](https://civitai.com/models/152525/realism-engine-sdxl) checkpoint.
We use the weight file [realismEngineSDXL_v30VAE](https://civitai.com/api/download/models/293240?type=Model&format=SafeTensor&size=pruned&fp=fp16), place it in here:
```
[4.0K]  ./DRiVE/sub/stable-diffusion-webui/models/Stable-diffusion
├── [ 377]  Put Stable Diffusion checkpoints here.txt
├── [6.5G]  real2avatar-xl-3.1-0504.safetensors
└── [6.5G]  realismEngineSDXL_v30VAE.safetensors
```


## prepare [controlnet](https://github.com/Mikubill/sd-webui-controlnet?tab=readme-ov-file#installation)
download weights:\
[ip-adapter-plus_sdxl_vit-h](https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models) \
[OpenPoseXL2](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/tree/main)

