<div align="center">

# [CVPR 2025] [DRiVE: Diffusion-based Rigging Empowers Generation of Versatile and Expressive Characters](https://driveavatar.github.io/)

🔥🔥🔥DRiVE has been accepted by CVPR 2025, See you in Tennessee, 🇺🇸USA 🔥🔥🔥
---

**[Mingze Sun](https://scholar.google.com/citations?user=TTW2mVoAAAAJ&hl=en)<sup>1\*</sup>**, **[Junhao Chen](https://scholar.google.com/citations?hl=en&user=uVMnzPMAAAAJ)<sup>1\*</sup>**, **[Junting Dong](https://scholar.google.com/citations?user=dEzL5pAAAAAJ&hl=en)<sup>2†</sup>**, **[Yurun Chen](https://scholar.google.com/citations?user=k8fKlQ0AAAAJ&hl=en&oi=sra)<sup>1</sup>**, **[Xinyu Jiang](https://scholar.google.com/citations?user=njfKRXQAAAAJ&hl=en)<sup>1</sup>**, **[Shiwei Mao](https://openreview.net/profile?id=~Shiwei_Mao1)<sup>1</sup>**, 

**[Puhua Jiang](https://scholar.google.com/citations?user=E-k3WcgAAAAJ&hl=en)<sup>1,3</sup>**, **[Jingbo Wang](https://scholar.google.com/citations?user=GStTsxAAAAAJ&hl=en)<sup>2</sup>**, **[Bo Dai](https://scholar.google.com/citations?user=KNWTvgEAAAAJ&hl=en)<sup>2</sup>**, **[Ruqi Huang](https://scholar.google.com/citations?user=cgRY63gAAAAJ&hl=en)<sup>1†</sup>**



<sup>1</sup> Tsinghua University  <sup>2</sup> Shanghai AI Laboratory  <sup>3</sup> PengCheng Laboratory



<!-- [![hf_space](https://img.shields.io/badge/🤗-LeaderBoard-blue.svg)](xxx)
[![hf_space](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](xxx)
[![hf_space](https://img.shields.io/badge/🤗-Online_Demo-yellow.svg)](xxx) -->
<!-- [![Home Page](static/images/homepage.svg)](https://dancetog.github.io/)  -->
<!-- <!-- [![Dataset](https://img.shields.io/badge/Dataset-PairFS_4K-green)](xxx) -->
<!-- [![Dataset](https://img.shields.io/badge/Dataset-DanceTogEval_100-green)](xxx)
[![Dataset Download](https://img.shields.io/badge/Benchmark-TogetherVideoBench-red)](xxx) --> 


<a href='https://driveavatar.github.io/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://openaccess.thecvf.com/content/CVPR2025/html/Sun_DRiVE_Diffusion-based_Rigging_Empowers_Generation_of_Versatile_and_Expressive_Characters_CVPR_2025_paper.html'><img src='https://img.shields.io/badge/CVPR-HTML-yellow'></a>
<a href='https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_DRiVE_Diffusion-based_Rigging_Empowers_Generation_of_Versatile_and_Expressive_Characters_CVPR_2025_paper.pdf'><img src='https://img.shields.io/badge/CVPR-PDF-red'></a>
<a href='https://openaccess.thecvf.com/content/CVPR2025/supplemental/Sun_DRiVE_Diffusion-based_Rigging_CVPR_2025_supplemental.pdf'><img src='https://img.shields.io/badge/CVPR-Supp.Mat.-golden'></a>


[![Dataset](https://img.shields.io/badge/Dataset-AnimeRig-green)](https://huggingface.co/yisuanwang/DRiVE)
[![Pretrained Weights](https://img.shields.io/badge/Pretrained_Weights-DRiVE-green)](https://huggingface.co/yisuanwang/DRiVE)
[![arXiv](https://img.shields.io/badge/Arxiv-2411.17423-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.17423)
[![github](https://img.shields.io/github/stars/yisuanwang/DRiVE.svg?style=social)](https://github.com/yisuanwang/DRiVE/) 
![visitors](https://visitor-badge.laobi.icu/badge?page_id=yisuanwang.DRiVE&left_color=green&right_color=red)

![teaser](assets/ezgif-5-ba5ef716cf.gif)

</div>

## Todo
✅ 1. Our fine-tuned SDXL pipeline for Text2Anime and Image2Anime. \
✅ 2. Our fine-tuned LGM with SV3D. \
✅ 3. Inference code for generating anime avatar 3DGS from text, image inputs. \
✅ 4. The code for generating skeleton bindings and skinning weights for 3DGS. \
⚪️ 5. The AnimeRig dataset, which contains nearly 10,000 3D meshes and 3DGS, along with their corresponding skeleton riggings and skinning.

## ⚙️ Install
See [install.md](docs/install.md).

## 🧑‍💻 Run
<!-- See [run.md](docs/run.md). -->
```bash ./scripts/runpipe2.sh```


## 📂 Citation
```bibtex
@InProceedings{Sun_2025_CVPR,
    author    = {Sun, Mingze and Chen, Junhao and Dong, Junting and Chen, Yurun and Jiang, Xinyu and Mao, Shiwei and Jiang, Puhua and Wang, Jingbo and Dai, Bo and Huang, Ruqi},
    title     = {DRiVE: Diffusion-based Rigging Empowers Generation of Versatile and Expressive Characters},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {21170-21180}
}
```


## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yisuanwang/DRiVE&type=Date)](https://star-history.com/#yisuanwang/DRiVE&Date)

## Acknowledgement
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [LGM](https://github.com/3DTopia/LGM)
- [RigNet](https://github.com/zhan-xu/RigNet)
- [CharacterGen](https://github.com/zjp-shadow/CharacterGen)
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [animagine-xl-3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1)
- [blender](https://www.blender.org/)
- [blip2-flan-t5-xxl](https://huggingface.co/Salesforce/blip2-flan-t5-xxl)
- [segformer-b5-finetuned-human-parsing](https://matei-dorian/segformer-b5-finetuned-human-parsing)
- [neural-blend-shapes](https://github.com/PeizhuoLi/neural-blend-shapes)
- [anime-segmentation](https://github.com/SkyTNT/anime-segmentation)