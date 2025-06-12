"""
Wraps a pretrained feature extractor to produce a feature pyramid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn, Tensor
import torchvision.models as tvm
from transformers import CLIPVisionModelWithProjection

from gecco_torch.structs import Context3d


@dataclass
class FeaturePyramidContext:
    features: list[Tensor]
    K: Tensor


class FeaturePyramidExtractor(nn.Module):
    def forward(self, ctx_raw: Context3d) -> FeaturePyramidContext:
        raise NotImplementedError()


class ConvNeXtExtractor(FeaturePyramidExtractor):
    def __init__(
        self,
        n_stages: int = 3,
        model: Literal["tiny", "small"] = "tiny",
        pretrained: bool = True,
    ):
        super().__init__()
        
        image_encoder_path = '/cpfs04/user/sunzeming/project/cvpr25/DRiVE/CLIP/CLIP-ViT-H-14-laion2B-s32B-b79K/'
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        self.image_encoder.requires_grad_(False)

    def _remove_stochastic_depth(self):
        """We found SD to harm generative performance"""
        for submodule in self.modules():
            if isinstance(submodule, tvm.convnext.CNBlock):
                submodule.stochastic_depth = torch.nn.Identity()

    def forward(self, raw_ctx: Context3d) -> FeaturePyramidContext:
        images = raw_ctx.image

        features = []
        with torch.no_grad():
            image_embeds = self.image_encoder(images).image_embeds # patch/class
            features.append(image_embeds.unsqueeze(1))
        
        return FeaturePyramidContext(
            features=features,
            K=raw_ctx.K,
        )
