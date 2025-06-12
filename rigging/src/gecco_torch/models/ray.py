"""
RayNetwork is a wrapper around SetTransformer that handles the projective lookup of CNN
features and integrates them in the network inputs. The CNN feature computation is handled by the Diffusion object -
this class only handles the lookup.

For unconditional modelling you should use gecco_torch.models.LinearLift instead, which simply embeds the geometry and passes it
through the SetTransformer.
"""
import torch
from torch import nn, Tensor
from einops import rearrange
from kornia.geometry.camera.perspective import project_points

from gecco_torch.reparam import Reparam
from gecco_torch.structs import Context3d
from gecco_torch.models.feature_pyramid import FeaturePyramidContext
from gecco_torch.models.set_transformer import SetTransformer
from gecco_torch.models.dgcnn import DecoderSimpleDGCNN


class GroupNormBNC(nn.GroupNorm):
    """
    A GroupNorm that supports batch channel last format (transformer default).
    """

    def forward(self, tensor_bnc: Tensor) -> Tensor:
        assert tensor_bnc.ndim == 3

        tensor_bcn = rearrange(tensor_bnc, "b n c -> b c n")
        result_bcn = super().forward(tensor_bcn)
        return rearrange(result_bcn, "b c n -> b n c")
    
    
class RayNetwork(nn.Module):
    def __init__(
        self,
        backbone: SetTransformer,
        reparam: Reparam,
        context_dims: list[int],
    ):
        """
        Args:
            backbone: The SetTransformer to use.
            reparam: The reparameterization scheme to use.
            context_dims: The number of channels in each CNN feature map.
        """
        super().__init__()
        self.backbone = backbone
        self.reparam = reparam
        self.context_dims = context_dims

        self.xyz_embed = nn.Linear(reparam.dim, backbone.feature_dim)
        self.img_feature_proj = nn.Sequential(nn.Linear(1024, backbone.feature_dim),
                                              torch.nn.LayerNorm(backbone.feature_dim)
                                             )
        
        self.output_proj = nn.Sequential(
            GroupNormBNC(16, backbone.feature_dim, affine=False),
            nn.Linear(backbone.feature_dim, reparam.dim),
        )
        
        self.pcd_encoder = DecoderSimpleDGCNN()
        self.gaussian_feature_proj = nn.Linear(128, backbone.feature_dim)
        

    def extra_repr(self) -> str:
        return f"context_dims={self.context_dims}"

    def extract_image_features(
        self,
        geometry_diffusion: Tensor,
        features: list[Tensor],
        ctx: Context3d,
    ) -> Tensor:
        # the input geometry is in diffusion (reparameterized) space, so we need to convert it to data space
        geometry_data = self.reparam.diffusion_to_data(geometry_diffusion, ctx)

        # project the geometry to the image plane
        hw_01 = project_points(geometry_data, ctx.K.unsqueeze(1))[..., :2]
        hw_flat = rearrange(hw_01, "b n t -> b n 1 t")
        
        # perform the projective lookup on each feature map
        lookups = []
        for feature in features:
            lookup = torch.nn.functional.grid_sample(
                feature, hw_flat * 2 - 1, align_corners=False
            )
            lookup = rearrange(lookup, "b c n 1 -> b n c")
            lookups.append(lookup)

        # concatenate the lookups
        return torch.cat(lookups, dim=-1)
    
    
    def extract_gaussian_features(
        self,
        ctx: Context3d,
        geometry_diffusion: Tensor,
        features: Tensor,
        k: Tensor,
    ) -> Tensor:
        # the input geometry is in diffusion (reparameterized) space, so we need to convert it to data space
        geometry_data = self.reparam.diffusion_to_data(geometry_diffusion, ctx)
        
        dists = torch.cdist(geometry_data, ctx.gaussian_pcd)  # Shape: (batch, K, N)
        dists, indices = torch.topk(dists, k, dim=-1, largest=False, sorted=True)  # Shape: (batch, K, k)
        knn_features = torch.gather(features.unsqueeze(1).expand(-1, geometry_data.size(1), -1, -1), 2, indices.unsqueeze(-1).expand(-1, -1, -1, features.size(-1)))  # Shape: (batch, K, k, 128)
        weights = 1.0 / (dists + 1e-8)  # Shape: (batch, K, k)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize weights
        interpolated_features = torch.sum(weights.unsqueeze(-1) * knn_features, dim=-2)  # Shape: (batch, K, 128)

        return interpolated_features
    

    def forward(
        self,
        geometry: Tensor,
        t: Tensor,
        raw_ctx: Context3d,
        post_context: FeaturePyramidContext,
        do_cache: bool = False,
        cache: list[Tensor] | None = None,
    ) -> tuple[Tensor | list[Tensor] | None]:
        # embed the geometry and the time. Since we use Gaussian activations, we can just use a single linear layer
        xyz_features = self.xyz_embed(geometry)
        t_features = t

        with torch.autocast(device_type="cuda", enabled=False):
            features_f32 = [f.to(dtype=torch.float32) for f in post_context.features]            
            gaussian_features_raw_f32 = self.pcd_encoder(raw_ctx.gaussian_pcd.transpose(1, 2)).to(dtype=torch.float32) 
            gaussian_features_f32 = self.extract_gaussian_features(raw_ctx, geometry, gaussian_features_raw_f32, k=10)
            
        img_features = self.img_feature_proj(features_f32[0])
        gaussian_features = self.gaussian_feature_proj(gaussian_features_f32)
        
        img_features = img_features.repeat(1, xyz_features.shape[1], 1)

        point_features = xyz_features + gaussian_features #+ img_features
        # point_features = xyz_features + img_features
        
        
        # run the SetTransformer
        processed, out_cache = self.backbone(
            point_features, gaussian_features, t_features, do_cache, cache
        )

        return self.output_proj(processed), out_cache
