"""
Definition of reparameterization schemes. Each reparameterization scheme
implements a `data_to_diffusion` and `diffusion_to_data` method, which
convert the data between "viewable" representation (simple xyz) and the
"diffusion" representation (possibly normalized, logarithmically spaced, etc).
"""
import torch
from torch import Tensor
from kornia.geometry.camera.perspective import project_points, unproject_points

from gecco_torch.structs import Context3d


class Reparam(torch.nn.Module):
    """
    Base class for reparameterization schemes.
    """

    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim

    def data_to_diffusion(self, data: Tensor, ctx: Context3d) -> Tensor:
        raise NotImplementedError()

    def diffusion_to_data(self, diff: Tensor, ctx: Context3d) -> Tensor:
        raise NotImplementedError()


class NoReparam(Reparam):
    """
    A placeholder for non-conditional models.
    """

    def data_to_diffusion(self, data: Tensor, ctx: Context3d) -> Tensor:
        return data

    def diffusion_to_data(self, diff: Tensor, ctx: Context3d) -> Tensor:
        return diff


class GaussianReparam(Reparam):
    """
    Maps data to diffusion space by normalizing it with the mean and standard deviation.
    """

    def __init__(self, mean: Tensor, sigma: Tensor):
        assert mean.ndim == 1
        assert mean.shape == sigma.shape

        super().__init__(mean.shape[0])

        self.register_buffer("mean", mean)
        self.register_buffer("sigma", sigma)

    def data_to_diffusion(self, data: Tensor, ctx: Context3d) -> Tensor:
        del ctx
        return (data - self.mean) / self.sigma

    def diffusion_to_data(self, diff: Tensor, ctx: Context3d) -> Tensor:
        del ctx
        return diff * self.sigma + self.mean

    def extra_repr(self) -> str:
        return f"mean={self.mean.flatten().tolist()}, sigma={self.sigma.flatten().tolist()}"


class UVLReparam(Reparam):
    """
    Maps data to diffusion space by projecting it to the image plane and
        1. taking the logarithm of the depth to map from [0, inf] to [-inf, inf]
        2. using the inverse hyperbolic tangent on projection coordinates (h, w)
           to map them from [0, 1] to [-inf, inf]
    and finally normalizing the result with the mean and standard deviation (like GaussianReparam).

    This allows for modelling point clouds constrained to the viewing frustum. Since a point exactly
    at an image border would have infinite inverse tanh, we relax the constraint a bit by mapping the
    projection coordinates to as if they were in [0, logit_scale] instead of [0, 1].
    """

    def __init__(self, mean: Tensor, sigma: Tensor, logit_scale: float = 1.1):
        assert mean.shape == (3,)
        assert sigma.shape == (3,)

        super().__init__(dim=3)

        self.register_buffer("uvl_mean", mean)
        self.register_buffer("uvl_std", sigma)
        self.logit_scale = logit_scale

    depth_to_real = staticmethod(torch.log)
    real_to_depth = staticmethod(torch.exp)

    def extra_repr(self) -> str:
        return (
            f"uvl_mean={self.uvl_mean.flatten().tolist()}, "
            f"uvl_std={self.uvl_std.flatten().tolist()}, "
            f"logit_scale={self.logit_scale}"
        )

    def _real_to_01(self, r: Tensor) -> Tensor:
        """
        Maps a real number (reparametrized projection height, width) to [0, 1]
        """
        s = torch.tanh(r)
        s = s * self.logit_scale
        s = s + 1.0
        s = s / 2
        return s

    def _01_to_real(self, s: Tensor) -> Tensor:
        """
        Maps a number in [0, 1] (projection height, width) to a real number
        """
        s = 2 * s
        s = s - 1.0
        s = s / self.logit_scale
        r = torch.arctanh(s)
        return r

    def xyz_to_hwd(self, xyz: Tensor, ctx: Context3d) -> Tensor:
        """
        Maps a point cloud to (image_plane, depth).
        """
        hw = project_points(xyz, ctx.K.unsqueeze(1))
        d = torch.linalg.norm(xyz, dim=-1, keepdim=True)

        return torch.cat([hw, d], dim=-1)

    def hwd_to_xyz(self, hwd: Tensor, ctx: Context3d) -> Tensor:
        """
        Maps (image_plane, depth) to a point cloud.
        """
        hw, d = hwd[..., :2], hwd[..., 2:]
        xyz = unproject_points(hw, d, ctx.K.unsqueeze(1), normalize=True)
        return xyz

    def hwd_to_uvl(self, hwd: Tensor) -> Tensor:
        """
        Maps (image_plane, depth) to R^3 (including normalization).
        """
        assert hwd.shape[-1] == 3

        h, w, d = hwd.unbind(-1)

        uvl_denorm = torch.stack(
            [
                self._01_to_real(h),
                self._01_to_real(w),
                self.depth_to_real(d),
            ],
            dim=-1,
        )

        uvl_norm = (uvl_denorm - self.uvl_mean) / self.uvl_std
        return uvl_norm

    def uvl_to_hwd(self, uvl: Tensor) -> Tensor:
        """
        Maps R^3 (including normalization) to (image_plane, depth).
        """
        assert uvl.shape[-1] == 3

        uvl_denorm = uvl * self.uvl_std + self.uvl_mean
        u, v, l = uvl_denorm.unbind(-1)

        hwd = torch.stack(
            [
                self._real_to_01(u),
                self._real_to_01(v),
                self.real_to_depth(l),
            ],
            dim=-1,
        )

        return hwd

    def data_to_diffusion(self, data: Tensor, ctx: Context3d) -> Tensor:
        """
        Maps a point cloud to R^3 (including normalization).
        """
        assert isinstance(ctx, Context3d)

        xyz = data
        hwd = self.xyz_to_hwd(xyz, ctx)
        uvl = self.hwd_to_uvl(hwd)

        return uvl

    def diffusion_to_data(self, diff: Tensor, ctx: Context3d) -> Tensor:
        """
        Maps R^3 (including normalization) to a point cloud.
        """
        assert isinstance(ctx, Context3d)

        uvl = diff
        hwd = self.uvl_to_hwd(uvl)
        xyz = self.hwd_to_xyz(hwd, ctx)

        return xyz
